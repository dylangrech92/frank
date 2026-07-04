"""Core agent loop with four named extension hooks for lifecycle phases.

Implements ``handle_user_message`` as the main entry point that drives the
read-eval-print cycle between a Session transcript and an LLMClient, plus three
explicit hook no-op functions and an over-cap handling seam exposed inline in
the loop body.
"""

import json
import sys
import time
import compaction

from llm import ChatResponse, LLMClient, OverCapError, ToolCall
from session import Session
from tools.registry import dispatch, schemas
from tools.result import ToolResult


# =============================================================================
# Extension hooks — four named no-op seams for later phases
# =============================================================================

# Module-level list tracking every mutation event since last ``diagnostics_inject_summary`` call.
_TURN_MUTATIONS: list[dict] = []


def _mutate_tracker(event: dict) -> None:
    """Record a mutation event with its pre-mutation publish baseline.

    Args:
        event: A mutation event dict with at least ``kind`` and ``path`` keys.
    """
    record = dict(event)
    try:
        from diagnostics import STORE
        from lsp.manager import path_to_uri

        uri = path_to_uri(record.get("path", ""))
        record["uri"] = uri
        record["baseline"] = STORE.snapshot_counts([uri]).get(uri, 0)
    except Exception:
        pass
    _TURN_MUTATIONS.append(record)


# Per-session episodic-extraction watermark (row count already handed to the encoder).
_EPISODIC_WATERMARKS: dict[int, int] = {}


from tools import _sandbox

_sandbox.subscribe_mutations(_mutate_tracker)


def diagnostics_inject_summary(session: Session) -> None:
    """After a tool-mutation turn, append the LSP diagnostics summary via session.

    Imports are performed lazily inside the function to avoid circular module
    imports at load time.

    The flow is:
        1.  Take accumulated mutation events and clear the list.
        2.  For each event whose kind is ``created``, ``changed`` or ``renamed``,
            check that a running LSP server serves the file's language and that
            the URI is in ``MANAGER._open_docs`` (i.e. an open document).
            Collect all such URIs.
        3.  Poll STORE until every collected URI has appeared in the diagnostics
            store, up to a 2-second deadline.
        4.  Call ``STORE.summary()`` and, when non-None, append the formatted
            string to the last tool-result message via
            ``session.amend_last_tool_result``.

    Args:
        session: Active ``Session`` holding the conversation transcript.
    """
    try:
        from diagnostics import STORE
        from lsp.manager import path_to_uri
        import main as main_module

        events = list(_TURN_MUTATIONS)
        _TURN_MUTATIONS.clear()

        if not events:
            return

        manager = main_module.MANAGER
        seen_uris: set[str] = set()
        filtered_uris: list[str] = []
        baselines: dict[str, int] = {}

        for event in events:
            kind = event.get("kind")
            path_val = event.get("path", "")

            if kind not in ("created", "changed", "renamed"):
                continue

            if manager is None:
                break

            lang_id = manager.language_for_path(path_val)
            if lang_id is None:
                continue

            uri_str = event.get("uri") or path_to_uri(path_val)
            open_docs = getattr(manager, "_open_docs", {})

            if uri_str in seen_uris or uri_str not in open_docs:
                continue

            seen_uris.add(uri_str)
            filtered_uris.append(uri_str)
            baselines[uri_str] = event.get("baseline", 0)

        if filtered_uris:
            STORE.wait_for_publish(filtered_uris, baselines, 2.0)

        s = STORE.summary()
        if s is not None:
            session.amend_last_tool_result(s)

    except Exception as exc:  # noqa: E722
        print(f"diagnostics-inject-error: {exc}", file=sys.stderr, flush=True)


def flashback_maybe_seed(session: Session) -> None:
    """Turn-zero memory recall injection seam.

    Delegates to the flashback module, which decides -- via the terse and
    continuation gates -- whether to seed a compact recalled-memory bundle onto
    the session for this turn. The bundle (if any) is stashed on
    ``session._flashback_block`` and injected by the flashback context provider;
    it is never persisted to the transcript. Never raises.
    """
    try:
        import memory.flashback as flashback

        flashback.maybe_seed(session)
    except Exception:
        pass


def episodic_maybe_extract(session: Session, client: LLMClient) -> None:
    """End-of-turn episodic extraction: enqueue a transcript window for off-thread
    encoding once enough new rows have accumulated past this session's watermark.

    Non-blocking: it only measures the row count and hands a window to the episodic
    write queue, then returns immediately so the REPL prompt never stalls.
    """
    try:
        import memory.episodic as episodic
    except Exception:
        return

    messages = session._messages
    total = len(messages)
    prev = _EPISODIC_WATERMARKS.get(id(session), 0)
    new_rows = total - prev

    # Emit the outcome of any PRIOR extraction that has since completed.
    last = episodic.pop_last_run()
    if last is not None:
        print(
            f"episodic: last extraction ran={last.get('ran')} "
            f"stored={last.get('stored')} updated={last.get('updated')} "
            f"deleted={last.get('deleted')} reason={last.get('reason')}",
            file=sys.stderr,
        )

    if new_rows < episodic.GATE_N:
        print(
            f"episodic: gate skipped ({new_rows}/{episodic.GATE_N} new rows past watermark)",
            file=sys.stderr,
        )
        return

    # Advance the watermark and hand off the most-recent window for encoding.
    _EPISODIC_WATERMARKS[id(session)] = total
    start = max(0, total - episodic.EXTRACTION_WINDOW)
    window = [(i, messages[i]) for i in range(start, total)]
    try:
        episodic.enqueue(str(session.project_root), window, client)
        print(
            f"episodic: enqueued window rows {start}..{total - 1} "
            f"({new_rows} new past watermark)",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"episodic-enqueue-error: {exc}", file=sys.stderr)


# The fourth hook — over-cap handling — lives directly in the ``handle_user_message``
# loop body as the OverCapError branch described by the contract below.  Its docstring
# lives inline there and is noted under the return path.


# =============================================================================
# Helpers
# =============================================================================

def render_tool_result(name: str, result: ToolResult) -> str:
    """Render a ``ToolResult`` value as plain text for a tool message content slot.

    The rendered string follows this shape::

        [name(STATUS code=CODE)]       # error status with code
        [name(STATUS)]                  # success status, no code
        <body>                          # only when body is non-empty
        hint: <hint_text>               # only when a hint is present

    The bracket header always contains the tool name and status.  When status
    is ``'error'`` and ``result.code`` is set, the kebab-case code is appended
    inside the brackets as ``code=THECODE``.

    Args:
        name: Registered tool name (used in the header).
        result: A ``ToolResult`` instance to render.

    Returns:
        A plain-text string suitable for ``content`` on a ``tool`` role message.
    """
    lines: list[str] = []

    if result.status == "error" and result.code is not None:
        header = f"[{name}({result.status} code={result.code})]"
    else:
        header = f"[{name}({result.status})]"

    lines.append(header)

    body = result.body
    if body:
        if isinstance(body, (dict, list)):
            body_str_val = json.dumps(result.body)
        else:
            body_str_val = str(result.body)
        lines.append(body_str_val)

    hint_text = result.hint
    if hint_text is not None and hint_text:
        lines.append(f"hint: {hint_text}")

    return "\n".join(lines)


# =============================================================================
# Main entry point
# =============================================================================

def handle_user_message(
    text: str,
    session: Session,
    client: LLMClient,
    verbose: bool = False,
    compaction_cfg: dict | None = None,
) -> str:
    """Execute one agent turn in response to a user message.

    The loop follows this contract exactly:

    1. **setup** — append_user(text), flashback_maybe_seed(session).
    2. **loop** (while True):
       a. assemble_context() from session.
       b. verbose → print full assembled context to stderr prefixed by "assembled context".
       c. client.chat(context, schemas()) inside try … except OverCapError.
       d. **OverCapError path** — append_assistant with a message saying the context
          exceeded the model limit; return that same text (turn ends without retry).
          *A later phase replaces this behaviour with compact-then-retry.*
       e. **normal success** — append_assistant(text, tool_calls), then:
          i.  If no tool calls → episodic_maybe_extract(session) + return response.text.
          ii. For each tool call:
              - echo to stderr the call name and arguments.
              - dispatch via registry (dispatch(name, arguments)).
              - echo rendered result to stderr via render_tool_result().
              - store append_tool_result(call.id, call.name, rendered_result).
          iii. After all calls -> diagnostics_inject_summary(session); continue loop.

    Args:
        text: User message string to begin the turn with.
        session: Active ``Session`` holding the conversation transcript.
        client: An ``LLMClient`` instance for sending chat requests.
        verbose: If True, prints the full assembled context lines (prefixed by
            "assembled context") to stderr before each LLM call.

    Returns:
        The final assistant text string — either a normal turn response, an
        episodic extraction pass-through, or the over-cap sentinel message.
    """
    session.append_user(text)
    flashback_maybe_seed(session)

    window = client.config.context_limit
    comp_cfg = compaction_cfg or {}
    tool_schemas = schemas()
    cap = compaction.compute_cap(window, comp_cfg)
    max_compactions = 5
    compactions = 0

    def _over_cap_giveup() -> str:
        msg = (
            "Context is over the model's token budget and compaction could not "
            "reduce it further. Start a new session or shorten the request."
        )
        session.append_assistant(msg)
        return msg

    while True:
        # Pre-flight: keep the assembled request at or below the shared cap,
        # compacting older turns before the call is ever made.
        context = session.assemble_context()
        est = compaction.estimate_tokens(context, tool_schemas)
        print(
            f"context: {len(context)} messages, ~{est} tokens (cap {cap})",
            file=sys.stderr,
        )
        while est > cap:
            if compactions >= max_compactions or not compaction.compact(
                session, client, window, comp_cfg
            ):
                return _over_cap_giveup()
            compactions += 1
            context = session.assemble_context()
            est = compaction.estimate_tokens(context, tool_schemas)
            print(
                f"context: {len(context)} messages, ~{est} tokens "
                f"(cap {cap}) [post-compaction #{compactions}]",
                file=sys.stderr,
            )

        if verbose:
            msg_lines: list[str] = []
            for i, m in enumerate(context):
                role_val = str(m.get("role", ""))
                content_val = str(m.get("content", ""))
                msg_lines.append(
                    f"assembled context\n[{i}] {role_val}: {content_val}"
                )
            print("\n".join(msg_lines), file=sys.stderr)

        try:
            response: ChatResponse = client.chat(context, tool_schemas)
        except OverCapError:
            # Provider rejected on length despite the estimate — compact and retry.
            if compactions >= max_compactions or not compaction.compact(
                session, client, window, comp_cfg
            ):
                return _over_cap_giveup()
            compactions += 1
            continue

        tool_calls: list[ToolCall] | None = (
            response.tool_calls if response.tool_calls else None
        )
        session.append_assistant(response.text or "", tool_calls=tool_calls)

        if not response.tool_calls:
            episodic_maybe_extract(session, client)
            return response.text

        for call in response.tool_calls:
            print(
                f"Tool call: {call.name}({json.dumps(call.arguments)})",
                file=sys.stderr,
            )

            result = dispatch(call.name, call.arguments)
            rendered = render_tool_result(call.name, result)

            print(rendered, file=sys.stderr)

            session.append_tool_result(call.id, call.name, rendered)

        diagnostics_inject_summary(session)
