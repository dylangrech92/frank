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
import ui

from llm import ChatResponse, LLMClient, OverCapError, ToolCall
from session import Session
from tools.registry import dispatch, get_tool, schemas
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


# Master switch for long-term-memory work inside the turn loop (flashback
# seeding + episodic extraction). main.py flips it off under --no-memory.
MEMORY_ENABLED: bool = True


def flashback_maybe_seed(session: Session) -> None:
    """Turn-zero memory recall injection seam.

    Delegates to the flashback module, which decides -- via the terse and
    continuation gates -- whether to seed a compact recalled-memory bundle onto
    the session for this turn. The bundle (if any) is stashed on
    ``session._flashback_block`` and injected by the flashback context provider;
    it is never persisted to the transcript. Never raises.
    """
    if not MEMORY_ENABLED:
        return
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
    if not MEMORY_ENABLED:
        return
    try:
        import memory.episodic as episodic
    except Exception:
        return

    messages = session._messages
    total = len(messages)
    prev = _EPISODIC_WATERMARKS.get(id(session), session.episodic_watermark)
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
    session.set_episodic_watermark(total)
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
        meta_key: value                 # one line per non-empty meta entry
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

    for meta_key, meta_value in result.meta.items():
        if meta_value is None or meta_value == "":
            continue
        lines.append(f"{meta_key}: {meta_value}")

    hint_text = result.hint
    if hint_text is not None and hint_text:
        lines.append(f"hint: {hint_text}")

    return "\n".join(lines)


def _guard_oversize_result(
    name: str,
    result: ToolResult,
    rendered: str,
    cap: int,
    est_context_before_result: int,
) -> tuple[ToolResult, str]:
    """Discard an oversized rendered tool result and substitute a clean error.

    Dylan's explicit spec: never silently truncate a tool result — if it would
    consume more than 75% of the remaining token budget, discard the body
    entirely and return a ``result-too-large`` error instead, so the oversized
    raw body never enters the session transcript.

    Args:
        name: Registered tool name.
        result: The ``ToolResult`` returned by ``dispatch()``.
        rendered: The already-rendered text for *result* (see ``render_tool_result``).
        cap: The shared token cap for this turn's context (see ``compaction.compute_cap``).
        est_context_before_result: Token estimate of the assembled context before
            this result is appended.

    Returns:
        ``(result, rendered)`` unchanged when the result fits, or a substituted
        ``(ToolResult.err(...), rendered_error_text)`` pair when it does not.
    """
    result_tokens = compaction.estimate_tokens([{"role": "tool", "content": rendered}])
    budget = 0.75 * max(cap - est_context_before_result, 1)
    if result_tokens <= budget:
        return result, rendered

    tool = get_tool(name)
    action = getattr(tool, "action", "complete the operation") if tool else "complete the operation"
    oversize_hint = (
        getattr(tool, "oversize_hint", "narrow the request or use a more specific tool")
        if tool
        else "narrow the request or use a more specific tool"
    )
    substituted = ToolResult.err(
        f"{name} failed to {action} — result is too large to fit in context — {oversize_hint}",
        code="result-too-large",
    )
    return substituted, render_tool_result(name, substituted)


def _loop_guard_check(
    name: str,
    rendered: str,
    seen_errors: dict[tuple[str, str], int],
) -> str:
    """Append a steer suffix when the same (tool, error) has repeated within a turn.

    Cheap prefix check on the rendered text (mirrors Chalie's ``dispatch_service``
    pattern) so the success path pays nothing: only rendered error envelopes
    (``"[name(error..."`` header) participate in the count.

    Args:
        name: Registered tool name.
        rendered: The rendered result text for this call (post oversize-guard).
        seen_errors: Per-turn counting dict, keyed by ``(name, rendered)``,
            mutated in place.

    Returns:
        *rendered* unchanged, or *rendered* with a ``[loop-guard]`` suffix
        appended when this exact (name, rendered) pair has now been seen twice.
    """
    if not rendered.startswith(f"[{name}(error"):
        return rendered

    key = (name, rendered)
    seen_errors[key] = seen_errors.get(key, 0) + 1
    if seen_errors[key] < 2:
        return rendered

    tool = get_tool(name)
    alternative = getattr(tool, "alternative", "a different tool or approach") if tool else "a different tool or approach"
    steer = (
        f"\n\n[loop-guard] {name} called with these exact arguments has failed "
        f"multiple times with the same error. Do not repeat this call. Fix the "
        f"specific problem named above, try {alternative}, or stop and tell the "
        f"user what error you are getting."
    )
    return rendered + steer


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
    cap = compaction.compute_cap(window, comp_cfg)
    max_compactions = 5
    compactions = 0
    # Per-turn (reset on every handle_user_message call) tracking of rendered
    # error envelopes seen so far, for the repeated-identical-failure loop-guard.
    seen_errors: dict[tuple[str, str], int] = {}

    def _over_cap_giveup() -> str:
        msg = (
            "Context is over the model's token budget and compaction could not "
            "reduce it further. Start a new session or shorten the request."
        )
        session.append_assistant(msg)
        return msg

    while True:
        # Recomputed every iteration so a mid-turn load_tool call is reflected in
        # the very next client.chat — a stale pre-loop snapshot would otherwise
        # withhold the just-loaded tool's schema until the following user turn.
        tool_schemas = schemas()

        # Pre-flight: keep the assembled request at or below the shared cap,
        # compacting older turns before the call is ever made.
        context = session.assemble_context()
        est = compaction.estimate_tokens(context, tool_schemas)
        print(
            ui.telemetry(f"context: {len(context)} messages, ~{est} tokens (cap {cap})"),
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
                ui.telemetry(
                    f"context: {len(context)} messages, ~{est} tokens "
                    f"(cap {cap}) [post-compaction #{compactions}]"
                ),
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
                ui.tool_call(f"Tool call: {call.name}({json.dumps(call.arguments)})"),
                file=sys.stderr,
            )

            result = dispatch(call.name, call.arguments)
            rendered = render_tool_result(call.name, result)
            result, rendered = _guard_oversize_result(call.name, result, rendered, cap, est)
            rendered = _loop_guard_check(call.name, rendered, seen_errors)

            print(ui.tool_result(rendered), file=sys.stderr)

            session.append_tool_result(call.id, call.name, rendered)

        diagnostics_inject_summary(session)
