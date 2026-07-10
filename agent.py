"""Core agent loop with four named extension hooks for lifecycle phases.

``handle_user_message`` is the public entry point that drives the
read-eval-print cycle between a Session transcript and an LLMClient. Its body is
split into a per-turn ``_TurnState`` dataclass (the ~dozen mutable locals the
turn threads through its LLM calls) plus three focused helpers along the natural
seams: ``_run_llm_with_compaction`` (the compaction ladder + one chat call, with
OverCapError retry), ``_dispatch_round`` (the tool-call dispatch loop and its
guards), and ``_finalize_answer`` (the no-tool-call gate cascade: empty-answer
bounce → H1 verification nudge → S3 gate marking → placeholder → return). Also
here: three explicit hook no-op functions and the over-cap give-up seam.

Harness turn guidance (empty-answer bounce, H1 verification nudge) is injected
via ``Session.append_steer`` — a ``user``-role message flagged ``steer`` and
content-prefixed with ``session.STEER_PREFIX`` (an explicit "automated message
from the harness, NOT from the user" marker) — not as a plain user message, so the model,
the compaction summarizer, and any transcript reader can all tell harness
guidance from real human input.
"""

import hashlib
import json
import sys
import compaction
import ui

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

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


# Per-session flag for the graph-memory usage nudge (H5) — fires at most once per
# session, keyed by id(session).
_GRAPH_MEMORY_NUDGE_FIRED: dict[int, bool] = {}


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


# Master switch for long-term-memory work inside the turn loop (orientation
# seeding + consolidation). main.py flips it off under --no-memory.
MEMORY_ENABLED: bool = True


def orientation_maybe_seed(session: Session) -> None:
    """Task-start memory orientation injection seam.

    Delegates to the orientation module, which builds a code-anchored brief --
    the zero-LLM derived project skeleton personalised to this turn's task,
    plus anchored knowledge atoms and graph decisions/specs recalled by a
    task-relative query -- and stashes it on ``session._orientation_block`` for
    injection by the orientation context provider. Never persisted to the
    transcript. Never raises.
    """
    if not MEMORY_ENABLED:
        return
    try:
        import memory.orientation as orientation

        orientation.orientation_maybe_seed(session)
    except Exception:
        pass


def consolidation_maybe_extract(session: Session, client: LLMClient) -> None:
    """End-of-turn consolidation: hand this turn's diff + transcript tail to the
    off-thread consolidation writer, which reconciles it into durable, anchored
    knowledge atoms (and, when warranted, graph decisions/pivots).

    Non-blocking: snapshots exactly what ``consolidation.consolidate`` needs --
    this turn's ``files_changed`` and a copy of the recent transcript tail --
    and hands it to the background writer queue, then returns immediately so
    the REPL prompt never stalls on the LLM call. The same queue is drained
    (not re-invoked) by ``main.session_end_jobs`` at teardown, so a session's
    final turn is guaranteed to finish before the process exits without being
    consolidated twice.
    """
    if not MEMORY_ENABLED:
        return
    try:
        import memory.consolidation as consolidation
    except Exception:
        return

    # Emit the outcome of any PRIOR consolidation pass that has since completed.
    last = consolidation.pop_last_run()
    if last is not None:
        print(
            f"consolidation: last pass ran={last.get('ran')} added={last.get('added')} "
            f"updated={last.get('updated')} deleted={last.get('deleted')} "
            f"decisions={last.get('decisions')} pivots={last.get('pivots')} "
            f"reason={last.get('reason')}",
            file=sys.stderr,
        )

    turn_report = dict(getattr(session, "turn_report", {}) or {})
    messages_tail = list(session._messages[-consolidation.TRANSCRIPT_TAIL_MESSAGES :])
    try:
        consolidation.enqueue(str(session.project_root), turn_report, messages_tail, client)
        print("consolidation: enqueued turn for background reconciliation", file=sys.stderr)
    except Exception as exc:
        print(f"consolidation-enqueue-error: {exc}", file=sys.stderr)


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


# Hard cap on identical tool calls within a single turn. A model stuck
# re-issuing the exact same successful no-op (e.g. replace_one with search ==
# replace) would otherwise spin forever: the success-path steer below nudges
# first, and once this many identical (name, arguments) pairs have been
# dispatched, the call is refused at dispatch time (see handle_user_message).
# Verification tools (run_command / run_tests / verify_scratch) are exempt —
# repeating an identical build/test inside an edit→verify→edit cycle is legitimate.
#
# Single source of truth for the verification-tool names. The repeat-cap
# exemption, the verification_runs recording branch, and _activate_verification_tools
# all read this one set so the list cannot drift across the file.
_VERIFICATION_TOOLS = frozenset({"run_command", "run_tests", "verify_scratch"})
_REPEAT_CALL_CAP = 3
_REPEAT_CAP_EXEMPT = _VERIFICATION_TOOLS


def _activate_verification_tools() -> None:
    """Idempotently activate the verification tools the verify steers name.

    Both reactive verify steers (reproduce-before-edit, H1) tell the model to run
    a verification tool by name, but those tools are catalog-gated — they are only
    in the request's tools array once activated. Firing a steer without activating
    them demands a tool the model cannot call. This puts every verification tool
    into the active set so the next round's ``schemas()`` (re-derived per round in
    ``handle_user_message``) carries them and the guidance is actionable.

    ``registry.activate`` is a no-op for a tool already active/pinned/unknown, so
    this is safe to call repeatedly and never raises.
    """
    from tools.registry import activate

    for name in _VERIFICATION_TOOLS:
        activate(name)


def _verification_run_passed(name: str, result: ToolResult) -> bool:
    """True only when a verification run genuinely passed — not merely completed.

    ``run_command`` and ``run_tests`` both return ``ToolResult.ok`` even when the
    underlying work failed: a nonzero process exit or a failing test count is
    carried as *metadata*, not as an error status. So ``result.status ==
    'success'`` alone cannot tell a green run from a red one. This inspects the
    tool-specific meta so pass/fail is first-class:

    * ``run_command`` — passed iff the process exited 0 (``exit_code`` in
      ``(0, None)``; ``None`` covers a run with no captured code).
    * ``run_tests`` — passed iff zero tests failed (``failed == 0``).
    * ``verify_scratch`` — status alone suffices: its nonzero-exit path already
      returns an error result, so a success result is a genuine pass.

    Any error-status result is a fail regardless of tool.
    """
    if result.status != "success":
        return False
    if name == "run_command":
        return result.meta.get("exit_code", 0) in (0, None)
    if name == "run_tests":
        return result.meta.get("failed", 0) == 0
    # verify_scratch (and any future verification tool): a success status is a
    # genuine pass because the failure path already returns an error result.
    return True


def _is_bug_report(message: str) -> bool:
    """True when the user's request reads as a bug report (case-insensitive).

    A plain substring match against ``_BUG_REPORT_LEXICON``. Keeps the
    no-failure-observed steer out of pure feature work — it only makes sense to
    warn "you are fixing a failure you never saw fail" when the request actually
    claims a failure.
    """
    low = message.lower()
    return any(term in low for term in _BUG_REPORT_LEXICON)

# Escalation ladder above the hard cap. Once the cap starts refusing an
# identical call, a determined model can re-issue it every round — each a full
# LLM round-trip that dispatches nothing. Worse, when compaction drops the block
# error and the tool result from context, the model loses even the feedback that
# it is stuck (only a fold-surviving steer on the user row remains). After this
# many consecutive blocked calls with no dispatch in between, the turn is
# force-finalized (see _blocked_loop_giveup / handle_user_message).
_BLOCKED_STREAK_CAP = 3

# Fold-surviving steer for a blocked round, worded plainly (small local models
# treat bracketed tags as noise) and spelling out that it is an automated
# harness message, NOT the user's — append_steer already prepends STEER_PREFIX,
# so this is the body only. It rides the user role, which _prune_messages keeps
# across a compaction boundary, so the feedback survives even when the block
# error and tool result behind it are folded away.
_BLOCKED_ROUND_STEER = (
    "The last tool call was blocked because it has already run {n} times this "
    "turn with identical arguments. If you were re-issuing it because the earlier "
    "result is no longer visible, that result was removed from context to save "
    "space — do not repeat the call. Use what you already know, take a different "
    "action, or give your final answer now."
)

# Fold-surviving reproduce-before-edit steer, same wire-safety and plain-
# language rules as _BLOCKED_ROUND_STEER (append_steer prepends STEER_PREFIX, so
# this is the body only — no duplicate "not from the user" preamble). Fires on
# the first relevant file mutation of a turn that has run nothing to observe the
# problem, steering observed-output-first debugging over assumption-driven edits.
_REPRO_BEFORE_EDIT_STEER = (
    "The edit you just made was applied successfully and is already in the files. "
    "Do not re-check whether the original request still applies — it does, and "
    "your edit is part of it. Before editing anything else, run the relevant "
    "command with run_command and read its actual output: for a reported bug, "
    "crash, or wrong output that means reproducing the failure; otherwise it "
    "means running the code to confirm your change. Base any further edits on "
    "that observed output, not on assumption. The run_command tool is loaded "
    "into your toolset now — call it directly."
)

# Fold-surviving no-failure-observed steer, same wire-safety and plain-language
# rules as the steers above (append_steer prepends STEER_PREFIX, so this is the
# body only). Fires on the turn's first relevant file mutation when the task
# reports a failure but every command and test run so far this turn has passed —
# i.e. the model is starting to "fix" a failure it has never actually seen fail.
# Mutually exclusive with the reproduce-before-edit steer above (that one owns
# the zero-runs case; this one owns the runs-all-passed case).
_NO_FAILURE_OBSERVED_STEER = (
    "The edit you just made was applied successfully and is already in the files. "
    "Do not re-check whether the original request still applies — it does. The "
    "task reports a failure, but every command and test run this turn has passed: "
    "the reported failure has never been observed on this project as it stands. "
    "Do not fix code you have not seen fail, and do not modify files or data to "
    "force a failure — a failure you manufacture is not the reported failure. Run "
    "the reported failing command on the project exactly as it is; if it passes, "
    "revert your edit and state in your final answer that the reported problem "
    "could not be reproduced."
)

# Word/phrase lexicon (case-insensitive substring) that marks a user request as a
# bug report rather than a pure feature task. Gates the no-failure-observed steer:
# warning "you are fixing a failure you never saw fail" only makes sense when the
# request actually claims a failure.
_BUG_REPORT_LEXICON = (
    "crash",
    "error",
    "exception",
    "traceback",
    "bug",
    "broken",
    "fails",
    "failing",
    "failure",
    "regression",
    "wrong output",
    "incorrect",
)

# Success-path loop-guard suffix, APPENDED to the full re-rendered result of a
# repeated identical successful call (repeat #2..#cap). Used whenever the body
# must still be shown — a non-dedupable tool, or a changed / possibly-folded
# result — so the model both sees the output and is told the repeat made no
# progress. The dedup stub below replaces the body entirely when it is provably
# redundant.
_REPEAT_STEER_SUFFIX = (
    "\n\n[loop-guard] {name} was just called with these exact arguments "
    "and succeeded. Repeating the identical successful call makes no "
    "progress. Do not re-issue it. If the result was not what you needed, "
    "change your arguments or approach; otherwise move on or tell the "
    "user you are done."
)

# Dedup stub that REPLACES the full render of a repeated identical successful
# read-only call whose body has not changed since its last full render this
# turn. Re-emitting a byte-identical body wastes context and rewards the
# re-issue; the stub points the model back at the earlier result instead. Only
# used when _repeat_render's two safety conditions hold (see there).
_REPEAT_DEDUP_STUB = (
    "[loop-guard] {name} was already called with these exact arguments this "
    "turn and the result has not changed — output omitted; use the earlier "
    "result above. Do not re-issue this call. If you need something different, "
    "change your arguments or approach."
)

# Dedup stub that REPLACES the full render of a repeated identical successful
# VERIFICATION call (run_command/run_tests/verify_scratch) when nothing has been
# modified since the identical earlier run this turn. Verification is exempt from
# the repeat suffix and the hard cap because the edit→retest cycle legitimately
# repeats — but re-running a byte-identical command with ZERO intervening file
# mutations cannot produce a different result, so its output is omitted and the
# model is told to change something before re-verifying. Separate wording from
# _REPEAT_DEDUP_STUB (world fact + one directive) so the message names the true
# reason (no change on disk, not "read-only re-read"). Only used when the
# fingerprint, compaction count, AND mutation count are all unchanged since the
# last full render (see _repeat_render).
_VERIFY_NOCHANGE_STUB = (
    "[no-change] {name} already ran with these exact arguments and nothing has "
    "been modified since — its output is identical to the result shown above. "
    "Make a change before re-running verification."
)


def _render_fingerprint(rendered: str) -> str:
    """Stable content fingerprint of a rendered tool result.

    Lets the repeat-render dedup tell an *unchanged* repeated read (safe to
    replace with a stub) from one whose body actually differs — e.g. a re-read
    of a file the model just edited, which MUST get the fresh body. Hashing
    keeps the per-key bookkeeping O(1) in memory regardless of render size, and
    ``errors="replace"`` guarantees encoding never raises on odd bytes.
    """
    return hashlib.sha256(rendered.encode("utf-8", "replace")).hexdigest()


def _repeat_render(
    name: str,
    key: tuple[str, str],
    rendered: str,
    seen_renders: dict[tuple[str, str], tuple[str, int, int]],
    compactions: int,
    mutations: int,
    *,
    verification: bool,
) -> str:
    """Return a dedup stub for a provably-redundant repeat, else the full render.

    Called only for a repeat (count >= 2) that is a success on a tool eligible
    for dedup — either a read-only tool (``parallel_safe`` and not
    verification-exempt) or a verification tool (run_command/run_tests/
    verify_scratch). The stamp recorded at the last full render is
    ``key -> (fingerprint, compactions, mutations)``; which of those three
    stamps gate the stub depends on the tool class. Full gate table (a matched
    row omits the body; any mismatch re-emits the full body and re-stamps):

    | tool class   | fingerprint | compaction | mutation  | render on match      |
    |--------------|-------------|------------|-----------|----------------------|
    | read-only    | match       | unchanged  | (ignored) | _REPEAT_DEDUP_STUB   |
    | verification | match       | unchanged  | unchanged | _VERIFY_NOCHANGE_STUB|

    The three conditions each guard a distinct hazard:

    (d) Identical arguments do NOT imply an identical result. A re-read of a
        file the model just edited is legitimate and MUST get the fresh body, so
        the render's content *fingerprint* is compared against the one recorded
        at the last full render; on any mismatch the fresh body is returned.
    (e) A compaction may have folded the earlier result out of context. A stub
        that points at a result no longer present strands the model (the known
        amnesia loop), so the stub is withheld unless ``compactions`` is
        unchanged since the last full render.
    (m) For VERIFICATION only: an identical command whose output is stamp-clean
        can still be worth re-running once a file has changed (the edit→retest
        cycle). So the verification stub is withheld unless the turn's mutation
        count is also unchanged since the last full render — nothing modified
        means the output cannot differ. Read-only tools do NOT gate on (m): a
        re-read whose body is byte-identical is redundant regardless of an
        unrelated edit to some OTHER file, so the mutation stamp is carried on
        the record but never consulted for them (the fingerprint already speaks
        for content).

    A verification tool is EXEMPT from the no-progress suffix (a legitimate
    retest is progress, not a no-op loop), so its full-render branch returns the
    body plain; a read-only tool's carries ``_REPEAT_STEER_SUFFIX``.

    ``seen_renders`` is mutated in place: it is re-stamped with the current
    ``(fingerprint, compactions, mutations)`` on every full-render return so the
    NEXT repeat compares against the most recent body/context/mutation-count —
    in particular, after a verification mutation-stamp mismatch this lets a later
    identical run with no further mutations stub again. Kept as a small pure
    function so an eval can drive every branch with a fabricated stamp dict.
    """
    fingerprint = _render_fingerprint(rendered)
    prev = seen_renders.get(key)
    if verification:
        # (d) AND (e) AND (m): body, context, and disk all unchanged -> omit.
        matched = prev is not None and prev == (fingerprint, compactions, mutations)
        stub = _VERIFY_NOCHANGE_STUB
        full = rendered  # verification is exempt from the no-progress suffix
    else:
        # (d) AND (e) only; the mutation stamp is carried but not consulted.
        matched = (
            prev is not None
            and prev[0] == fingerprint
            and prev[1] == compactions
        )
        stub = _REPEAT_DEDUP_STUB
        full = rendered + _REPEAT_STEER_SUFFIX.format(name=name)
    if matched:
        return stub.format(name=name)
    # Mismatch on a gating condition: re-emit the body and re-stamp so the next
    # repeat compares against the fresh/re-materialised result and its stamps.
    seen_renders[key] = (fingerprint, compactions, mutations)
    return full


def _call_signature(arguments: dict) -> str:
    """Return a stable, hashable signature for a tool call's arguments.

    JSON with sorted keys gives a deterministic string for dict arguments
    regardless of insertion order; ``default=str`` keeps non-serialisable
    values from blowing up. Falls back to ``str(arguments)`` on any encoding
    error so counting never raises.
    """
    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(arguments)


def _repeat_call_check(
    name: str,
    arguments: dict,
    result: ToolResult,
    rendered: str,
    seen_calls: dict[tuple[str, str], int],
    seen_renders: dict[tuple[str, str], tuple[str, int, int]],
    compactions: int,
    mutations: int,
) -> str:
    """Steer (and, when safe, dedup) a repeated identical successful call.

    Companion to ``_loop_guard_check``: that function owns repeated identical
    *failures* (keyed on the rendered error envelope, which a cooperative model
    rarely produces twice — see evals/inline_loop_guard.py); this one owns
    repeated identical *successes* — the no-op loop where a model re-issues the
    exact same successful call (e.g. ``replace_one`` with search == replace, or
    the same ``read_file`` twice) over and over, making no progress. Only
    success results are steered here; errors are left to ``_loop_guard_check``
    to avoid double-suffixing.

    From repeat #2 on, the full body is normally re-rendered with a no-progress
    suffix (``_REPEAT_STEER_SUFFIX``). But when the body has provably not changed
    it is replaced by a short stub instead (``_repeat_render`` owns the gate
    table). Two tool classes are dedup-eligible, with different gates:

    * A read-only tool (``parallel_safe`` and not verification-exempt) is stubbed
      when the fingerprint and compaction count are unchanged since the last full
      render — a re-read whose body is byte-identical is redundant regardless of
      any unrelated edit.
    * A verification tool (run_command/run_tests/verify_scratch) is normally
      exempt from the repeat machinery, but a byte-identical re-run with ZERO
      intervening file mutations cannot produce a different result, so it too is
      stubbed — under the read-only gate PLUS an unchanged mutation count, and
      with its own ``_VERIFY_NOCHANGE_STUB`` wording. It never receives the
      no-progress suffix and is never hard-capped: a genuine edit→retest repeat
      (mutation count advanced) re-renders the full body plainly and re-stamps,
      so a later identical run with no new mutation stubs again.

    ``seen_renders``, ``compactions`` and ``mutations`` are always required so no
    caller can silently lose the dedup by forgetting to pass them.

    The count this maintains is also read pre-dispatch by the hard-cap block in
    ``handle_user_message`` to *refuse* an identical call once it has repeated
    ``_REPEAT_CALL_CAP`` times — a steer alone does not reliably break a
    determined loop (the failure-path steer is known not to fire live), so the
    cap guarantees termination. The stub therefore only ever replaces renders
    for repeats #2..#cap; the blocked call at #cap+1 keeps its own block message.

    Args:
        name: Registered tool name.
        arguments: The call's parsed arguments.
        result: The dispatched ToolResult (used to skip the success steer on
            errors, which ``_loop_guard_check`` owns).
        rendered: The rendered result text for this call.
        seen_calls: Per-turn counting dict keyed by ``(name, signature)``,
            mutated in place. Incremented for *every* call (success or error)
            so the pre-dispatch hard cap sees an accurate tally.
        seen_renders: Per-turn dedup stamps, ``(name, signature) ->
            (render fingerprint, compactions-at-render, mutations-at-render)``,
            mutated in place.
        compactions: Compactions performed so far this turn — the condition-(e)
            input that detects a fold between the last full render and this one.
        mutations: Monotonic count of file-mutation events this turn — the
            condition-(m) input that gates the VERIFICATION stub (an identical
            re-run after zero mutations cannot differ). Read-only dedup ignores
            it (the fingerprint already speaks for content), but it is stamped on
            the record for every dedup-eligible tool.

    Returns:
        *rendered* unchanged on a first success or an error; the full body plus
        ``_REPEAT_STEER_SUFFIX`` on a read-only repeat that cannot be safely
        deduped; the full body plain on an unstubbable verification repeat; or a
        short ``_REPEAT_DEDUP_STUB`` / ``_VERIFY_NOCHANGE_STUB`` replacing the
        body when the gate says the result is provably redundant.
    """
    key = (name, _call_signature(arguments))
    seen_calls[key] = seen_calls.get(key, 0) + 1

    # Errors are owned by _loop_guard_check; don't double-steer.
    if result.status != "success":
        return rendered

    # Two dedup-eligible classes, each with its own gate in _repeat_render:
    #   * verification tools (run_command/run_tests/verify_scratch) — stubbed
    #     only when nothing has changed on disk since the identical run;
    #   * read-only tools (parallel_safe, non-exempt) — stubbed on an unchanged
    #     body. get_tool -> None on an unknown name yields False (no dedup).
    is_verification = name in _VERIFICATION_TOOLS
    is_readonly = (
        name not in _REPEAT_CAP_EXEMPT
        and getattr(get_tool(name), "parallel_safe", False)
    )
    dedupable = is_verification or is_readonly

    if seen_calls[key] < 2:
        # First success: stamp the full render's fingerprint + compaction + mutation
        # count so a later identical repeat can tell an unchanged result (dedup to a
        # stub) from a changed one (fresh body after an edit / a real retest).
        if dedupable:
            seen_renders[key] = (_render_fingerprint(rendered), compactions, mutations)
        return rendered

    if dedupable:
        return _repeat_render(
            name, key, rendered, seen_renders, compactions, mutations,
            verification=is_verification,
        )
    return rendered + _REPEAT_STEER_SUFFIX.format(name=name)


# Write tools that mutate exactly the file named by their own `path` argument
# (as opposed to e.g. `replace_many`, which can touch an unbounded, only-known-
# after-the-fact set of paths, or `move_file`, whose mutation event fires on
# the *destination* path while `path` names the *source* -- a move changes a
# file's location, never its content, so it can never introduce a lint issue
# and is deliberately excluded here). Limiting reactive lint-delta injection to
# this single-path set keeps the pre-edit snapshot cheap and exact: one lint
# call on one known path before dispatch, one after.
_LINT_TRACKED_TOOLS = frozenset({"create_file", "update_file", "replace_one", "format"})

# Cap on how many new lint issues are appended per call, mirroring the
# `lint` tool's own `_MAX_RENDERED_ISSUES` guard against flooding context.
_LINT_DELTA_CAP = 20


def _lint_resolve_call_path(call: ToolCall, project_root: str) -> str | None:
    """Resolve *call*'s ``path`` argument to the absolute string mutation events use.

    Tools resolve their ``path`` argument via ``resolve_in_root(Path.cwd(), path)``
    and emit mutation events keyed by that resolved absolute path (see
    ``tools/_sandbox.py``) -- not by the raw, project-root-relative argument the
    model passed. This mirrors that same resolution so the argument can be
    matched against ``_scan_new_mutations``'s path set.

    Returns ``None`` on any resolution failure (escapes the root, wrong type,
    etc.) -- callers treat that identically to "nothing to snapshot".
    """
    path_val = call.arguments.get("path")
    if not isinstance(path_val, str) or not path_val:
        return None
    try:
        from tools._sandbox import resolve_in_root

        return str(resolve_in_root(project_root, path_val))
    except Exception:
        return None


def _lint_pre_snapshot(call: ToolCall, project_root: str):
    """Capture the pre-edit lint issues for *call*'s target file, if lintable.

    Returns ``None`` when the call isn't one of ``_LINT_TRACKED_TOOLS``, its
    ``path`` argument is missing/not a string/escapes the root, or its
    extension has no configured/available linter -- in every such case
    reactive lint-delta injection silently does nothing for this call.

    Args:
        call: The about-to-be-dispatched tool call.
        project_root: Absolute project root, as required by ``run_lint``.

    Returns:
        A list of ``LintIssue`` (the pre-edit snapshot for the path), or
        ``None`` when no snapshot could or should be taken.
    """
    if call.name not in _LINT_TRACKED_TOOLS:
        return None

    resolved_path = _lint_resolve_call_path(call, project_root)
    if resolved_path is None:
        return None

    try:
        from tools._lint import EXTENSION_LANGUAGE, run_lint
        from pathlib import Path as _Path

        if _Path(resolved_path).suffix.lower() not in EXTENSION_LANGUAGE:
            return None

        report = run_lint([resolved_path], project_root)
        if report.unavailable:
            return None  # no configured/available linter for this language
        return report.issues
    except Exception:
        return None


def _lint_delta_suffix(pre_issues, call: ToolCall, project_root: str) -> str:
    """Return an appended ``[lint] ...`` block for issues new since *pre_issues*.

    Re-lints the same path lint-snapshotted before dispatch and diffs against
    *pre_issues* by identity (path, line, col, rule, message). Only issues
    absent from the pre-edit snapshot are surfaced -- pre-existing project
    lint noise is never injected (the whole point of a delta, not a full
    report). Capped at ``_LINT_DELTA_CAP`` lines plus a "+N more" tail.

    Args:
        pre_issues: The pre-edit issue list returned by ``_lint_pre_snapshot``
            (never ``None`` when this is called).
        call: The tool call that was just dispatched (successfully, and
            confirmed via ``_scan_new_mutations`` to have mutated this path).
        project_root: Absolute project root, as required by ``run_lint``.

    Returns:
        An empty string when there is nothing new to report (including on any
        internal error), or a ``"\\n\\n[lint] ..."`` block ready to append to
        the rendered tool result.
    """
    resolved_path = _lint_resolve_call_path(call, project_root)
    if resolved_path is None:
        return ""

    try:
        from tools._lint import run_lint

        report = run_lint([resolved_path], project_root)
        if report.unavailable:
            return ""

        pre_keys = {(i.path, i.line, i.col, i.rule, i.message) for i in pre_issues}
        new_issues = [
            i for i in report.issues
            if (i.path, i.line, i.col, i.rule, i.message) not in pre_keys
        ]
        if not new_issues:
            return ""

        shown = new_issues[:_LINT_DELTA_CAP]
        lines = [f"[lint] {i.path}:{i.line} {i.rule} {i.message}" for i in shown]
        remaining = len(new_issues) - len(shown)
        if remaining > 0:
            lines.append(f"[lint] +{remaining} more")
        return "\n\n" + "\n".join(lines)
    except Exception:
        return ""


def _scan_new_mutations(events: list[dict]) -> tuple[bool, set[str]]:
    """Inspect newly observed mutation events for H1/H5 tracking.

    Args:
        events: A slice of ``_TURN_MUTATIONS`` added since the last check.

    Returns:
        ``(any_relevant, paths)`` — whether any event's ``kind`` is one of
        ``created``/``changed``/``renamed`` (the kinds that count as a real file
        mutation for verification/graph-memory purposes), and the set of
        distinct ``path`` values among those relevant events.
    """
    any_relevant = False
    paths: set[str] = set()
    for event in events:
        if event.get("kind") in ("created", "changed", "renamed"):
            any_relevant = True
            path_val = event.get("path")
            if path_val:
                paths.add(path_val)
    return any_relevant, paths


def _web_search_focus_check(name: str, result: ToolResult, rendered: str, searches_without_read: int) -> tuple[str, int]:
    """Nudge the model to stop paraphrase-searching and read a result instead.

    Reactive, zero-cost-on-happy-path mechanism mirroring ``_loop_guard_check``:
    tracks consecutive successful ``web_search`` calls (per turn) with no
    intervening ``web_read``.  Only ``web_search`` and ``web_read`` participate —
    every other tool passes through the counter untouched.  Failed ``web_search``
    calls do not count (the loop-guard already owns repeated failures).

    Args:
        name: Registered tool name for this call.
        result: The (possibly oversize-guard-substituted) ``ToolResult``.
        rendered: The rendered result text for this call.
        searches_without_read: Running per-turn count of consecutive successful
            ``web_search`` calls since the last ``web_read``.

    Returns:
        ``(rendered, searches_without_read)`` — *rendered* gets a ``[focus]``
        suffix appended when this is the 3rd+ consecutive successful
        ``web_search`` without a ``web_read`` in between; the updated counter is
        always returned.
    """
    if name == "web_read":
        return rendered, 0

    if name != "web_search" or result.status != "success":
        return rendered, searches_without_read

    searches_without_read += 1
    if searches_without_read >= 3:
        n = searches_without_read
        focus = (
            f"\n\n[focus] This is web search #{n} this turn without reading any "
            f"result. Searching again is unlikely to add new information — pick "
            f"the most relevant result and web_read it, or answer with what you "
            f"already have."
        )
        rendered = rendered + focus

    return rendered, searches_without_read


# =============================================================================
# Per-turn state and loop helpers
# =============================================================================

@dataclass
class _TurnState:
    """Mutable per-turn state threaded through one ``handle_user_message`` call.

    Groups the locals the turn loop shares across its LLM calls, dispatch
    rounds, and gate cascade so the helpers can read and mutate them by
    reference. One instance lives for exactly one turn and is discarded.
    """

    # S4 per-turn report (files_changed, verification_runs, usage, gate flags);
    # also exposed on ``session.turn_report`` so --json mode can read it back
    # even if the turn raises before returning. ``answer`` holds the UNPREFIXED
    # final text — the single source of truth for both the return value (which
    # may still get the "[UNVERIFIED CHANGES] " prefix layered on for prose
    # callers) and the envelope's answer field.
    turn_report: dict
    cap: int
    window: int
    comp_cfg: dict
    # Token estimate of the assembled context the current chat request was built
    # from (after any pre-flight compaction). Reused by the oversize-result
    # guard and the usage calibration for that same request.
    est: int = 0
    # Compactions performed this turn, capped across the whole turn.
    compactions: int = 0
    # Chars streamed through on_delta for the CURRENT chat call only (reset
    # before each call), so the return path knows whether the answer already
    # reached the sink or must be delivered whole (non-streaming/gated fallback).
    streamed: int = 0
    # The turn's original user message, kept for the bug-report lexicon check the
    # no-failure-observed steer gates on. Set once at turn start.
    user_message: str = ""
    # H1/S3 verify gate: True once a file was created/changed/renamed with no
    # subsequent PASSING run_tests/run_command/verify_scratch; cleared the moment
    # such a call passes (a run that merely completed but exited nonzero / left a
    # failing test does NOT clear it). The nudge fires at most once per turn.
    needs_verification: bool = False
    verification_nudge_fired: bool = False
    # Reproduce-before-edit steer: True once the first relevant file mutation
    # of the turn landed with zero verification runs recorded so far. Fires the
    # fold-surviving steer at most once per turn (see _dispatch_sequential_call /
    # _dispatch_round).
    repro_steer_fired: bool = False
    # No-failure-observed steer: True once the turn's first relevant file mutation
    # landed while >= 1 verification run had been recorded AND every one passed,
    # on a task whose original request reads as a bug report. Mutually exclusive
    # with repro_steer_fired (that owns the zero-runs case). Fires the fold-
    # surviving steer at most once per turn (see _dispatch_sequential_call /
    # _dispatch_round).
    no_failure_steer_fired: bool = False
    # Empty-answer bounce: the model returned no text and no tool calls (a
    # local-model bare-stop failure mode). Fires at most once per turn; a second
    # empty turn falls through to a transparent placeholder at the return site.
    empty_answer_nudge_fired: bool = False
    # H5 graph-memory nudge input: whether a record(kind='decision'/'spec') call ran
    # this turn (paired with mutated_paths); the fired-once-per-session flag lives in
    # _GRAPH_MEMORY_NUDGE_FIRED.
    record_decision_or_spec_called: bool = False
    # Reactive web-search focus nudge: consecutive successful web_search calls
    # since the last web_read.
    searches_without_read: int = 0
    # files_changed dedupe set: first-tool-wins, in order of first mutation.
    files_changed_seen: set[str] = field(default_factory=set)
    # Turn-start pre-images for net-change detection: normalized absolute path
    # (the SAME resolved key a files_changed entry carries, so the two join
    # exactly) -> sha256 hex of the file's bytes BEFORE this turn first touched
    # it, or None meaning "did not exist at turn start". Captured at the dispatch
    # seam before a mutating tool writes; first capture per path wins (= the
    # turn-start state). A path ABSENT from this map is UNKNOWN — e.g. a file
    # whose first mutation came from a run_command shell side effect, which
    # carries no path argument to snapshot — and is never guessed at annotation
    # time. Read once at the turn-exit choke point to flag reverted entries.
    preimages: dict[str, str | None] = field(default_factory=dict)
    # Rendered error envelopes seen this turn, for the repeated-identical-failure
    # loop-guard (keyed by (name, rendered)).
    seen_errors: dict[tuple[str, str], int] = field(default_factory=dict)
    # Identical (name, arg-signature) pairs dispatched this turn, for the
    # repeated-successful-call loop-guard steer plus the hard dispatch cap.
    seen_calls: dict[tuple[str, str], int] = field(default_factory=dict)
    # Full-render dedup stamps for repeated identical successful dedup-eligible
    # calls, keyed by (name, arg-signature) ->
    # (render fingerprint, compactions-at-render, mutations-at-render).
    # Lets _repeat_call_check replace a provably-unchanged repeat with a short stub
    # instead of re-emitting the body: the fingerprint forces a fresh body when a
    # re-read reflects a just-applied edit, the compaction count forces one when a
    # fold may have dropped the earlier result out of context, and the mutation
    # count forces one for verification tools when a file changed since the run
    # (read-only dedup carries but does not gate on it). Distinct from seen_calls,
    # whose int tally the pre-dispatch hard cap depends on.
    seen_renders: dict[tuple[str, str], tuple[str, int, int]] = field(default_factory=dict)
    # Every distinct path mutated this turn (H1/H5 tracking).
    mutated_paths: set[str] = field(default_factory=set)
    # Monotonic count of file-mutation EVENTS this turn (not distinct paths — a
    # second edit to an already-mutated path bumps it), spanning both direct
    # edit-tool mutations and run_command's snapshot-diff shell mutations (both
    # publish on the _TURN_MUTATIONS bus). Recorded at every dedup-eligible full
    # render; a verification repeat is only stubbed when this is unchanged since
    # the identical earlier run, so a byte-identical re-run with nothing modified
    # is deduped while a genuine edit→retest re-renders the full body.
    mutation_events: int = 0
    # Loop-guard escalation: consecutive tool calls refused by the repeat-call
    # hard cap, counted regardless of key (alternating between two blocked calls
    # is just as stuck). Incremented on every hard-cap block, reset to 0 the
    # moment any call actually dispatches (sequential or a parallel-safe batch) —
    # a real dispatch proves progress. When it reaches ``_BLOCKED_STREAK_CAP`` the
    # turn is force-finalized instead of looping the same blocked call forever
    # (see handle_user_message / _blocked_loop_giveup).
    blocked_streak: int = 0


# How many files / verification runs a synthesized give-up answer lists
# before collapsing the tail into a "+N more" count, so a huge turn cannot bloat
# the envelope.
_GIVEUP_FILES_CAP = 10
_GIVEUP_RUNS_CAP = 5


def _turn_verified(state: "_TurnState") -> bool | None:
    """Single source of truth for ``turn_report["verified"]``.

    Tri-state, so a consumer can tell an unverified edit from a turn that never
    touched code:

    * ``None`` — the turn mutated no file, so there was nothing to verify (e.g.
      a read-only research turn). Neither pass nor fail applies.
    * ``True`` — the turn mutated a file AND the verification gate is clear (a
      ``run_tests``/``run_command``/``verify_scratch`` call PASSED after the last
      mutation, clearing ``needs_verification`` — a run that merely completed but
      exited nonzero or left a failing test does not count).
    * ``False`` — the turn mutated a file but the gate is still open (no passing
      run after the last mutation).

    Shared verbatim by ``_finalize_answer`` and both early-exit give-up paths so
    the flag means the same thing however the turn ended — a give-up envelope no
    longer under-reports verified work, and a no-op turn no longer masquerades as
    an unverified edit.
    """
    if not state.mutated_paths:
        return None
    return not state.needs_verification


def _annotate_reverted_files(state: "_TurnState") -> None:
    """Flag ``files_changed`` entries the turn left byte-identical to their start.

    ``files_changed`` is an activity log — a file edited and then reverted to its
    original bytes still appears — so a consumer reading it alone would conclude
    the tree changed when the net state did not. This walks the log once and adds
    ``entry["reverted"] = True`` to an entry ONLY when the harness knows the
    file's turn-start state (a pre-image was captured, see ``_capture_preimage``)
    AND the file ended the turn identical to it:

    * a recorded hash that matches the current on-disk bytes (edited then written
      back), or
    * a recorded ``None`` (did not exist at turn start) and the path no longer
      exists (created then deleted).

    A path with no recorded pre-image is UNKNOWN and never annotated — the flag
    is absent, never guessed. The key is left off entirely otherwise (absent =
    changed-or-unknown), keeping the envelope shape minimal and append-only.
    Called at the single turn-exit choke point beside the ``verified`` stamp so
    every exit path (normal finalize and both give-up paths) reports net state.
    """
    from pathlib import Path as _Path

    for entry in state.turn_report["files_changed"]:
        key = entry.get("path")
        if key not in state.preimages:
            continue  # unknown turn-start state — never guess
        pre = state.preimages[key]
        target = _Path(key)
        if pre is None:
            if not target.exists():
                entry["reverted"] = True
            continue
        try:
            if (
                target.exists()
                and hashlib.sha256(target.read_bytes()).hexdigest() == pre
            ):
                entry["reverted"] = True
        except OSError:
            continue  # unreadable now — cannot confirm a revert, leave absent


def _synthesize_giveup_answer(reason: str, retry_hint: str, state: "_TurnState") -> str:
    """Build a truthful give-up answer from the facts already in ``turn_report``.

    The two early-exit give-up paths used to emit a static string claiming
    "a report of what was accomplished is unavailable", which is false whenever
    the turn had already applied edits and verified them on disk. ``turn_report``
    already holds those facts (``files_changed`` + ``verification_runs``, each run
    with whether it passed), so this synthesizes an honest envelope from them with
    no extra LLM call. *reason* is
    the leading sentence naming why the harness ended the turn; *retry_hint* is
    the path-appropriate advice used only when the turn did no work at all.
    """
    files = state.turn_report["files_changed"]
    runs = state.turn_report["verification_runs"]

    if not files and not runs:
        # No mutations and no runs — nothing to report beyond the plain give-up,
        # staying close to each path's original terminal message.
        return f"{reason} No files were changed and nothing was run this turn — {retry_hint}"

    parts = [reason, "Work already applied this turn."]

    if files:
        seen: list[str] = []
        for entry in files:
            path = entry.get("path")
            if path and path not in seen:
                seen.append(path)
        shown = seen[:_GIVEUP_FILES_CAP]
        seg = "Files changed: " + ", ".join(shown)
        extra = len(seen) - len(shown)
        if extra > 0:
            seg += f" (+{extra} more)"
        parts.append(seg + ".")

    if runs:
        shown_runs = runs[:_GIVEUP_RUNS_CAP]
        rendered = [
            f"{r.get('tool')} {str(r.get('detail', ''))!r} "
            f"({'passed' if r.get('passed') else 'failed'})"
            for r in shown_runs
        ]
        seg = "Verification runs: " + "; ".join(rendered)
        extra = len(runs) - len(shown_runs)
        if extra > 0:
            seg += f" (+{extra} more)"
        parts.append(seg + ".")

    parts.append(
        "The turn was cut short, so parts of the request may be incomplete — "
        "review the applied changes before retrying."
    )
    return " ".join(parts)


def _finalize_giveup(
    session: Session,
    state: "_TurnState",
    reason: str,
    retry_hint: str,
    on_delta: Callable[[str], None] | None,
) -> str:
    """Shared tail for the two early-exit give-up paths.

    Both ``_over_cap_giveup`` and ``_blocked_loop_giveup`` end a turn without
    reaching ``_finalize_answer``. They fold their common tail here: synthesize a
    truthful answer from ``turn_report`` (no extra LLM call), record it as the
    turn's final answer across transcript + ``turn_report`` + ``on_delta``, and
    stamp ``turn_report["verified"]`` with the one shared formula (``_turn_verified``).
    """
    msg = _synthesize_giveup_answer(reason, retry_hint, state)
    session.append_assistant(msg)
    state.turn_report["answer"] = msg
    _annotate_reverted_files(state)
    state.turn_report["verified"] = _turn_verified(state)
    if on_delta is not None:
        on_delta(msg + "\n")
    return msg


def _over_cap_giveup(
    session: Session, state: "_TurnState", on_delta: Callable[[str], None] | None
) -> str:
    """Terminal give-up when compaction cannot get the context under cap.

    Records a truthful synthesized answer as the turn's final answer
    (transcript, ``turn_report``, and on_delta payload), stamps ``verified``, and
    returns it verbatim.
    """
    reason = (
        "This turn was ended by the harness: the context went over the model's "
        "token budget and compaction could not reduce it further."
    )
    return _finalize_giveup(
        session, state, reason, "start a new session or shorten the request.", on_delta
    )


def _blocked_loop_giveup(
    session: Session, state: "_TurnState", on_delta: Callable[[str], None] | None
) -> str:
    """Terminal give-up when the same tool call is blocked in a tight loop.

    The escalation ladder above the repeat-call hard cap: once the harness has
    refused ``_BLOCKED_STREAK_CAP`` identical calls in a row with no dispatch in
    between (the model ignored both the block error and the fold-surviving steer),
    further chat rounds only burn a full LLM round-trip per blocked call. This
    ends the turn instead — mirroring ``_over_cap_giveup``: records a truthful
    synthesized answer as the turn's final answer (transcript,
    ``turn_report``, and on_delta payload), stamps ``verified``, and returns it
    verbatim.
    """
    print(
        ui.telemetry(
            f"loop-guard: escalated — turn force-finalized after "
            f"{state.blocked_streak} consecutive blocked calls"
        ),
        file=sys.stderr,
    )
    reason = (
        f"This turn was ended by the harness: the same tool call was repeated "
        f"and blocked {state.blocked_streak} times in a row with no progress."
    )
    return _finalize_giveup(
        session, state, reason, "rephrase or split the request and try again.", on_delta
    )


def _record_llm_usage(
    session: Session,
    state: "_TurnState",
    response: ChatResponse,
    est: int,
    context: list[dict],
) -> None:
    """Fold a completed chat response's token usage into session + turn stats.

    S2 calibration (real ``prompt_tokens`` vs the estimate, remembered together
    with where this context ended so the next trigger check can compose real +
    delta), S4 per-turn usage accumulation on ``turn_report``, and the
    cumulative stats.json row. Usage fields stay ``None`` all turn on providers
    that never report usage.
    """
    if response.prompt_tokens is not None:
        # S2 — calibrate the fallback estimator toward this request's real
        # usage, then remember it (plus where this context ended) so the next
        # pre-flight trigger check can compose real + delta instead of
        # re-estimating the whole transcript from scratch.
        compaction.update_calibration(session, response.prompt_tokens, est)
        session.last_prompt_tokens = response.prompt_tokens
        session.last_prompt_context_len = len(context)
        print(
            ui.telemetry(
                f"usage: actual prompt_tokens={response.prompt_tokens} "
                f"vs estimated ~{est} tokens (delta {response.prompt_tokens - est:+d}) "
                f"[ema {session.token_estimate_ratio:.2f}]"
            ),
            file=sys.stderr,
        )

    # S4 — sum usage across every LLM call this turn (None until the first
    # real figure arrives, then a running total).
    if response.prompt_tokens is not None:
        state.turn_report["usage"]["prompt_tokens"] = (
            (state.turn_report["usage"]["prompt_tokens"] or 0) + response.prompt_tokens
        )
    if response.completion_tokens is not None:
        state.turn_report["usage"]["completion_tokens"] = (
            (state.turn_report["usage"]["completion_tokens"] or 0)
            + response.completion_tokens
        )

    # Usage stats: fold this LLM call into the session's cumulative stats.json
    # row (run time, tool-call count, token totals).
    session.record_llm_call(
        response.prompt_tokens,
        response.completion_tokens,
        len(response.tool_calls),
    )


def _run_llm_with_compaction(
    session: Session,
    client: LLMClient,
    state: "_TurnState",
    tool_schemas: list[dict],
    delta_cb: Callable[[str], None] | None,
    verbose: bool,
    on_delta: Callable[[str], None] | None,
) -> ChatResponse | str:
    """Compact the assembled context under cap, then make one chat call.

    Runs the full pre-flight compaction ladder (summarizer loop with a thrash
    guard, then the free force_fold last resort), makes exactly one
    ``client.chat`` call, and folds the response's usage into the session/turn
    stats. When the provider rejects the request with ``OverCapError`` despite
    the estimate, it compacts once and retries the whole cycle.

    Returns the ``ChatResponse`` on success, or the over-cap give-up string
    (already delivered to the transcript/on_delta) when compaction is exhausted
    — the caller returns that string as the turn's answer.
    """
    max_compactions = 5
    # Thrash guard: bail out of the summarizer loop after this many consecutive
    # compactions that failed to reduce the context (further calls won't
    # converge) and fall through to the force_fold last resort.
    max_non_shrink = 2

    while True:
        # Pre-flight: keep the assembled request at or below the shared cap,
        # compacting older turns before the call is ever made.
        context = session.assemble_context()
        est = compaction.estimate_tokens(context, tool_schemas)
        # S2 — usage-driven trigger: real prompt_tokens from the last response
        # (plus a calibrated estimate of what was appended since) when
        # available, otherwise the calibrated fallback estimate. See
        # compaction.trigger_estimate for the composition rationale.
        trigger = compaction.trigger_estimate(session, context, tool_schemas)
        print(
            ui.telemetry(f"context: {len(context)} messages, ~{est} tokens (cap {state.cap})"),
            file=sys.stderr,
        )
        if session.last_prompt_tokens is not None:
            print(
                ui.telemetry(
                    f"trigger: ~{trigger} tokens (real {session.last_prompt_tokens} "
                    f"+ calibrated delta, ema {session.token_estimate_ratio:.2f})"
                ),
                file=sys.stderr,
            )
        non_shrink_streak = 0
        while trigger > state.cap:
            if state.compactions >= max_compactions or not compaction.compact(
                session, client, state.window, state.comp_cfg
            ):
                # Summarization exhausted (budget or nothing foldable) — break
                # to the force_fold last resort rather than dead-ending.
                break
            state.compactions += 1
            # Compaction reshaped the assembled context (summary spliced in),
            # so the prior real-usage baseline's index no longer lines up —
            # fall back to the calibrated estimate until the next response.
            session.last_prompt_tokens = None
            context = session.assemble_context()
            est = compaction.estimate_tokens(context, tool_schemas)
            prev_trigger = trigger
            trigger = compaction.trigger_estimate(session, context, tool_schemas)
            print(
                ui.telemetry(
                    f"context: {len(context)} messages, ~{est} tokens "
                    f"(cap {state.cap}) [post-compaction #{state.compactions}]"
                ),
                file=sys.stderr,
            )
            # Thrash guard: stop paying for summarizer calls that aren't
            # reducing the context. Two consecutive no-reduction compactions
            # mean further calls won't converge — fall through to force_fold.
            if trigger >= prev_trigger:
                non_shrink_streak += 1
                if non_shrink_streak >= max_non_shrink:
                    break
            else:
                non_shrink_streak = 0

        # Overflow ladder: summarization could not get under cap. As a last
        # resort, drop all but the most recent messages with no LLM call (older
        # context is lost rather than failing the turn), then re-check.
        if trigger > state.cap and compaction.force_fold(session, state.comp_cfg):
            state.compactions += 1
            session.last_prompt_tokens = None
            context = session.assemble_context()
            est = compaction.estimate_tokens(context, tool_schemas)
            trigger = compaction.trigger_estimate(session, context, tool_schemas)
            print(
                ui.telemetry(
                    f"context: {len(context)} messages, ~{est} tokens "
                    f"(cap {state.cap}) [force-fold #{state.compactions}]"
                ),
                file=sys.stderr,
            )
        if trigger > state.cap:
            return _over_cap_giveup(session, state, on_delta)

        if verbose:
            msg_lines: list[str] = []
            for i, m in enumerate(context):
                role_val = str(m.get("role", ""))
                content_val = str(m.get("content", ""))
                msg_lines.append(
                    f"assembled context\n[{i}] {role_val}: {content_val}"
                )
            print("\n".join(msg_lines), file=sys.stderr)

        # S3 — hard verify gate: once the H1 bounce has fired and the mutation
        # is still unresolved, this call's answer may need a harness-added
        # "[UNVERIFIED CHANGES] " prefix decided *after* the response is fully
        # in hand. Streaming the raw text through on_delta as it arrives would
        # violate the "sink receives the final text exactly once" contract (the
        # prefix must lead, and it can't be inserted retroactively into an
        # already-streamed prefix-less stream). So this one call is buffered
        # (delta_cb withheld) and delivered whole — with or without the marker —
        # once the gate decision is made in _finalize_answer. Any iteration
        # where the gate isn't in this pending state streams exactly as before.
        try:
            state.streamed = 0
            gate_pending = state.verification_nudge_fired and state.needs_verification
            chat_delta_cb = None if gate_pending else delta_cb
            state.turn_report["usage"]["llm_calls"] += 1
            response: ChatResponse = client.chat(context, tool_schemas, chat_delta_cb)
        except OverCapError:
            # Provider rejected on length despite the estimate — compact and
            # retry; if summarization can't help, fall back to force_fold.
            if state.compactions < max_compactions and compaction.compact(
                session, client, state.window, state.comp_cfg
            ):
                state.compactions += 1
            elif compaction.force_fold(session, state.comp_cfg):
                state.compactions += 1
            else:
                return _over_cap_giveup(session, state, on_delta)
            # Same reasoning as the pre-flight compaction path above: the
            # baseline this real measurement was keyed to no longer applies.
            session.last_prompt_tokens = None
            continue

        state.est = est
        _record_llm_usage(session, state, response, est, context)
        return response


def _maybe_graph_memory_nudge(session: Session, state: "_TurnState") -> None:
    """H5 — graph-memory usage nudge, fired at most once per session.

    When this turn's mutations spanned 3+ distinct paths with no
    record(kind='decision'/'spec') call, append a one-line reminder to the last
    tool result already in the transcript (the same append mechanism
    diagnostics_inject_summary uses). Must run BEFORE ``session.append_assistant``
    records this turn's answer, since ``amend_last_tool_result`` only touches
    ``_messages[-1]`` when its role is "tool".
    """
    if (
        len(state.mutated_paths) >= 3
        and not state.record_decision_or_spec_called
        and not _GRAPH_MEMORY_NUDGE_FIRED.get(id(session), False)
    ):
        _GRAPH_MEMORY_NUDGE_FIRED[id(session)] = True
        print(ui.telemetry("graph-memory-nudge: fired"), file=sys.stderr)
        session.amend_last_tool_result(
            "\n\n[memory] This change spans several files. If a design "
            "decision drove it, record it with record(kind='decision') so "
            "future sessions inherit the reasoning."
        )


def _maybe_arm_first_mutation_steer(state: "_TurnState") -> None:
    """Arm at most one first-mutation verify steer for the turn.

    Called the moment the turn's FIRST relevant file mutation lands. The two
    steers are mutually exclusive and cover disjoint situations:

    * zero verification runs recorded so far -> reproduce-before-edit (observe
      the problem before editing on assumption);
    * >= 1 run recorded AND every one passed, on a task whose original request
      reads as a bug report -> no-failure-observed (the model is starting to
      "fix" a failure it has never seen fail).

    A run that failed at least once means a failure WAS observed this turn, so
    neither steer arms. Each flag is one-shot; both load the verification tools
    the steer names so the tool is callable in the very next round.
    """
    runs = state.turn_report["verification_runs"]
    if not runs:
        if not state.repro_steer_fired:
            state.repro_steer_fired = True
            _activate_verification_tools()
            print(
                ui.telemetry(
                    "repro-steer: fired (mutation before any verification "
                    "run this turn)"
                ),
                file=sys.stderr,
            )
        return
    if (
        not state.no_failure_steer_fired
        and all(r["passed"] for r in runs)
        and _is_bug_report(state.user_message)
    ):
        state.no_failure_steer_fired = True
        _activate_verification_tools()
        print(
            ui.telemetry(
                "no-failure-steer: fired (edit on a reported failure that has "
                "not been observed this turn)"
            ),
            file=sys.stderr,
        )


def _record_verification_run(
    state: "_TurnState", call: ToolCall, result: ToolResult
) -> None:
    """Record one verification run and update the pass-based verify gate.

    Appends a ``verification_runs`` entry — ``tool``/``status``/``detail`` (kept
    verbatim; evals assert on them) plus a first-class ``passed`` bool from
    ``_verification_run_passed`` — and clears ``needs_verification`` ONLY when the
    run passed. A run that merely completed but failed (a nonzero ``run_command``
    exit, a failing ``run_tests`` count, a ``verify_scratch`` snippet error) no
    longer counts as verification: the gate stays open so a later ``verified:
    true`` cannot be stamped off a run that never went green, and the model gets
    no false signal that the change is confirmed.
    """
    if call.name == "run_command":
        detail = str(call.arguments.get("cmd", ""))
    else:
        detail = str(call.arguments.get("path") or ".")
    passed = _verification_run_passed(call.name, result)
    state.turn_report["verification_runs"].append(
        {
            "tool": call.name,
            "status": result.status,
            "detail": detail,
            "passed": passed,
        }
    )
    if passed:
        state.needs_verification = False


# Files larger than this are not pre-imaged for net-change detection: hashing
# them on every mutating call would tax the dispatch hot path, and the reverted
# flag is a best-effort truthful signal, not a guarantee — an un-hashed path
# stays UNKNOWN (no reverted key), never guessed.
_PREIMAGE_MAX_BYTES = 5 * 1024 * 1024


def _capture_preimage(state: "_TurnState", call: ToolCall, project_root: str) -> None:
    """Record the turn-start state of *call*'s target file, once per path.

    Called at the dispatch seam BEFORE a mutating tool writes, so the recorded
    hash reflects the file as it was when the turn first touched it. The key is
    the resolved absolute path the mutation event (and thus the ``files_changed``
    entry) uses — via the same ``_lint_resolve_call_path`` resolution — so the
    two join exactly at annotation time.

    Records nothing (leaving the path UNKNOWN) unless the call carries a
    resolvable ``path`` argument on a non-parallel_safe tool: a ``run_command``
    shell side effect has no ``path`` argument, so a file it mutates first has no
    capturable pre-image and stays unknown, never guessed. A nonexistent target
    records ``None`` (did-not-exist at turn start); an oversize
    (> ``_PREIMAGE_MAX_BYTES``) or unreadable file records nothing (unknown).

    A path already mutated this turn is skipped (``files_changed_seen``): its
    turn-start state is gone, so a pre-image taken now would be a mid-turn state,
    not the start — this is what keeps a path whose FIRST mutation was a
    pre-imageless ``run_command`` side effect unknown even when a later edit tool
    touches it. Combined with the ``preimages`` guard, first capture per path
    wins and a later write never overwrites a recorded turn-start state.
    """
    if getattr(get_tool(call.name), "parallel_safe", False):
        return  # read-only tools never mutate — nothing to pre-image
    key = _lint_resolve_call_path(call, project_root)
    if key is None or key in state.preimages or key in state.files_changed_seen:
        return
    from pathlib import Path as _Path

    target = _Path(key)
    if not target.exists():
        state.preimages[key] = None  # did not exist at turn start
        return
    try:
        if target.stat().st_size > _PREIMAGE_MAX_BYTES:
            return  # too large to hash cheaply — leave unknown, never guess
        state.preimages[key] = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError:
        return  # unreadable — unknown


def _record_file_mutations(
    state: "_TurnState", call: ToolCall, new_events: list[dict]
) -> None:
    """Fold this call's mutation events into the S4 ``files_changed`` log.

    Correlated with the dispatched call's own slice of ``_TURN_MUTATIONS`` (NOT
    the module-level list read later — ``diagnostics_inject_summary`` drains it at
    the end of the dispatch round). Counts EVERY relevant mutation event (before
    the path dedup) so a second edit to an already-mutated path still bumps
    ``mutation_events`` — the stamp the verification dedup gates on, which
    ``len(mutated_paths)`` would miss. Appends to ``files_changed`` deduped by
    path, first-tool-wins, in order of first mutation.
    """
    for event in new_events:
        if event.get("kind") not in ("created", "changed", "renamed"):
            continue
        state.mutation_events += 1
        path = event.get("path")
        if path and path not in state.files_changed_seen:
            state.files_changed_seen.add(path)
            state.turn_report["files_changed"].append(
                {"path": path, "tool": call.name}
            )


def _dispatch_sequential_call(
    session: Session,
    state: "_TurnState",
    call: ToolCall,
) -> tuple[ToolResult, str]:
    """Dispatch one tool call on the sequential path, applying the per-call guards.

    Echoes the call to stderr, snapshots the mutation log, the target file's
    lint state, and its net-change pre-image before dispatch, applies the
    loop-guard hard cap, then attributes any mutation events this specific call
    produced (H1/S3 verification tracking, S4 files_changed accounting via
    ``_record_file_mutations``, and the I2 reactive lint-delta).

    The hard cap refuses an identical call once it has repeated
    ``_REPEAT_CALL_CAP`` times this turn (verification tools exempt — a
    rebuild/retest cycle legitimately repeats). The count is maintained
    post-render by ``_repeat_call_check``; here we read the tally of *previous*
    identical calls and block before dispatching, guaranteeing a stuck no-op loop
    ends. A block increments ``state.blocked_streak`` (feeding the escalation
    ladder in handle_user_message), while any real dispatch resets it to 0 —
    consecutive blocks with no dispatch in between are the stuck signal.

    Returns ``(result, lint_suffix)`` — the dispatched (or blocked) ToolResult
    and the reactive lint-delta suffix (empty unless a single-path write tool
    introduced a lint regression). The caller runs the shared rendered-result
    pipeline on the pair.
    """
    lint_suffix = ""
    print(
        ui.tool_call(f"Tool call: {call.name}({json.dumps(call.arguments)})"),
        file=sys.stderr,
    )
    # Snapshot before/after this specific call so a mutation event can be
    # attributed to it (parallel_safe tools never mutate, so this tracking only
    # needs the sequential path — see Tool.parallel_safe).
    pre_mutation_len = len(_TURN_MUTATIONS)
    # I2 — reactive lint-delta injection: snapshot this call's target file's lint
    # issues BEFORE dispatch (only for the single-path write tools in
    # _LINT_TRACKED_TOOLS; None means "nothing to compare against", so no delta is
    # ever appended).
    lint_pre = _lint_pre_snapshot(call, str(session.project_root))
    # Net-change pre-image: capture the target file's turn-start state before this
    # call writes, keyed to match its future files_changed entry. Harmless on
    # read-only/non-path calls (they record nothing); files first mutated by
    # run_command have no path argument here and stay unknown.
    _capture_preimage(state, call, str(session.project_root))
    _repeat_key = (call.name, _call_signature(call.arguments))
    _repeat_n = state.seen_calls.get(_repeat_key, 0)
    if call.name not in _REPEAT_CAP_EXEMPT and _repeat_n >= _REPEAT_CALL_CAP:
        result: ToolResult = ToolResult.err(
            f"{call.name} has already been called {_repeat_n} "
            f"times this turn with identical arguments. Repeating "
            f"it makes no progress — the call is blocked. Change "
            f"your arguments or approach, or stop and report what "
            f"you have.",
            code="loop-guard-blocked",
        )
        state.blocked_streak += 1
        print(
            ui.telemetry(
                f"loop-guard: blocked repeated {call.name} "
                f"(#{_repeat_n + 1} identical this turn)"
            ),
            file=sys.stderr,
        )
    else:
        result = dispatch(call.name, call.arguments)
        # A call that actually ran proves progress — clear the consecutive-block
        # streak so a later block starts counting from zero again.
        state.blocked_streak = 0
    new_events = _TURN_MUTATIONS[pre_mutation_len:]
    if new_events:
        any_relevant, new_paths = _scan_new_mutations(new_events)
        if any_relevant:
            # The turn's FIRST relevant mutation is detected before the paths are
            # recorded below — both first-mutation verify steers arm at exactly
            # this moment, keyed on what had (or had not) been run to observe the
            # problem yet. The steers themselves are appended after the round's
            # tool results in _dispatch_round (never between a tool_calls row and
            # its results).
            first_mutation = not state.mutated_paths
            state.needs_verification = True
            state.mutated_paths |= new_paths
            if first_mutation:
                _maybe_arm_first_mutation_steer(state)
            _record_file_mutations(state, call, new_events)
            resolved_call_path = _lint_resolve_call_path(
                call, str(session.project_root)
            )
            if lint_pre is not None and resolved_call_path in new_paths:
                lint_suffix = _lint_delta_suffix(
                    lint_pre, call, str(session.project_root)
                )
    return result, lint_suffix


def _dispatch_round(
    session: Session,
    state: "_TurnState",
    response: ChatResponse,
    on_delta: Callable[[str], None] | None,
) -> None:
    """Dispatch this response's tool calls and record their results.

    Streams any intermediate assistant text once, dispatches the batch
    (concurrently when every call is parallel_safe, else sequentially via
    ``_dispatch_sequential_call`` with the per-call guards: loop-guard hard cap,
    reactive lint-delta, and H1/H5/S4 mutation tracking), runs the shared
    rendered-result pipeline on each result (oversize-result guard, loop-guard /
    repeat-call / web-search-focus steers, reactive lint-delta suffix), appends
    each to the transcript, then injects the LSP diagnostics summary.

    Two fold-surviving steers may be appended AFTER all tool results (never
    between an assistant tool_calls message and its results, which would break
    the OpenAI wire protocol), each riding the user role so it survives a
    compaction fold. Blocked-round steer: when this round refused at least one
    call at the hard cap but has not yet hit the escalation cap — the model's only
    remaining feedback that it is stuck. Reproduce-before-edit steer: when this
    round produced the turn's first relevant file
    mutation with zero verification runs so far — steering the model to reproduce
    the reported problem before it keeps editing on assumption.
    """
    calls = response.tool_calls
    # Intermediate assistant text on a tool-call iteration that did NOT stream
    # is still worth surfacing; streamed text already reached the sink.
    if on_delta is not None and response.text:
        if state.streamed == 0:
            on_delta(response.text)
        on_delta("\n")

    # Concurrent dispatch when the whole batch is read-only and thread-safe
    # (see Tool.parallel_safe). Any unsafe or unknown tool in the batch forces
    # the sequential path, preserving effect ordering.
    parallel_results: list[ToolResult] | None = None
    if len(calls) > 1 and all(
        getattr(get_tool(c.name), "parallel_safe", False) for c in calls
    ):
        for call in calls:
            print(
                ui.tool_call(f"Tool call: {call.name}({json.dumps(call.arguments)})"),
                file=sys.stderr,
            )
        with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
            parallel_results = list(
                pool.map(lambda c: dispatch(c.name, c.arguments), calls)
            )
        # A completed parallel-safe batch never blocks and is real progress —
        # clear any pending consecutive-block streak, same as a dispatch.
        state.blocked_streak = 0

    blocked_this_round = False
    # Snapshot the one-shot first-mutation steer flags before the loop so the
    # post-round block can tell whether THIS round is the one that flipped each
    # (and so should append that steer). A no-op on every later round once fired.
    repro_fired_before = state.repro_steer_fired
    no_failure_fired_before = state.no_failure_steer_fired
    for i, call in enumerate(calls):
        # Only ever populated on the sequential path (parallel_safe tools never
        # mutate, so there is nothing to lint-delta there).
        if parallel_results is not None:
            result = parallel_results[i]
            lint_suffix = ""
        else:
            result, lint_suffix = _dispatch_sequential_call(session, state, call)
        if result.code == "loop-guard-blocked":
            blocked_this_round = True

        if call.name in _VERIFICATION_TOOLS:
            _record_verification_run(state, call, result)
        if call.name == "record" and str(call.arguments.get("kind", "")).strip().lower() in (
            "decision",
            "spec",
        ):
            state.record_decision_or_spec_called = True

        rendered = render_tool_result(call.name, result)
        result, rendered = _guard_oversize_result(
            call.name, result, rendered, state.cap, state.est
        )
        rendered = _loop_guard_check(call.name, rendered, state.seen_errors)
        rendered = _repeat_call_check(
            call.name, call.arguments, result, rendered, state.seen_calls,
            state.seen_renders, state.compactions, state.mutation_events,
        )
        rendered, state.searches_without_read = _web_search_focus_check(
            call.name, result, rendered, state.searches_without_read
        )
        if lint_suffix:
            rendered = rendered + lint_suffix

        print(ui.tool_result(rendered), file=sys.stderr)

        session.append_tool_result(call.id, call.name, rendered)

    diagnostics_inject_summary(session)

    # Fold-surviving post-round steers (blocked-round + the two first-mutation
    # verify steers): all ride the user role — which
    # _prune_messages keeps across a compaction fold — and are appended HERE,
    # after every tool result and diagnostics_inject_summary, so their user row
    # lands after the last tool message and never between an assistant tool_calls
    # entry and its results (which would break the OpenAI wire protocol). Each is
    # driven by a flag settled during the loop above and fires at most once here.
    #
    # Blocked-round steer: this round refused a call at the hard cap but the streak has not yet
    # reached the escalation cap (which handle_user_message enforces after this
    # returns) — bounded at ~2 steers per turn.
    if blocked_this_round and 0 < state.blocked_streak < _BLOCKED_STREAK_CAP:
        session.append_steer(_BLOCKED_ROUND_STEER.format(n=_REPEAT_CALL_CAP))
    # Reproduce-before-edit steer: the turn's first relevant file mutation just landed with zero
    # verification runs so far — steer the model to reproduce before editing on.
    if state.repro_steer_fired and not repro_fired_before:
        session.append_steer(_REPRO_BEFORE_EDIT_STEER)
    # No-failure-observed steer: the turn's first relevant file mutation just
    # landed while every recorded run had passed on a bug-report task — steer the
    # model to demonstrate the reported failure before "fixing" what it never saw
    # fail. Mutually exclusive with the reproduce-before-edit steer above (see
    # _maybe_arm_first_mutation_steer).
    if state.no_failure_steer_fired and not no_failure_fired_before:
        session.append_steer(_NO_FAILURE_OBSERVED_STEER)


def _finalize_answer(
    session: Session,
    state: "_TurnState",
    response: ChatResponse,
    on_delta: Callable[[str], None] | None,
) -> str | None:
    """Run the no-tool-call gate cascade and return the turn's answer.

    In order: bounce once on an empty answer (empty-answer nudge), bounce once
    on an unverified file mutation (H1 verification nudge), then — on the second
    pass through — mark the S3 verify gate, surface an empty-answer placeholder,
    and deliver the answer through on_delta.

    Returns the final answer string (terminal — the caller returns it), or
    ``None`` when a nudge bounced (the caller loops again). ``append_assistant``
    for this response has already run in ``handle_user_message`` before this
    call, so the transcript holds the model's original, unprefixed text. The
    end-of-turn consolidation hook is NOT fired here — ``handle_user_message``
    owns it as the single choke point across every turn-exit path.
    """
    text = response.text or ""

    # Empty-answer retry: the model ended the turn with no text and no tool
    # calls — a common local-model failure mode (a bare stop token after
    # consuming tool results) that REPL mode would silently swallow (it discards
    # the return value and only on_delta delivers output, so an empty return
    # leaves the user at a blank prompt with the tool calls having visibly run).
    # Inject a harness steer and loop once more rather than re-rolling the
    # identical request (which risks a deterministic re-collapse). Bounded to a
    # single retry per turn; a second empty turn is surfaced via the placeholder
    # below.
    if not text.strip() and not state.empty_answer_nudge_fired:
        state.empty_answer_nudge_fired = True
        print(
            ui.telemetry("empty-answer-nudge: fired (empty assistant turn)"),
            file=sys.stderr,
        )
        session.append_steer(
            "You produced no answer this turn. Respond now with a concise "
            "summary of what you did or found, grounded in the tool results "
            "above. Do not call more tools unless a result is genuinely missing."
        )
        return None

    # H1 — post-mutation verification nudge: the model is about to end the turn
    # having mutated files without running anything to verify the change. Inject
    # a harness steer and do one more loop iteration instead of returning. Fires
    # at most once per turn.
    if state.needs_verification and not state.verification_nudge_fired:
        state.verification_nudge_fired = True
        # Load the verification tools this nudge names so the next round's
        # schemas() carries them and the guidance is actionable.
        _activate_verification_tools()
        print(
            ui.telemetry("verification-nudge: fired (unverified file mutation)"),
            file=sys.stderr,
        )
        session.append_steer(
            "You modified files this turn but ran nothing to verify the "
            "change. Verify it now with verify_scratch (a throwaway "
            "snippet, no file pollution), run_tests, or run_command "
            "against a separate script — never by adding repro/test code "
            "to a production file or repurposing its "
            "`if __name__ == \"__main__\"` block. These tools are loaded "
            "into your toolset now — call one directly. Or state explicitly "
            "in your answer that the change is unverified. Either way, end "
            "your answer with a one-line verification breakdown: what "
            "you checked (tests, commands, diagnostics) and what it "
            "showed."
        )
        return None

    # S3 — hard verify gate: this is the SECOND final answer of the turn (the
    # bounce above already fired and needs_verification is still set — a
    # run_tests/run_command/verify_scratch call never succeeded in between).
    # Accept it, but mark it: prefix the *returned* text with a harness-side
    # "[UNVERIFIED CHANGES] " so the caller sees the state, unless the model
    # already declared the change unverified in its own words (case-insensitive
    # match). The transcript already recorded the model's original, unprefixed
    # text — only the return value / on_delta payload gets the marker.
    final_text = text
    # S4 — record the UNPREFIXED answer before any marker is layered on; this is
    # the single source of truth the --json envelope reads back via
    # session.turn_report.
    state.turn_report["answer"] = final_text
    if state.needs_verification and state.verification_nudge_fired:
        state.turn_report["declared_unverified"] = "unverified" in final_text.lower()
        if not state.turn_report["declared_unverified"]:
            final_text = "[UNVERIFIED CHANGES] " + final_text
            print(
                ui.telemetry(
                    "verification-gate: unresolved after bounce — "
                    "marking [UNVERIFIED CHANGES]"
                ),
                file=sys.stderr,
            )
    # Net-change annotation + one shared formula for verified, both stamped at
    # this single turn-exit choke point (shared with the give-up paths via
    # _annotate_reverted_files / _turn_verified) so every exit reports the same
    # truth.
    _annotate_reverted_files(state)
    state.turn_report["verified"] = _turn_verified(state)

    # Empty-answer placeholder: if the model returned no text at all (the retry
    # above already fired once and still came back empty), surface a transparent
    # placeholder. Layered onto the return value / on_delta payload ONLY — the
    # transcript and turn_report["answer"] already hold the real empty string as
    # the source of truth. Uses the original response text (not the
    # possibly-prefixed final_text) so it overrides a vacuous
    # "[UNVERIFIED CHANGES] " prefix too.
    if not text.strip():
        final_text = "(no response from model)"
        print(
            ui.telemetry("empty-answer: no text after retry — surfacing placeholder"),
            file=sys.stderr,
        )

    if on_delta is not None and final_text:
        if state.streamed == 0:
            on_delta(final_text)  # non-streaming / gated fallback: deliver whole
        on_delta("\n")
    return final_text


# =============================================================================
# Main entry point
# =============================================================================

def handle_user_message(
    text: str,
    session: Session,
    client: LLMClient,
    verbose: bool = False,
    compaction_cfg: dict | None = None,
    on_delta: Callable[[str], None] | None = None,
) -> str:
    """Execute one agent turn in response to a user message.

    The loop follows this contract exactly:

    1. **setup** — append_user(text), orientation_maybe_seed(session).
    2. **loop** (while True): assemble + chat + dispatch, until one of the three
       turn-exit paths below settles ``answer`` and breaks. Each exit path has
       already appended its final assistant text to the transcript and recorded
       ``turn_report["answer"]`` before it breaks:
       a. **over-cap exit** — ``_run_llm_with_compaction`` returns a string when
          the compaction ladder cannot get the context under cap (or the
          provider keeps rejecting on length). That give-up message is the
          answer; the turn ends without retry.
       b. **normal finalize exit** — a no-tool-call response clears the
          empty-answer / H1 gate cascade in ``_finalize_answer``, which returns
          the final text (optionally "[UNVERIFIED CHANGES] "-marked or the
          "(no response from model)" placeholder). A bounced gate returns
          ``None`` and the loop continues instead.
       c. **blocked-loop escalation exit** — after ``_dispatch_round`` leaves
          ``state.blocked_streak >= _BLOCKED_STREAK_CAP`` (the model kept
          re-issuing a hard-cap-blocked call), ``_blocked_loop_giveup``
          force-finalizes the turn with a plain-text give-up answer.
    3. **single choke point** — after the loop breaks, the end-of-turn
       consolidation hook (``consolidation_maybe_extract``) fires exactly once,
       on EVERY exit path, so a turn force-finalized by (a)/(c) is mined into
       durable memory just like a normally-finalized (b) turn. It runs after the
       final answer / give-up message is settled and appended, so the transcript
       tail it snapshots includes it.

    Per no-tool-call round the assistant text is appended (append_assistant),
    the H5 graph-memory nudge may fire, and the gate cascade runs; per tool-call
    round each call is echoed to stderr, dispatched via the registry, its
    rendered result stored, then diagnostics_inject_summary(session) runs before
    the loop continues.

    Args:
        text: User message string to begin the turn with.
        session: Active ``Session`` holding the conversation transcript.
        client: An ``LLMClient`` instance for sending chat requests.
        verbose: If True, prints the full assembled context lines (prefixed by
            "assembled context") to stderr before each LLM call.
        on_delta: Optional sink for assistant text as it is produced. When set,
            the final answer is guaranteed to be delivered through it exactly
            once — streamed fragment-by-fragment when the provider streams,
            or as one whole-text call on the non-streaming fallback — followed
            by a single ``"\\n"``. Intermediate assistant text on tool-call
            iterations streams through it too. Callers that pass *on_delta*
            must NOT print the returned string again. The one client.chat call
            immediately after the S3 verify-gate bounce is always delivered
            whole (never streamed fragment-by-fragment), since whether it
            needs the "[UNVERIFIED CHANGES] " prefix can only be decided once
            the full response is in hand.

    Returns:
        The final assistant text string — either a normal turn response, a
        consolidation pass-through, the over-cap sentinel message, the
        blocked-loop give-up message (the escalation ladder force-finalizing a
        turn stuck at ``_BLOCKED_STREAK_CAP`` consecutive hard-cap blocks), or a
        turn response prefixed with "[UNVERIFIED CHANGES] " when the S3 hard
        verify gate bounced once and the follow-up answer still neither
        verified the mutation nor declared it unverified.
    """
    session.append_user(text)
    orientation_maybe_seed(session)

    # S4 — per-turn structured result report for --json one-shot mode, exposed
    # via ``session.turn_report`` so a caller (main.py) can build a result
    # envelope even when this call raises before returning — whatever was
    # accumulated up to the exception still reflects reality. ``answer`` holds
    # the UNPREFIXED text (see _TurnState / _finalize_answer for the single-
    # source-of-truth contract).
    turn_report: dict = {
        "files_changed": [],
        "verification_runs": [],
        "verified": None,
        "declared_unverified": False,
        "answer": None,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "llm_calls": 0},
    }
    session.turn_report = turn_report

    window = client.config.context_limit
    comp_cfg = compaction_cfg or {}
    state = _TurnState(
        turn_report=turn_report,
        cap=compaction.compute_cap(window, comp_cfg),
        window=window,
        comp_cfg=comp_cfg,
        user_message=text,
    )

    def _sink(piece: str) -> None:
        state.streamed += len(piece)
        on_delta(piece)  # type: ignore[misc]  # only ever passed when on_delta is set

    delta_cb = _sink if on_delta is not None else None

    # Every turn-exit path assigns ``answer`` and breaks to the single choke
    # point below (the end-of-turn consolidation hook), so the turn is mined into
    # memory exactly once regardless of which path finalized it. See this
    # function's docstring for the three exit paths.
    answer: str
    while True:
        # Recomputed every iteration so a mid-turn load_tool call is reflected in
        # the very next client.chat — a stale pre-loop snapshot would otherwise
        # withhold the just-loaded tool's schema until the following user turn.
        tool_schemas = schemas()

        # Compaction ladder + one chat call (with OverCapError retry). Returns
        # the give-up string when compaction is exhausted.
        outcome = _run_llm_with_compaction(
            session, client, state, tool_schemas, delta_cb, verbose, on_delta
        )
        if isinstance(outcome, str):
            # Over-cap exit: the give-up message is already in the transcript /
            # turn_report / on_delta.
            answer = outcome
            break
        response = outcome

        tool_calls: list[ToolCall] | None = (
            response.tool_calls if response.tool_calls else None
        )

        # H5 nudge must run before append_assistant records this turn's answer
        # (it amends the last tool result, which append_assistant would displace).
        if not response.tool_calls:
            _maybe_graph_memory_nudge(session, state)

        session.append_assistant(response.text or "", tool_calls=tool_calls)

        if not response.tool_calls:
            # No-tool-call gate cascade: bounce (empty-answer / H1) → loop again,
            # or produce the final answer (S3 mark / placeholder) → break to exit.
            final = _finalize_answer(session, state, response, on_delta)
            if final is not None:
                answer = final
                break
            continue

        _dispatch_round(session, state, response, on_delta)

        # Escalation ladder: the model has re-issued a call the hard cap keeps
        # refusing, ignoring both the block error and the fold-surviving steer.
        # Every further round is a wasted LLM round-trip that dispatches nothing,
        # so force-finalize the turn instead of looping unbounded (observed live:
        # the same call blocked hundreds of times until an external timeout).
        if state.blocked_streak >= _BLOCKED_STREAK_CAP:
            answer = _blocked_loop_giveup(session, state, on_delta)
            break

    # Single choke point: mine this settled turn — its diff plus the
    # transcript tail, now including the final answer / give-up message — into
    # durable memory exactly once, on every exit path (normal finalize,
    # blocked-loop escalation, over-cap). Force-finalized turns did real work
    # too, so they must be consolidated, not silently dropped.
    consolidation_maybe_extract(session, client)
    return answer
