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
from typing import Callable

from llm import ChatResponse, LLMClient, OverCapError, ToolCall
from session import Session
from tools.registry import dispatch, get_tool, schemas
from tools.result import ToolResult
from turn.lint_delta import (
    _lint_delta_suffix,
    _lint_pre_snapshot,
    _lint_resolve_call_path,
)
from turn.outcome import _blocked_loop_giveup, _over_cap_giveup, _stamp_turn_outcome
from turn.rendering import _guard_oversize_result, render_tool_result
from turn.repeat_dedup import (
    _REPEAT_STEER_SUFFIX,
    _render_fingerprint,
    _repeat_render,
)
from turn.steering import (
    _BLOCKED_ROUND_STEER,
    _BLOCKED_STREAK_CAP,
    _is_bug_report,
    _NO_FAILURE_OBSERVED_STEER,
    _REPRO_BEFORE_EDIT_STEER,
    _verification_nudge_text,
)
from turn.state import _TurnState
from turn.verification import (
    _available_verification_tools,
    _REPEAT_CALL_CAP,
    _REPEAT_CAP_EXEMPT,
    _run_failure_is_environment_noise,
    _VERIFICATION_TOOLS,
    _verification_run_passed,
)


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


def orientation_maybe_seed(session: Session) -> None:
    """Task-start memory orientation injection seam.

    Delegates to the orientation module, which builds a code-anchored brief --
    the zero-LLM derived project skeleton personalised to this turn's task,
    plus anchored knowledge atoms and graph decisions/specs recalled by a
    task-relative query -- and stashes it on ``session._orientation_block`` for
    injection by the orientation context provider. Never persisted to the
    transcript. Never raises.
    """
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


def _loop_guard_check(
    name: str,
    rendered: str,
    seen_errors: dict[tuple[str, str], int],
) -> str:
    """Append a steer suffix when the same (tool, error) has repeated within a turn.

    Cheap prefix check on the rendered text (mirrors a standard dispatch
    service pattern) so the success path pays nothing: only rendered error envelopes
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


def _arm_verify_steer(state: "_TurnState", flag_attr: str, telemetry: str) -> None:
    """Arm one first-mutation verify steer: flip its one-shot flag and emit its
    telemetry line.

    The shared mechanics behind both first-mutation steers so the two conditions
    in ``_maybe_arm_first_mutation_steer`` read declaratively. Both steers' text
    names ``run_command``, which every mode capable of mutating files also
    carries (see ``_REPRO_BEFORE_EDIT_STEER``), so there is no tool set to
    activate here — the tool is already in the request's static per-mode set.
    """
    setattr(state, flag_attr, True)
    print(ui.telemetry(telemetry), file=sys.stderr)


def _maybe_arm_first_mutation_steer(state: "_TurnState") -> None:
    """Arm at most one first-mutation verify steer for the turn.

    Called the moment the turn's FIRST relevant file mutation lands. The two
    steers are mutually exclusive and cover disjoint situations:

    * zero verification runs recorded so far -> reproduce-before-edit (observe
      the problem before editing on assumption);
    * >= 1 run recorded, at least one of which PASSED, and every non-passed run
      was shell-level environment noise (see _run_failure_is_environment_noise),
      on a task whose original request reads as a bug report -> no-failure-
      observed (the model is starting to "fix" a failure it has never seen fail).

    A genuinely failed run — one that actually exercised project code and failed
    — means a failure WAS observed this turn, so neither steer arms. Known limit:
    a turn whose ONLY runs are environment noise (nothing passed) arms nothing —
    the reproduce-before-edit steer owns the zero-runs case, and with no passing
    run we cannot assert the project is green, so we stay silent rather than
    steer on an unproven premise. Each flag is one-shot.
    """
    runs = state.turn_report["verification_runs"]
    if not runs:
        if not state.repro_steer_fired:
            _arm_verify_steer(
                state,
                "repro_steer_fired",
                "repro-steer: fired (mutation before any verification "
                "run this turn)",
            )
        return
    passed_or_noise_only = any(r["passed"] for r in runs) and all(
        r["passed"] or r["noise"] for r in runs
    )
    if (
        not state.no_failure_steer_fired
        and passed_or_noise_only
        and _is_bug_report(state.user_message)
    ):
        _arm_verify_steer(
            state,
            "no_failure_steer_fired",
            "no-failure-steer: fired (edit on a reported failure that has "
            "not been observed this turn)",
        )


def _record_verification_run(
    state: "_TurnState", call: ToolCall, result: ToolResult, project_root: str
) -> None:
    """Record one verification run and update the pass-based verify gate.

    Appends a ``verification_runs`` entry — ``tool``/``status``/``detail`` (kept
    verbatim; evals assert on them), a first-class ``passed`` bool from
    ``_verification_run_passed``, and a ``noise`` bool that is meaningful only
    when ``passed`` is False: True when the failure was shell-level environment
    noise that never reached project code (``_run_failure_is_environment_noise``)
    and always False for a passing run. ``noise`` is consumed by the
    no-failure-observed steer, which treats a noise-only failure as "no failure
    observed on the project". Clears ``needs_verification`` ONLY when the run
    passed: a run that merely completed but failed (a nonzero ``run_command``
    exit, a failing ``run_tests`` count, a ``verify_scratch`` snippet error) no
    longer counts as verification, so the gate stays open and no false
    ``verified: true`` can be stamped off a run that never went green.
    """
    if call.name == "run_command":
        detail = str(call.arguments.get("cmd", ""))
    else:
        detail = str(call.arguments.get("path") or ".")
    passed = _verification_run_passed(call.name, result)
    noise = (not passed) and _run_failure_is_environment_noise(
        call, result, project_root
    )
    state.turn_report["verification_runs"].append(
        {
            "tool": call.name,
            "status": result.status,
            "detail": detail,
            "passed": passed,
            "noise": noise,
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
            _record_verification_run(
                state, call, result, str(session.project_root)
            )
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
    # at most once per turn, and only names verification tools the active mode
    # actually carries (_available_verification_tools). If none are available —
    # unreachable in practice, since every mode with a mutating tool also carries
    # run_command — there is nothing truthful to steer with, so this falls
    # straight through to the S3 gate below instead of bouncing on a toolless
    # nudge.
    if state.needs_verification and not state.verification_nudge_fired:
        state.verification_nudge_fired = True
        available = _available_verification_tools()
        if available:
            print(
                ui.telemetry("verification-nudge: fired (unverified file mutation)"),
                file=sys.stderr,
            )
            session.append_steer(_verification_nudge_text(available))
            return None
        print(
            ui.telemetry(
                "verification-nudge: skipped (no verification tool in active mode)"
            ),
            file=sys.stderr,
        )

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
    # Net-change annotation + one shared formula for verified, both stamped in
    # order at this single turn-exit choke point via _stamp_turn_outcome (shared
    # with the give-up paths) so every exit reports the same truth the same way.
    _stamp_turn_outcome(state)

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
        # Recomputed every iteration for symmetry with the rest of the loop, but
        # the result is constant for the life of the process: the active mode's
        # tool set is fixed at launch (registry.activate_mode) and nothing
        # mid-turn can add to or remove from it.
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
