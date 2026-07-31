"""Per-turn guards that catch a model going in circles and steer it out.

Two layers, both scoped to a single turn.

*Reactive steers* applied to a dispatched tool result: ``_loop_guard_check``
owns repeated identical *failures*, ``_repeat_call_check`` owns repeated
identical *successes*, and ``_call_signature`` gives both a stable counting key.

*Runaway bounds* that end a turn the steers failed to break:
``_repeat_cap_block_count`` is the one enforcement reader of the identical-call
tally (both dispatch paths ask it, so the cap cannot apply on one and not the
other), ``_round_abandoned_result`` stands in for the calls a bounded round
never dispatches, and ``_text_runaway_count`` catches the loop no call-level
tally can see — a model re-emitting identical narration round after round.

They belong together because they share the per-turn counting dicts and the
``[loop-guard]`` vocabulary, and stay separate from ``turn.repeat_dedup``, which
owns only how a repeated render is stubbed or suffixed.
"""

from __future__ import annotations

import json

from tools.registry import get_tool
from tools.result import ToolResult
from turn.repeat_dedup import (
    _REPEAT_STEER_SUFFIX,
    _render_fingerprint,
    _repeat_render,
)
from turn.verification import (
    _REPEAT_CALL_CAP,
    _REPEAT_CAP_EXEMPT,
    _VERIFICATION_TOOLS,
)

# Hard cap on identical non-empty assistant texts emitted on TOOL-BEARING rounds
# within one turn. A model can loop without ever repeating a tool call — it
# re-narrates the same plan verbatim and re-issues calls that differ just enough
# to slip the identical-call tally — so the call-level guards are blind to it.
# Measured across this harness's own session logs: three turns looped on
# identical narration (3x, 4x and 6x), every one of them carrying tool calls, and
# none tripped any existing guard. Deliberately NOT applied to the terminal
# no-tool-call round: a final answer that happens to echo earlier narration is a
# legitimate ending, not a loop.
_TEXT_RUNAWAY_CAP = 3


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


def _repeat_cap_block_count(
    name: str,
    arguments: dict,
    seen_calls: dict[tuple[str, str], int],
) -> int | None:
    """The prior identical-call tally when this call is over the hard cap, else None.

    The single ENFORCEMENT reader of ``seen_calls`` (``_repeat_call_check``
    remains its single writer), shared by both dispatch paths so the cap cannot
    apply on one and silently not the other. It previously lived inline on the
    sequential path only, so an all-``parallel_safe`` batch — which skips that
    path entirely — was never checked: one measured turn issued the same
    ``read_file`` 102 times against a cap of 3, and the whole session recorded 8
    blocks.

    Args:
        name: Registered tool name.
        arguments: The call's parsed arguments.
        seen_calls: The turn's ``(name, signature) -> count`` tally.

    Returns:
        ``None`` when the call may dispatch — either an exempt tool (a
        build/test or measure/re-measure cycle legitimately repeats an identical
        call) or fewer than ``_REPEAT_CALL_CAP`` identical calls so far.
        Otherwise the count of previous identical calls, which the caller quotes
        back in the block message.
    """
    if name in _REPEAT_CAP_EXEMPT:
        return None
    count = seen_calls.get((name, _call_signature(arguments)), 0)
    return count if count >= _REPEAT_CALL_CAP else None


def _batch_exceeds_repeat_cap(calls, seen_calls: dict[tuple[str, str], int]) -> bool:
    """True when dispatching *calls* as one concurrent batch would breach the cap.

    The concurrent path dispatches the whole batch before a single result is
    tallied, so it cannot enforce a cap the way the sequential loop does — where
    ``_repeat_call_check`` raises the count after each call and the next call
    meets it. This replays that same call-by-call tally over the batch on a
    private copy, and reports whether any call in it would have been refused.

    That covers both ways a batch breaches the cap, and the second is the one a
    per-call check alone misses: a call already at the cap from earlier rounds
    (measured: 102 identical ``read_file`` calls in one turn against a cap of 3),
    and a batch that repeats a call past the cap all by itself — the burst shape,
    where the whole loop arrives in a single message and no earlier round exists
    to have tallied anything.

    A True answer sends the batch down the sequential path, which blocks the
    surplus calls and lets the round bound end the turn. Losing concurrency there
    is the point: a batch that repeats one call past the cap is a loop, not work.
    """
    tally = dict(seen_calls)
    for call in calls:
        if _repeat_cap_block_count(call.name, call.arguments, tally) is not None:
            return True
        if call.name not in _REPEAT_CAP_EXEMPT:
            key = (call.name, _call_signature(call.arguments))
            tally[key] = tally.get(key, 0) + 1
    return False


def _round_abandoned_result(blocked_streak: int) -> ToolResult:
    """Stand-in result for a call a bounded round never dispatched.

    The hard cap refuses one call at a time, which bounds nothing when a single
    assistant message carries the whole loop: one measured turn arrived with
    **2,555 tool calls**, 2,500 of them byte-identical, and the harness dutifully
    walked all of them — 2,497 blocked, one transcript row each — because the
    escalation ladder is only evaluated after the round returns. So the round
    itself stops at ``_BLOCKED_STREAK_CAP`` and the remaining calls get this
    instead of a dispatch.

    Every remaining call still gets a result row: the wire protocol pairs one
    tool message to every id in the assistant's ``tool_calls``, and a session is
    resumable, so a round that simply stopped emitting would leave a transcript
    the next request cannot send. Worded as a plain statement of fact with no
    steer — the escalation ladder force-finalizes the turn the moment the round
    returns, so no model ever reads this to act on it.
    """
    return ToolResult.err(
        f"Not dispatched — the harness stopped this round after {blocked_streak} "
        f"consecutive blocked calls and is ending the turn.",
        code="round-abandoned",
    )


def _text_runaway_count(text: str, seen_texts: dict[str, int]) -> int:
    """Tally one tool-bearing round's assistant text; report a runaway at the cap.

    Companion to the call-level guards, which cannot see a model that loops on
    prose (see ``_TEXT_RUNAWAY_CAP`` for the measured cases). Callers must invoke
    this ONLY on rounds that carry tool calls — a terminal answer is allowed to
    repeat earlier narration.

    Args:
        text: The round's assistant text (empty and whitespace-only are ignored,
            since a tool-only round legitimately emits no prose every time).
        seen_texts: The turn's ``stripped text -> count`` tally, mutated in place.

    Returns:
        ``0`` while the turn may continue; otherwise the emission count that
        reached ``_TEXT_RUNAWAY_CAP``, for the caller to quote in its give-up.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0
    seen_texts[stripped] = seen_texts.get(stripped, 0) + 1
    count = seen_texts[stripped]
    return count if count >= _TEXT_RUNAWAY_CAP else 0


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
