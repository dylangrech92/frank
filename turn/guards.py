"""Per-turn guards that catch a model going in circles and steer it out.

Three reactive, zero-cost-on-the-happy-path checks applied to a dispatched tool
result: ``_loop_guard_check`` owns repeated identical *failures*,
``_repeat_call_check`` owns repeated identical *successes*, and
``_call_signature`` gives both a stable counting key. They belong together
because they share the per-turn counting dicts and the ``[loop-guard]`` steer
vocabulary, and stay separate from ``turn.repeat_dedup``, which owns only how a
repeated render is stubbed or suffixed.
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
from turn.verification import _REPEAT_CAP_EXEMPT, _VERIFICATION_TOOLS


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
