"""Dispatch-level check for the success-path loop-guard steer (no LLM required).

Mirrors ``inline_loop_guard.py`` (which covers the repeated-identical-*failure*
path owned by ``_loop_guard_check``). This one covers the
repeated-identical-*success* path owned by ``_repeat_call_check``: the no-op
loop where a model re-issues the exact same successful call (e.g.
``edit_file`` with search == replace) over and over. A passive steer alone
does not reliably break a determined loop, so ``_REPEAT_CALL_CAP`` and the
exempt set are also asserted here; the dispatch-time hard block itself is read
from ``turn.guards._repeat_cap_block_count`` by both dispatch paths and is
exercised end-to-end by ``runaway_bounds.py``.

Imports ``agent`` directly, drives two identical successful
``_repeat_call_check`` calls with the same arguments, and asserts:

    1st occurrence — rendered result has NO "[loop-guard]" suffix.
    2nd occurrence — rendered result HAS a "[loop-guard]" suffix.
    error results — never get the success-path suffix (owned by
                    ``_loop_guard_check``).
    constants    — ``_REPEAT_CALL_CAP`` is set and run_command/run_tests are
                    exempt from the hard cap.

Exits 0 on success, 1 on any assertion failure or unexpected exception. Runs
with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file), and touches no repo files.
"""

from __future__ import annotations

import sys


def main() -> int:
    import agent
    from tools.result import ToolResult

    ok = ToolResult.ok("Replaced 1 occurrence in app.py", path="app.py", occurrences=1)
    arguments = {"path": "app.py", "search": "x", "replace": "x"}
    rend = "[edit_file(success)] Replaced 1 occurrence in app.py"

    seen: dict[tuple[str, str], int] = {}
    seen_renders: dict[tuple[str, str], tuple[str, int, int]] = {}
    rendered1 = agent._repeat_call_check("edit_file", arguments, ok, rend, seen, seen_renders, 0, 0)
    rendered2 = agent._repeat_call_check("edit_file", arguments, ok, rend, seen, seen_renders, 0, 0)

    ok_flag = True

    if "[loop-guard]" in rendered1:
        print("FAIL: first successful occurrence already carries a [loop-guard] suffix", file=sys.stderr)
        print(f"  rendered1={rendered1!r}", file=sys.stderr)
        ok_flag = False

    if "[loop-guard]" not in rendered2:
        print("FAIL: second identical successful call did not get a [loop-guard] suffix", file=sys.stderr)
        print(f"  rendered2={rendered2!r}", file=sys.stderr)
        ok_flag = False

    # Errors must NOT be steered by the success-path guard (owned by
    # _loop_guard_check) — drive two identical error results and assert no
    # success-path suffix appears on either.
    err = ToolResult.err("boom", code="boom")
    erend = "[edit_file(error code=boom)] boom"
    seen_err: dict[tuple[str, str], int] = {}
    seen_renders_err: dict[tuple[str, str], tuple[str, int, int]] = {}
    e1 = agent._repeat_call_check("edit_file", arguments, err, erend, seen_err, seen_renders_err, 0, 0)
    e2 = agent._repeat_call_check("edit_file", arguments, err, erend, seen_err, seen_renders_err, 0, 0)
    if "[loop-guard]" in e1 or "[loop-guard]" in e2:
        print("FAIL: error result was steered by the success-path guard", file=sys.stderr)
        print(f"  e1={e1!r} e2={e2!r}", file=sys.stderr)
        ok_flag = False

    # Hard-cap configuration: a finite cap guarantees termination, and the
    # verification tools are exempt (a rebuild/retest cycle legitimately
    # repeats an identical command).
    if not isinstance(agent._REPEAT_CALL_CAP, int) or agent._REPEAT_CALL_CAP < 2:
        print(f"FAIL: _REPEAT_CALL_CAP must be >= 2, got {agent._REPEAT_CALL_CAP!r}", file=sys.stderr)
        ok_flag = False
    from turn.verification import _REPEAT_CAP_EXEMPT

    if not {"run_command", "run_tests"} <= _REPEAT_CAP_EXEMPT:
        print(f"FAIL: run_command/run_tests must be exempt, got {_REPEAT_CAP_EXEMPT!r}", file=sys.stderr)
        ok_flag = False

    # Signature must be order-independent so dict insertion order never splits
    # what is logically the same call into two counts. Read from its home module:
    # both dispatch paths key on it via turn.guards, agent.py no longer does.
    from turn.guards import _call_signature

    if _call_signature({"a": 1, "b": 2}) != _call_signature({"b": 2, "a": 1}):
        print("FAIL: _call_signature is not order-independent", file=sys.stderr)
        ok_flag = False

    if ok_flag:
        print("PASS: success-path loop-guard absent on 1st, present on 2nd; errors exempt; cap configured")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
