"""Dispatch-level check for the loop-guard steer (no LLM required).

The live loop-guard steer (``agent._loop_guard_check``) never reliably fires
in a real session — a cooperative model refuses to repeat an identical
failing call twice in one turn, so the trigger has to be driven directly at
the dispatch/agent-hook level instead of through a live conversation.

This script imports ``agent`` and ``tools.registry`` directly, drives two
identical failing dispatches of an unloaded tool through
``agent._loop_guard_check`` (mirroring exactly what ``handle_user_message``
does per tool call), and asserts:

    1st occurrence — rendered result has NO "[loop-guard]" suffix.
    2nd occurrence — rendered result HAS a "[loop-guard]" suffix.

Exits 0 on success, 1 on any assertion failure or unexpected exception. Runs
with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file), and touches no repo files.
"""

from __future__ import annotations

import sys


def main() -> int:
    import agent
    from tools.registry import dispatch
    from agent import render_tool_result

    # 'read_file' is a real registered tool that is never loaded here, so
    # dispatch() always returns the same not-loaded error — a stable,
    # reproducible (name, rendered) pair for the loop-guard counter.
    name = "read_file"
    arguments = {"path": "does_not_matter.py"}

    seen_errors: dict[tuple[str, str], int] = {}

    result1 = dispatch(name, arguments)
    rendered1 = render_tool_result(name, result1)
    rendered1 = agent._loop_guard_check(name, rendered1, seen_errors)

    result2 = dispatch(name, arguments)
    rendered2 = render_tool_result(name, result2)
    rendered2 = agent._loop_guard_check(name, rendered2, seen_errors)

    ok = True

    if "[loop-guard]" in rendered1:
        print("FAIL: first occurrence already carries a [loop-guard] suffix", file=sys.stderr)
        print(f"  rendered1={rendered1!r}", file=sys.stderr)
        ok = False

    if "[loop-guard]" not in rendered2:
        print("FAIL: second identical failing dispatch did not get a [loop-guard] suffix", file=sys.stderr)
        print(f"  rendered2={rendered2!r}", file=sys.stderr)
        ok = False

    if ok:
        print("PASS: loop-guard absent on 1st occurrence, present on 2nd")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
