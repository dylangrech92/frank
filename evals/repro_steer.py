"""E13 reproduce-before-edit steer check, end-to-end through the real turn loop.

The H1 verification nudge only fires when the model tries to END a turn, so a run
that edits, re-edits, and never reaches the end-of-turn gate is never steered
toward observed-output-first debugging. E13 closes that gap upstream: the first
relevant file mutation of a turn that has run nothing to observe the problem gets
a fold-surviving steer, once per turn, appended after the round's tool results.
This script drives that against the *real* production hot path
(``agent.handle_user_message`` with a stub LLM client, no network) and asserts:

a. Fires — a scripted model that edits a file with NO prior verification run this
   turn gets the reproduce-before-edit steer appended (a user-role, STEER_PREFIX
   row carrying the steer body verbatim) and the ``repro-steer: fired`` telemetry
   line on stderr.

b. Suppressed after a run — a scripted model that calls run_command FIRST (with a
   deliberately FAILING command, to prove the property is exit-status-agnostic:
   ``verification_runs`` records every run regardless of exit code) and only then
   edits gets NO repro steer and NO telemetry line.

c. One-shot — a second file edit in scenario (a)'s same turn does not append a
   second steer; exactly one repro steer row exists after the turn.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file); it chdir's into its own throwaway temp project dir (create_file and
run_command resolve against cwd) and restores the cwd afterward, disables memory
side effects for the run, and touches no repo files.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile


def _tool_call_response(name: str, arguments: dict):
    """Return a factory yielding a ChatResponse that issues one tool call.

    Each invocation gets a fresh call id (the wire protocol pairs a tool result
    to its call id) but the same name+arguments.
    """
    from llm import ChatResponse, ToolCall

    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        tc = ToolCall(id=f"call-{counter['n']}", name=name, arguments=dict(arguments))
        return ChatResponse(text="", tool_calls=[tc])

    return factory


def _final_answer_response(text: str):
    """Return a factory yielding a no-tool-call ChatResponse (turn ends)."""
    from llm import ChatResponse

    def factory():
        return ChatResponse(text=text, tool_calls=[])

    return factory


def _steer_rows(session, steer_body: str) -> list[dict]:
    """User-role steer rows on the transcript whose content carries *steer_body*."""
    from session import STEER_PREFIX

    return [
        m
        for m in session._messages
        if m.get("role") == "user"
        and m.get("steer")
        and str(m.get("content", "")).startswith(STEER_PREFIX)
        and steer_body in str(m.get("content", ""))
    ]


def check_fires_and_one_shot() -> list[str]:
    """a. + c. First unverified edit steers; a second edit does not double-steer."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient
    from session import Session

    # create_file / run_command are catalog tools (not PINNED); a live model
    # activates them via load_tool before use. Do the same so dispatch runs them.
    registry.activate("create_file")
    registry.activate("run_command")

    failures: list[str] = []

    original_cwd = os.getcwd()
    original_memory = agent.MEMORY_ENABLED
    tmp = tempfile.mkdtemp(prefix="repro-steer-fires-")
    os.chdir(tmp)
    agent.MEMORY_ENABLED = False
    try:
        session = Session(tmp, "test-model", "You are a test agent.")
        # Two distinct edits in one turn, no run_command anywhere, then a plain
        # final answer. The first edit must steer; the second must not.
        script = [
            _tool_call_response("create_file", {"path": "buggy_a.py", "content": "x = 1\n"}),
            _tool_call_response("create_file", {"path": "buggy_b.py", "content": "y = 2\n"}),
            _final_answer_response("Edited both files."),
        ]
        client = _StubClient(script)

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                "fix the wrong output", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )
        stderr = buf.getvalue()

        rows = _steer_rows(session, agent._REPRO_BEFORE_EDIT_STEER)
        if len(rows) != 1:
            failures.append(
                f"expected exactly 1 reproduce-before-edit steer row after two "
                f"edits in one turn, found {len(rows)} (one-shot broken?)"
            )
        if stderr.count("repro-steer: fired") != 1:
            failures.append(
                f"expected the 'repro-steer: fired' telemetry line exactly once, "
                f"stderr had {stderr.count('repro-steer: fired')}"
            )
    finally:
        agent.MEMORY_ENABLED = original_memory
        os.chdir(original_cwd)

    return failures


def check_suppressed_after_run() -> list[str]:
    """b. A prior (failing) run_command suppresses the steer -- status-agnostic."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient
    from session import Session

    # create_file / run_command are catalog tools (not PINNED); a live model
    # activates them via load_tool before use. Do the same so dispatch runs them.
    registry.activate("create_file")
    registry.activate("run_command")

    failures: list[str] = []

    original_cwd = os.getcwd()
    original_memory = agent.MEMORY_ENABLED
    tmp = tempfile.mkdtemp(prefix="repro-steer-suppressed-")
    os.chdir(tmp)
    agent.MEMORY_ENABLED = False
    try:
        session = Session(tmp, "test-model", "You are a test agent.")
        # A FAILING command runs first (nonzero exit still records a verification
        # run), then an edit, then a plain final answer. No repro steer must fire.
        script = [
            _tool_call_response("run_command", {"cmd": "exit 7"}),
            _tool_call_response("create_file", {"path": "buggy_a.py", "content": "x = 1\n"}),
            _final_answer_response("Reproduced then edited."),
        ]
        client = _StubClient(script)

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                "fix the crash", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )
        stderr = buf.getvalue()

        rows = _steer_rows(session, agent._REPRO_BEFORE_EDIT_STEER)
        if rows:
            failures.append(
                f"reproduce-before-edit steer fired {len(rows)} time(s) despite a "
                f"prior run_command this turn (exit-status-agnostic suppression "
                f"broken)"
            )
        if "repro-steer: fired" in stderr:
            failures.append(
                "'repro-steer: fired' telemetry appeared despite a prior "
                "run_command this turn"
            )
    finally:
        agent.MEMORY_ENABLED = original_memory
        os.chdir(original_cwd)

    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("fires-and-one-shot", check_fires_and_one_shot),
        ("suppressed-after-run", check_suppressed_after_run),
    ):
        try:
            failures = fn()
        except Exception as exc:  # a raised exception is itself a failure
            failures = [f"{label} raised {type(exc).__name__}: {exc}"]
        for f in failures:
            print(f"FAIL [{label}]: {f}", file=sys.stderr)
        all_failures.extend(failures)

    if all_failures:
        return 1
    print(
        "PASS: an unverified edit steers reproduce-before-edit once; a prior "
        "(failing) run_command suppresses it; a second edit does not double-steer"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
