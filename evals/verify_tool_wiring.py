"""Verify-tool wiring check: harness steers name only tools the active mode carries.

Two harness steers tell the model to verify its edits by name — the reproduce-
before-edit steer names ``run_command`` directly; the H1 post-mutation
verification nudge names whichever subset of run_command/run_tests/
verify_scratch the active mode's ``_available_verification_tools()``
reports. Tools are now a static per-mode set fixed for the life of the process
(``registry.activate_mode``, ``modes.py``) — there is no more load_tool-style
dynamic activation, so a steer naming a tool absent from the active mode would
hand the model an uncallable instruction with no rescue path. This script
proves the opposite: every steer only ever names tools the active mode
actually carries, and the tool it does name is genuinely callable end to end.

This script drives the *real* production hot path (``agent.handle_user_message``
with a stub LLM client, no network) plus direct calls into the real
``turn``/``tools.registry`` module functions, and asserts:

a. ``turn.verification._available_verification_tools()`` returns exactly the
   documented per-mode set: research -> [] (no mutating tools, nothing to
   verify); code -> ['run_command', 'verify_scratch'] (code mode's own
   instructions forbid running the suite, so never run_tests); test ->
   ['run_command', 'run_tests', 'verify_scratch'] (all three); performance_debug
   -> ['run_command'] (profiling tools are not verification tools).

b. The H1 nudge text built from each mode's available list names exactly
   those tools and nothing else — code mode's nudge never mentions run_tests
   or run_tests' clause text; test mode's does; performance_debug's names only
   run_command.

c. End-to-end through the real turn loop, in 'code' mode: an edit with zero
   prior verification runs fires the reproduce-before-edit steer, and the very
   next round calls run_command DIRECTLY — it dispatches successfully
   (verification_runs records status=success), proving the steer named a tool
   that was already callable (mode-static from process start), not one that
   needed a since-removed per-call activation step.

d. A read-only turn in the same mode never fires the steer and never touches
   verification_runs — the happy path is undisturbed.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before
exec'ing this file); (c)/(d) chdir into their own throwaway temp project dir
(create_file/run_command resolve against cwd) and restore the cwd afterward,
disable memory side effects for the run, and touch no repo files. Each check
restores whichever mode it found active in a finally, so the four checks
sharing one process never contaminate each other's precondition.
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


def check_available_verification_tools_per_mode() -> list[str]:
    """a. _available_verification_tools() matches the documented per-mode set."""
    import tools.registry as registry
    from turn.verification import _available_verification_tools

    failures: list[str] = []
    expected = {
        "research": [],
        "code": ["run_command", "verify_scratch"],
        "test": ["run_command", "run_tests", "verify_scratch"],
        "performance_debug": ["run_command"],
    }

    saved_mode = registry.current_mode()
    try:
        for mode_name, expected_tools in expected.items():
            registry.activate_mode(mode_name)
            available = _available_verification_tools()
            if available != expected_tools:
                failures.append(
                    f"mode {mode_name!r}: expected _available_verification_tools() "
                    f"== {expected_tools!r}, got {available!r}"
                )
    finally:
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    return failures


def check_nudge_names_only_available_tools() -> list[str]:
    """b. The H1 nudge text names exactly the active mode's available tools."""
    import tools.registry as registry
    from turn.steering import _verification_nudge_text
    from turn.verification import _available_verification_tools

    failures: list[str] = []
    saved_mode = registry.current_mode()
    try:
        registry.activate_mode("code")
        code_text = _verification_nudge_text(_available_verification_tools())
        if "run_tests" in code_text:
            failures.append(
                f"code mode's H1 nudge names run_tests, which code mode does not "
                f"carry: {code_text!r}"
            )
        if "run_command" not in code_text:
            failures.append(f"code mode's H1 nudge does not name run_command: {code_text!r}")
        if "verify_scratch" not in code_text:
            failures.append(f"code mode's H1 nudge does not name verify_scratch: {code_text!r}")

        registry.activate_mode("test")
        test_text = _verification_nudge_text(_available_verification_tools())
        if "run_tests" not in test_text:
            failures.append(f"test mode's H1 nudge does not name run_tests: {test_text!r}")
        if "run_command" not in test_text:
            failures.append(f"test mode's H1 nudge does not name run_command: {test_text!r}")
        if "verify_scratch" not in test_text:
            failures.append(f"test mode's H1 nudge does not name verify_scratch: {test_text!r}")

        registry.activate_mode("performance_debug")
        perf_text = _verification_nudge_text(_available_verification_tools())
        if "run_tests" in perf_text:
            failures.append(
                f"performance_debug mode's H1 nudge names run_tests, which it "
                f"does not carry: {perf_text!r}"
            )
        if "verify_scratch" in perf_text:
            failures.append(
                f"performance_debug mode's H1 nudge names verify_scratch, which "
                f"it does not carry: {perf_text!r}"
            )
        if "run_command" not in perf_text:
            failures.append(
                f"performance_debug mode's H1 nudge does not name run_command: {perf_text!r}"
            )
    finally:
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    return failures


def check_repro_steer_fires_and_tool_is_callable() -> list[str]:
    """c. In code mode, the repro steer fires and names a tool already callable."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient, disable_memory_hooks
    disable_memory_hooks()
    from session import Session

    failures: list[str] = []

    # create_file and run_command are both declared by 'code' mode (modes.py)
    # from the start of the process — mode-gating has no per-tool activation
    # step, so run_command is already callable in round 2 with nothing needing
    # to fire at round 1's steer to unlock it.
    saved_mode = registry.current_mode()
    registry.activate_mode("code")

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="verify-wiring-fires-")
    os.chdir(tmp)
    try:
        session = Session(tmp, "test-model", "You are a test agent.")
        # Round 1 edits (no run yet -> repro steer fires); round 2 calls
        # run_command DIRECTLY; round 3 ends the turn.
        script = [
            _tool_call_response("create_file", {"path": "buggy.py", "content": "x = 1\n"}),
            _tool_call_response("run_command", {"cmd": "echo verified"}),
            _final_answer_response("Edited then verified."),
        ]
        client = _StubClient(script)

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                "fix the wrong output", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )
        stderr = buf.getvalue()

        if "repro-steer: fired" not in stderr:
            failures.append(
                "reproduce-before-edit steer did not fire — check (c) premise "
                "(an unverified edit steers) is not met"
            )

        # The direct run_command call must have dispatched: verification_runs
        # records it with status success (a not-in-mode rejection would record
        # status=error instead, because the recording branch still fires).
        runs = session.turn_report["verification_runs"]
        rc_runs = [r for r in runs if r.get("tool") == "run_command"]
        if not rc_runs:
            failures.append(
                "run_command produced no verification_runs entry — the direct "
                "call named by the steer never reached dispatch"
            )
        elif not any(r.get("status") == "success" for r in rc_runs):
            failures.append(
                f"run_command was called directly after the steer but did not "
                f"succeed (statuses: {[r.get('status') for r in rc_runs]}) — the "
                f"steer named a tool that was not actually callable"
            )

        # Belt-and-braces: no tool-result row may carry a not-in-mode rejection.
        rc_rows = [
            m
            for m in session._messages
            if m.get("role") == "tool" and m.get("name") == "run_command"
        ]
        if any("not-in-mode" in str(m.get("content", "")) for m in rc_rows):
            failures.append(
                "a run_command tool result was a not-in-mode rejection — the "
                "steer named a tool outside the active mode"
            )
    finally:
        os.chdir(original_cwd)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    return failures


def check_no_steer_on_readonly_turn() -> list[str]:
    """d. A read-only turn never fires the repro steer or touches verification_runs."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient, disable_memory_hooks
    disable_memory_hooks()
    from session import Session

    failures: list[str] = []

    saved_mode = registry.current_mode()
    registry.activate_mode("code")

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="verify-wiring-noop-")
    os.chdir(tmp)
    try:
        with open(os.path.join(tmp, "readme.txt"), "w", encoding="utf-8") as fh:
            fh.write("hello\n")

        session = Session(tmp, "test-model", "You are a test agent.")
        script = [
            _tool_call_response("read_file", {"path": "readme.txt"}),
            _final_answer_response("Read the file."),
        ]
        client = _StubClient(script)

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                "what does readme.txt say", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )
        stderr = buf.getvalue()

        if "repro-steer: fired" in stderr:
            failures.append(
                "reproduce-before-edit steer fired on a read-only turn (no "
                "mutation happened) — the happy path was disturbed"
            )
        if session.turn_report["verification_runs"]:
            failures.append(
                f"verification_runs is non-empty after a read-only turn: "
                f"{session.turn_report['verification_runs']!r}"
            )
    finally:
        os.chdir(original_cwd)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("available-verification-tools-per-mode", check_available_verification_tools_per_mode),
        ("nudge-names-only-available-tools", check_nudge_names_only_available_tools),
        ("repro-steer-fires-and-tool-is-callable", check_repro_steer_fires_and_tool_is_callable),
        ("no-steer-on-readonly-turn", check_no_steer_on_readonly_turn),
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
        "PASS: _available_verification_tools() matches the documented per-mode "
        "set (research=[], code=[run_command,verify_scratch], "
        "test=all three, performance_debug=[run_command]); the H1 nudge names "
        "exactly a mode's available tools (code never names run_tests); the "
        "repro steer fires and names a tool that is genuinely callable next "
        "round; and a read-only turn never fires it"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
