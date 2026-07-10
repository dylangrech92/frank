"""Verify-tool wiring check, end-to-end through the real turn loop.

Two harness steers tell the model to verify its edits by name — the reproduce-
before-edit steer names ``run_command``; the H1 verification nudge names
``verify_scratch`` / ``run_tests`` / ``run_command``. Those tools are catalog-
gated: they only enter the request's tools array once activated via ``load_tool``.
Before this change a steer fired without activating them, demanding a tool the
model could not call. The wiring connects the two fire sites to ``agent._activate_
verification_tools()`` so the very next round's ``schemas()`` carries the tools.

This script drives the *real* production hot path (``agent.handle_user_message``
with a stub LLM client, no network) and asserts:

a. Steer-fire activates — a scripted model that edits a file with NO prior
   verification run gets the reproduce-before-edit steer, and can then call
   ``run_command`` DIRECTLY in the next round with no ``load_tool`` in between:
   the call dispatches (verification_runs records it status=success, not a
   not-loaded error). run_command is NOT pre-activated in this scenario's setup —
   the point is that the steer fire activates it.

b. No-steer, no-activation — a turn that only reads a file (read_file is PINNED;
   no mutation, so the repro steer never fires) leaves ``run_command`` inactive
   afterward. The happy path is untouched.

c. Idempotence — calling ``agent._activate_verification_tools()`` twice raises
   nothing and leaves each verification tool in ``schemas()`` exactly once (no
   duplicate schema entries).

Registry-reset mechanism: the tools registry is module-global and its active set
(``tools.registry._active``) is the documented single-session home for loaded
tools, with no public deactivate API. Inline evals already run each as their own
subprocess (evals/run.py), so cross-scenario contamination cannot happen; but the
three checks below share one process, and check (a) intentionally activates
run_command. So each check first reaches into ``registry._active`` to clear the
verification tools (mirroring how these evals already reach transcript internals
like ``session._messages``), establishing a clean precondition, and restores the
original active set in a finally. Check (a) additionally runs first for good
measure. This is the clean mechanism the registry allows.

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


def _reset_verification_tools(registry) -> set[str]:
    """Clear the verification tools from the active set; return the prior set.

    Establishes a clean precondition so a check's assertions do not observe a
    verification tool another check activated earlier in this same process. The
    caller restores the returned snapshot in a finally.
    """
    import agent

    saved = set(registry._active)
    for name in agent._VERIFICATION_TOOLS:
        registry._active.discard(name)
    return saved


def _restore_active(registry, saved: set[str]) -> None:
    """Restore ``registry._active`` to the snapshot taken by _reset_verification_tools."""
    registry._active.clear()
    registry._active.update(saved)


def check_steer_fire_activates() -> list[str]:
    """a. The repro steer fire activates run_command so the next round can call it."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient
    from session import Session

    failures: list[str] = []

    saved = _reset_verification_tools(registry)
    # create_file is a catalog tool; a live model activates it via load_tool
    # before use. Do the same so the round-1 edit dispatches. run_command is
    # deliberately NOT activated here — the steer fire must activate it.
    registry.activate("create_file")

    original_cwd = os.getcwd()
    original_memory = agent.MEMORY_ENABLED
    tmp = tempfile.mkdtemp(prefix="verify-wiring-fires-")
    os.chdir(tmp)
    agent.MEMORY_ENABLED = False
    try:
        if registry.is_loaded("run_command"):
            failures.append(
                "precondition broken: run_command was already loaded before the "
                "steer fired (reset mechanism failed)"
            )

        session = Session(tmp, "test-model", "You are a test agent.")
        # Round 1 edits (no run yet -> repro steer fires -> activates run_command);
        # round 2 calls run_command DIRECTLY (no load_tool); round 3 ends the turn.
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
                "reproduce-before-edit steer did not fire — check (a) premise "
                "(an unverified edit steers) is not met"
            )

        # The direct run_command call must have dispatched: verification_runs
        # records it with status success (a not-loaded rejection would record
        # status=error instead, because the recording branch still fires).
        runs = session.turn_report["verification_runs"]
        rc_runs = [r for r in runs if r.get("tool") == "run_command"]
        if not rc_runs:
            failures.append(
                "run_command produced no verification_runs entry — the direct "
                "call after the steer never reached dispatch"
            )
        elif not any(r.get("status") == "success" for r in rc_runs):
            failures.append(
                f"run_command was called directly after the steer but did not "
                f"succeed (statuses: {[r.get('status') for r in rc_runs]}) — the "
                f"steer fire did not activate the tool it names"
            )

        # Belt-and-braces: no tool-result row may carry a not-loaded rejection.
        rc_rows = [
            m
            for m in session._messages
            if m.get("role") == "tool" and m.get("name") == "run_command"
        ]
        if any("not-loaded" in str(m.get("content", "")) for m in rc_rows):
            failures.append(
                "a run_command tool result was a not-loaded rejection — the tool "
                "was not activated by the steer fire"
            )
    finally:
        agent.MEMORY_ENABLED = original_memory
        os.chdir(original_cwd)
        _restore_active(registry, saved)

    return failures


def check_no_steer_no_activation() -> list[str]:
    """b. A read-only turn never fires the steer and leaves run_command inactive."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient
    from session import Session

    failures: list[str] = []

    saved = _reset_verification_tools(registry)

    original_cwd = os.getcwd()
    original_memory = agent.MEMORY_ENABLED
    tmp = tempfile.mkdtemp(prefix="verify-wiring-noop-")
    os.chdir(tmp)
    agent.MEMORY_ENABLED = False
    try:
        # A real file to read so read_file (PINNED, no activation needed) succeeds.
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
        if registry.is_loaded("run_command"):
            failures.append(
                "run_command became active after a read-only turn — verification "
                "tools must only activate when a verify steer fires"
            )
    finally:
        agent.MEMORY_ENABLED = original_memory
        os.chdir(original_cwd)
        _restore_active(registry, saved)

    return failures


def check_idempotence() -> list[str]:
    """c. Activating twice raises nothing and never duplicates a schema entry."""
    import agent
    import tools.registry as registry

    failures: list[str] = []

    saved = _reset_verification_tools(registry)
    try:
        agent._activate_verification_tools()
        agent._activate_verification_tools()

        names = [
            s.get("function", {}).get("name")
            for s in registry.schemas()
        ]
        for tool in sorted(agent._VERIFICATION_TOOLS):
            count = names.count(tool)
            if count != 1:
                failures.append(
                    f"{tool} appears {count} time(s) in the schemas array after "
                    f"activating twice (expected exactly 1 — duplicate or missing)"
                )
    except Exception as exc:
        failures.append(
            f"_activate_verification_tools raised on repeat call: "
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        _restore_active(registry, saved)

    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("steer-fire-activates", check_steer_fire_activates),
        ("no-steer-no-activation", check_no_steer_no_activation),
        ("idempotence", check_idempotence),
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
        "PASS: a fired verify steer activates the tools it names (run_command "
        "callable directly next round); a read-only turn leaves them inactive; "
        "and activating twice never duplicates a schema entry"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
