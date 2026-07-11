"""Reproduce-before-edit steer check, end-to-end through the real turn loop.

The H1 verification nudge only fires when the model tries to END a turn, so a run
that edits, re-edits, and never reaches the end-of-turn gate is never steered
toward observed-output-first debugging. The reproduce-before-edit steer closes
that gap upstream: the first
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

d. No-failure-observed fires — a bug-report task whose only runs this turn PASSED
   gets the no-failure-observed steer on its first edit (once, user role, with
   telemetry) and NOT the reproduce-before-edit steer (a run was recorded).

e. No-failure-observed suppressed by a genuine failing run — a recorded
   nonzero-exit run that actually exercised project code means a failure WAS
   observed, so neither first-mutation steer fires.

f. No-failure-observed scoped to bug reports — a pure feature task (no bug-report
   lexicon) never gets the no-failure-observed steer.

g. Gate honesty — a mutation followed by ONLY a nonzero-exit run_command leaves
   ``turn_report['verified']`` False (a failing run does not clear the verify
   gate); a passing run flips it True. The failing run is still recorded, marked
   ``passed=False``.

h. Environment-noise failures do not count as an observed project failure. A run
   that failed only at the shell level — a `cd` into a directory that does not
   exist, a `git` command in a tree with no `.git`, or a command-not-found (exit
   127) — never reached project code, so when a real run also PASSED this turn
   the no-failure-observed steer still fires (h1 cd, h2 git, h3 exit-127). The
   noise run is still recorded with ``passed=False``. A genuine nonzero-exit run
   (not 127, no `cd` prefix, not `git`) alongside a passing run is NOT noise and
   still suppresses the steer (h4).

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
    from evals._stub import _StubClient, disable_memory_hooks
    disable_memory_hooks()
    from session import Session

    # create_file / run_command are catalog tools (not PINNED); a live model
    # activates them via load_tool before use. Do the same so dispatch runs them.
    registry.activate("create_file")
    registry.activate("run_command")

    failures: list[str] = []

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="repro-steer-fires-")
    os.chdir(tmp)
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
        # Mutual exclusion: zero runs is the reproduce-before-edit steer's
        # territory, so the no-failure-observed steer must NOT also fire even
        # though the task ("fix the wrong output") reads as a bug report.
        nf_rows = _steer_rows(session, agent._NO_FAILURE_OBSERVED_STEER)
        if nf_rows:
            failures.append(
                f"no-failure-observed steer fired {len(nf_rows)} time(s) on a "
                f"zero-runs turn — it must defer to the reproduce-before-edit "
                f"steer (mutual exclusion broken)"
            )
    finally:
        os.chdir(original_cwd)

    return failures


def check_suppressed_after_run() -> list[str]:
    """b. A prior (failing) run_command suppresses the steer -- status-agnostic."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient, disable_memory_hooks
    disable_memory_hooks()
    from session import Session

    # create_file / run_command are catalog tools (not PINNED); a live model
    # activates them via load_tool before use. Do the same so dispatch runs them.
    registry.activate("create_file")
    registry.activate("run_command")

    failures: list[str] = []

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="repro-steer-suppressed-")
    os.chdir(tmp)
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
        os.chdir(original_cwd)

    return failures


def _drive_turn(task: str, script: list, tmp_prefix: str):
    """Drive one scripted turn through the real turn loop; return (session, stderr).

    Activates the catalog tools the scripts use, isolates cwd + memory, and
    captures stderr so the caller can assert on both transcript rows and
    telemetry. Restores cwd/memory afterward. Touches no repo files.
    """
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient, disable_memory_hooks
    disable_memory_hooks()
    from session import Session

    registry.activate("create_file")
    registry.activate("run_command")

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix=tmp_prefix)
    os.chdir(tmp)
    try:
        session = Session(tmp, "test-model", "You are a test agent.")
        client = _StubClient(script)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                task, session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )
        return session, buf.getvalue()
    finally:
        os.chdir(original_cwd)


def check_no_failure_steer_fires() -> list[str]:
    """d. A passing run then a bug-report edit fires the no-failure-observed steer.

    A task that reports a failure, a run_command that PASSES (exit 0), then a file
    edit: the reported failure was never observed, so the first mutation must
    append the no-failure-observed steer exactly once, riding the user role, with
    the telemetry line — and must NOT fire the reproduce-before-edit steer (a run
    was recorded, so the zero-runs steer's precondition is not met).
    """
    import agent

    script = [
        _tool_call_response("run_command", {"cmd": "exit 0"}),
        _tool_call_response("create_file", {"path": "buggy.py", "content": "x = 1\n"}),
        _final_answer_response("Applied a defensive fix."),
    ]
    session, stderr = _drive_turn(
        "list --category food crashes with a KeyError; reproduce and fix",
        script,
        "repro-steer-nofail-",
    )

    failures: list[str] = []
    nf_rows = _steer_rows(session, agent._NO_FAILURE_OBSERVED_STEER)
    if len(nf_rows) != 1:
        failures.append(
            f"expected exactly 1 no-failure-observed steer row after a passing "
            f"run then a bug-report edit, found {len(nf_rows)}"
        )
    if stderr.count("no-failure-steer: fired") != 1:
        failures.append(
            f"expected the 'no-failure-steer: fired' telemetry line exactly "
            f"once, stderr had {stderr.count('no-failure-steer: fired')}"
        )
    repro_rows = _steer_rows(session, agent._REPRO_BEFORE_EDIT_STEER)
    if repro_rows:
        failures.append(
            f"reproduce-before-edit steer fired {len(repro_rows)} time(s) despite "
            f"a recorded run this turn (should defer to no-failure-observed)"
        )
    return failures


def check_no_failure_suppressed_on_failing_run() -> list[str]:
    """e. A recorded FAILING run suppresses the no-failure-observed steer.

    A bug-report task, a run_command that FAILS (nonzero exit), then an edit: a
    failure WAS observed this turn, so neither the no-failure-observed steer nor
    the reproduce-before-edit steer may fire.
    """
    import agent

    script = [
        _tool_call_response("run_command", {"cmd": "exit 7"}),
        _tool_call_response("create_file", {"path": "buggy.py", "content": "x = 1\n"}),
        _final_answer_response("Reproduced then edited."),
    ]
    session, stderr = _drive_turn(
        "the list command crashes with a KeyError; fix it",
        script,
        "repro-steer-nofail-failrun-",
    )

    failures: list[str] = []
    nf_rows = _steer_rows(session, agent._NO_FAILURE_OBSERVED_STEER)
    if nf_rows:
        failures.append(
            f"no-failure-observed steer fired {len(nf_rows)} time(s) despite a "
            f"FAILING run this turn (a failure was observed — must suppress)"
        )
    if "no-failure-steer: fired" in stderr:
        failures.append(
            "'no-failure-steer: fired' telemetry appeared despite a failing run"
        )
    return failures


def check_no_failure_suppressed_without_lexicon() -> list[str]:
    """f. A non-bug-report task never fires the no-failure-observed steer.

    A pure feature task (no bug-report lexicon), a passing run, then an edit: the
    no-failure-observed steer must NOT fire — it is scoped to reported failures.
    """
    import agent

    script = [
        _tool_call_response("run_command", {"cmd": "exit 0"}),
        _tool_call_response("create_file", {"path": "feature.py", "content": "x = 1\n"}),
        _final_answer_response("Added the feature."),
    ]
    session, stderr = _drive_turn(
        "add a --json flag to the list subcommand",
        script,
        "repro-steer-nofail-nolex-",
    )

    failures: list[str] = []
    nf_rows = _steer_rows(session, agent._NO_FAILURE_OBSERVED_STEER)
    if nf_rows:
        failures.append(
            f"no-failure-observed steer fired {len(nf_rows)} time(s) on a task "
            f"with no bug-report lexicon (must stay out of feature work)"
        )
    if "no-failure-steer: fired" in stderr:
        failures.append(
            "'no-failure-steer: fired' telemetry appeared on a non-bug-report task"
        )
    return failures


def _check_noise_run_fires(noise_cmd: str, tmp_prefix: str) -> list[str]:
    """Shared driver for h1-h3: a noise run + a passing run + a bug-report edit.

    A run that failed only at the shell level (*noise_cmd*) followed by a PASSING
    run and then a file edit must still fire the no-failure-observed steer exactly
    once (with telemetry) — the noise failure never reached project code, so the
    reported failure remains unobserved. The noise run is still recorded with
    ``passed=False``, and it must be classified ``noise=True``.
    """
    import agent

    script = [
        _tool_call_response("run_command", {"cmd": noise_cmd}),
        _tool_call_response("run_command", {"cmd": "exit 0"}),
        _tool_call_response("create_file", {"path": "buggy.py", "content": "x = 1\n"}),
        _final_answer_response("Applied a defensive fix."),
    ]
    session, stderr = _drive_turn(
        "list --category food crashes with a KeyError; reproduce and fix",
        script,
        tmp_prefix,
    )

    failures: list[str] = []
    nf_rows = _steer_rows(session, agent._NO_FAILURE_OBSERVED_STEER)
    if len(nf_rows) != 1:
        failures.append(
            f"expected exactly 1 no-failure-observed steer row after a "
            f"{noise_cmd!r} noise run + a passing run + a bug-report edit, found "
            f"{len(nf_rows)}"
        )
    if stderr.count("no-failure-steer: fired") != 1:
        failures.append(
            f"expected the 'no-failure-steer: fired' telemetry line exactly once "
            f"for noise cmd {noise_cmd!r}, stderr had "
            f"{stderr.count('no-failure-steer: fired')}"
        )
    runs = session.turn_report["verification_runs"]
    noise_runs = [r for r in runs if r["detail"] == noise_cmd]
    if not noise_runs:
        failures.append(f"noise run {noise_cmd!r} produced no verification_runs entry")
    else:
        if noise_runs[0].get("passed") is not False:
            failures.append(
                f"noise run {noise_cmd!r} recorded passed="
                f"{noise_runs[0].get('passed')!r}, expected False"
            )
        if noise_runs[0].get("noise") is not True:
            failures.append(
                f"noise run {noise_cmd!r} recorded noise="
                f"{noise_runs[0].get('noise')!r}, expected True"
            )
    return failures


def check_noise_cd_fires() -> list[str]:
    """h1. A `cd` into a nonexistent directory is noise -- steer still fires."""
    return _check_noise_run_fires(
        "cd /definitely/not/a/real/dir && echo x", "repro-steer-noise-cd-"
    )


def check_noise_git_fires() -> list[str]:
    """h2. A `git` command in a tree with no `.git` is noise -- steer still fires."""
    return _check_noise_run_fires("git log", "repro-steer-noise-git-")


def check_noise_exit127_fires() -> list[str]:
    """h3. A command-not-found (exit 127) is noise -- steer still fires."""
    return _check_noise_run_fires(
        "definitely_not_a_real_command_xyz", "repro-steer-noise-127-"
    )


def check_genuine_failure_suppresses() -> list[str]:
    """h4. A genuine nonzero-exit run (not noise) still suppresses the steer.

    A project run that exits nonzero without matching any environment-noise shape
    (not 127, no `cd` prefix, not `git`) is a genuinely observed failure, so even
    with a passing run also present this turn the no-failure-observed steer must
    NOT fire.
    """
    import agent

    script = [
        _tool_call_response("run_command", {"cmd": "exit 7"}),
        _tool_call_response("run_command", {"cmd": "exit 0"}),
        _tool_call_response("create_file", {"path": "buggy.py", "content": "x = 1\n"}),
        _final_answer_response("Reproduced then edited."),
    ]
    session, stderr = _drive_turn(
        "the list command crashes with a KeyError; fix it",
        script,
        "repro-steer-genuine-fail-",
    )

    failures: list[str] = []
    nf_rows = _steer_rows(session, agent._NO_FAILURE_OBSERVED_STEER)
    if nf_rows:
        failures.append(
            f"no-failure-observed steer fired {len(nf_rows)} time(s) despite a "
            f"GENUINE failing run this turn (a real failure was observed — must "
            f"suppress even with a passing run present)"
        )
    if "no-failure-steer: fired" in stderr:
        failures.append(
            "'no-failure-steer: fired' telemetry appeared despite a genuine "
            "failing run"
        )
    return failures


def check_gate_honesty() -> list[str]:
    """g. The verify gate/flag key on a PASSING run, not a merely-completed one.

    A mutation followed by ONLY a nonzero-exit run_command leaves the turn's
    verified flag False (the gate stayed open — a failing run is not
    verification); the same mutation followed by a passing run flips it True.
    ``turn_report['verified']`` is ``_turn_verified`` applied to the end-of-turn
    state, so it is the observable proof of the gate's honesty.
    """
    failures: list[str] = []

    fail_script = [
        _tool_call_response("create_file", {"path": "buggy.py", "content": "x = 1\n"}),
        _tool_call_response("run_command", {"cmd": "exit 7"}),
        _final_answer_response("Edited; the check failed."),
    ]
    fail_session, _ = _drive_turn(
        "fix the crash in the list command", fail_script, "repro-steer-gate-fail-"
    )
    if fail_session.turn_report["verified"] is not False:
        failures.append(
            f"mutation + only a FAILING run: verified="
            f"{fail_session.turn_report['verified']!r}, expected False "
            f"(a nonzero-exit run must not clear the verify gate)"
        )

    pass_script = [
        _tool_call_response("create_file", {"path": "buggy.py", "content": "x = 1\n"}),
        _tool_call_response("run_command", {"cmd": "exit 0"}),
        _final_answer_response("Edited and verified."),
    ]
    pass_session, _ = _drive_turn(
        "fix the crash in the list command", pass_script, "repro-steer-gate-pass-"
    )
    if pass_session.turn_report["verified"] is not True:
        failures.append(
            f"mutation + a PASSING run: verified="
            f"{pass_session.turn_report['verified']!r}, expected True"
        )

    # The failing run is still RECORDED (status/detail unchanged) but marked
    # passed=False, so the give-up/report facts stay truthful.
    runs = fail_session.turn_report["verification_runs"]
    fail_runs = [r for r in runs if r["tool"] == "run_command"]
    if not fail_runs:
        failures.append("failing run_command produced no verification_runs entry")
    elif fail_runs[0].get("passed") is not False:
        failures.append(
            f"failing run_command recorded passed={fail_runs[0].get('passed')!r}, "
            f"expected False"
        )
    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("fires-and-one-shot", check_fires_and_one_shot),
        ("suppressed-after-run", check_suppressed_after_run),
        ("no-failure-fires", check_no_failure_steer_fires),
        ("no-failure-suppressed-failing-run", check_no_failure_suppressed_on_failing_run),
        ("no-failure-suppressed-no-lexicon", check_no_failure_suppressed_without_lexicon),
        ("noise-cd-fires", check_noise_cd_fires),
        ("noise-git-fires", check_noise_git_fires),
        ("noise-exit127-fires", check_noise_exit127_fires),
        ("genuine-failure-suppresses", check_genuine_failure_suppresses),
        ("gate-honesty", check_gate_honesty),
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
        "(failing) run_command suppresses it; a second edit does not double-steer; "
        "a passing run then a bug-report edit steers no-failure-observed once "
        "(suppressed by a genuine failing run or a non-bug task, mutually "
        "exclusive with reproduce-before-edit); shell-level environment noise "
        "(cd into a missing dir, git with no .git, exit 127) does not count as an "
        "observed project failure so the steer still fires alongside a passing "
        "run, while a genuine nonzero-exit run still suppresses it; and the "
        "verify gate keys on a PASSING run"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
