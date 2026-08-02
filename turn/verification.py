from __future__ import annotations

import re
import shlex
import sys

import ui
from llm import ToolCall
from tools.registry import is_loaded
from tools.result import ToolResult
from turn.state import _TurnState
from turn.steering import _is_bug_report


# =============================================================================
# Repeat cap
# =============================================================================

# Hard cap on identical tool calls within a single turn. A model stuck
# re-issuing the exact same successful no-op (e.g. edit_file with search ==
# replace) would otherwise spin forever: the success-path steer below nudges
# first, and once this many identical (name, arguments) pairs have been
# dispatched, the call is refused at dispatch time (see handle_user_message).
# Verification tools (run_command / run_tests / verify_scratch) are exempt —
# repeating an identical build/test inside an edit→verify→edit cycle is legitimate.
#
# Single source of truth for the verification-tool names. The repeat-cap
# exemption, the verification_runs recording branch, and
# _available_verification_tools all read this one set so the list cannot
# drift across the file.
#
# Profiling tools are repeat-cap exempt because a measure -> edit -> re-measure
# loop legitimately re-issues identical calls, but they are NOT verification tools.
#
# verify mode's observation tools are exempt for the same structural reason: its
# instructions MANDATE re-snapshotting after every DOM mutation, and scroll /
# wait_for are legitimately called repeatedly with identical arguments, so a
# normal multi-step verification would false-positive the cap and break its own
# evidence chain. `report` is exempt so a model retrying it after an
# evidence-gate rejection can keep retrying — the gate is the whole point of the
# mode, and capping the retry would strand the run with no verdict.
_VERIFICATION_TOOLS = frozenset({"run_command", "run_tests", "verify_scratch"})
_PROFILING_TOOLS = frozenset({"profile_command", "profile_hotspots", "profile_memory", "trace_execution"})
_BROWSER_OBSERVE_TOOLS = frozenset({
    "report",
    "snapshot",
    "console_logs",
    "network_requests",
    "screenshot",
    "scroll",
    "wait_for",
})
_REPEAT_CALL_CAP = 3
_REPEAT_CAP_EXEMPT = _VERIFICATION_TOOLS | _PROFILING_TOOLS | _BROWSER_OBSERVE_TOOLS


def _available_verification_tools() -> list[str]:
    """Sorted verification tools the active mode actually carries.

    Tools are now a static per-mode set fixed at launch (``registry.activate_mode``)
    — there is nothing left to dynamically activate. The verify steers still need
    to name real, callable tools, so this reports the intersection of
    ``_VERIFICATION_TOOLS`` with the active mode via ``registry.is_loaded`` (which
    now means "in the active mode"), letting each steer name exactly what the
    model can call: all three in qa mode, ``run_command`` + ``verify_scratch``
    in code mode (never ``run_tests`` — code mode's own instructions forbid
    running the suite), and ``run_command`` alone in performance_debug. Empty in
    research and verify modes, which have no verification tools and no way to
    mutate.
    """
    return sorted(name for name in _VERIFICATION_TOOLS if is_loaded(name))


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


def _run_failure_is_environment_noise(
    call: ToolCall, result: ToolResult, project_root: str
) -> bool:
    """True only when a FAILED run failed at the shell level, never reaching code.

    A verification run that failed normally counts as "the reported failure was
    observed this turn" and suppresses the no-failure-observed steer. But some
    shell-level failures never exercised the project at all — a hallucinated
    directory, a missing binary, a git probe in a tree that is not a repository —
    so treating them as an observed failure lets the model skip reproducing the
    real one. This recognises exactly three such shapes, each decided from a
    *fact about the world* (an exit code, a path on disk, a directory's
    contents), never from matching output text:

    * ``run_command`` exited 127 — the OS reported command-not-found, so nothing
      ran.
    * ``run_command`` begins with a ``cd <target>`` segment whose target does not
      exist on disk (resolved against *project_root* when relative). The ``cd``
      itself failed, so nothing after it ran.
    * ``run_command`` whose first token is ``git`` while *project_root* holds no
      ``.git`` — any git failure there is environment probing, not project
      behavior.

    ``run_tests`` and ``verify_scratch`` failures are never noise: they only run
    project code. Anything not matching one of the three shapes is treated as a
    genuinely observed failure, and any command-parse error returns False (not
    noise) — the conservative default keeps a real failure from being waved off.
    """
    from pathlib import Path

    if call.name != "run_command":
        return False
    # Shape (a): the OS could not find the command — exit 127, nothing ran.
    if result.meta.get("exit_code") == 127:
        return True
    cmd = str(call.arguments.get("cmd", ""))
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False
    if not tokens:
        return False
    # Shape (b): a leading `cd <target>` whose target is not on disk. The target
    # is the first shell word after `cd` in the first segment (the command up to
    # the first &&, ||, ;, or | — anything after a failed cd never ran).
    if tokens[0] == "cd":
        segment = re.split(r"&&|\|\||;|\|", cmd, maxsplit=1)[0]
        try:
            seg_tokens = shlex.split(segment)
        except ValueError:
            return False
        if len(seg_tokens) > 1:
            target = Path(seg_tokens[1])
            if not target.is_absolute():
                target = Path(project_root) / seg_tokens[1]
            if not target.exists():
                return True
    # Shape (c): a git command in a tree that is not a git repository.
    if tokens[0] == "git" and not (Path(project_root) / ".git").exists():
        return True
    return False


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