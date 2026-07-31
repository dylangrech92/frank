from __future__ import annotations

import re
import shlex

from llm import ToolCall
from tools.registry import is_loaded
from tools.result import ToolResult


# =============================================================================
# Repeat cap
# =============================================================================

# Hard cap on identical tool calls within a single turn. A model stuck
# re-issuing the exact same successful no-op (e.g. replace_one with search ==
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
_VERIFICATION_TOOLS = frozenset({"run_command", "run_tests", "verify_scratch"})
_PROFILING_TOOLS = frozenset({"profile_command", "profile_hotspots", "profile_memory", "trace_execution"})
_REPEAT_CALL_CAP = 3
_REPEAT_CAP_EXEMPT = _VERIFICATION_TOOLS | _PROFILING_TOOLS


def _available_verification_tools() -> list[str]:
    """Sorted verification tools the active mode actually carries.

    Tools are now a static per-mode set fixed at launch (``registry.activate_mode``)
    — there is nothing left to dynamically activate. The verify steers still need
    to name real, callable tools, so this reports the intersection of
    ``_VERIFICATION_TOOLS`` with the active mode via ``registry.is_loaded`` (which
    now means "in the active mode"), letting each steer name exactly what the
    model can call: all three in test mode, ``run_command`` + ``verify_scratch``
    in code mode (never ``run_tests`` — code mode's own instructions forbid
    running the suite), and ``run_command`` alone in performance_debug. Empty in
    research mode, which has no verification tools and no way to mutate.
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