"""Dispatch-level feature test for the ``verify_scratch`` tool (TKT-1466, no LLM).

Drives the real production hot path — ``tools.registry.dispatch`` — exactly as
``handle_user_message`` does per tool call, so this exercises arg validation,
the mode gate, and ``VerifyScratch.run`` together, not the tool class in
isolation. Zero mocks: every snippet below is written to a real OS temp file
and executed by a real subprocess (real python/sh interpreter), against the
real project root as cwd, exactly as a live model turn would.

Asserts the documented contract from ``tools/verify_scratch.py``:

    1. exit 0                          -> ToolResult status='success'.
    2. non-zero exit (failed assert)   -> status='error', code='verification-failed',
                                           with stdout/stderr in the body (the model
                                           must be able to see WHY it failed).
    3. interpreter outside the allow-list
                                        -> status='error', code='invalid-interpreter',
                                           and NO temp file / side effect is ever
                                           created (checked before the enum gate is
                                           even guaranteed to have been evaluated —
                                           i.e. a real observable absence, not a
                                           trust in the source).
    4. cwd is the real project root, not the OS temp dir the snippet lives in
       (so imports/relative paths resolve against the real project).
    5. the temp file created for a run is deleted afterward regardless of
       whether the snippet passed or failed (no leak into the OS temp dir).

Exits 0 on success, 1 on any assertion failure or unexpected exception. Runs
with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file), and chdir's to the repo root itself before dispatching so the cwd
invariant is reproduced explicitly rather than depending on whichever
directory happened to invoke this script (mirrors how main.py's real process
is always launched with cwd already at the project root).
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dispatch(snippet: str, interpreter: str = "python", timeout: int = 60):
    from tools.registry import dispatch

    return dispatch(
        "verify_scratch",
        {"snippet": snippet, "interpreter": interpreter, "timeout": timeout},
    )


def check_success_snippet() -> list[str]:
    """A passing snippet (exit 0) must report status='success'."""
    failures: list[str] = []
    result = _dispatch("assert 1 == 1\n")
    if result.status != "success":
        failures.append(
            f"passing snippet: expected status='success', got {result.status!r} "
            f"code={result.code!r} body={result.body!r}"
        )
    return failures


def check_failure_snippet() -> list[str]:
    """A failing assertion must report status='error', code='verification-failed',
    and the body must still carry the traceback/stderr so the model can see why."""
    failures: list[str] = []
    result = _dispatch("assert False, 'deliberate failure for TKT-1466'\n")
    if result.status != "error":
        failures.append(f"failing snippet: expected status='error', got {result.status!r}")
    if result.code != "verification-failed":
        failures.append(f"failing snippet: expected code='verification-failed', got {result.code!r}")
    body = result.body if isinstance(result.body, str) else str(result.body)
    if "AssertionError" not in body:
        failures.append(f"failing snippet: body has no AssertionError, model can't see why it failed: {body!r}")
    if "Traceback" not in body:
        failures.append(f"failing snippet: body has no traceback: {body!r}")
    return failures


def check_bad_interpreter() -> list[str]:
    """An interpreter outside the allow-list must be rejected with
    code='invalid-interpreter' and must never create the temp file or run
    anything — proven by a sentinel path embedded in the (rejected) interpreter
    string never coming into existence."""
    failures: list[str] = []
    sentinel = Path(tempfile.gettempdir()) / "verify_scratch_contract_sentinel_TKT1466.txt"
    if sentinel.exists():
        sentinel.unlink()

    result = _dispatch("print('should never run')", interpreter=f"sh; touch {sentinel}")

    if result.status != "error":
        failures.append(f"bad interpreter: expected status='error', got {result.status!r}")
    if result.code != "invalid-interpreter":
        failures.append(f"bad interpreter: expected code='invalid-interpreter', got {result.code!r}")
    if sentinel.exists():
        failures.append(
            "bad interpreter: sentinel file was created -- shell injection / "
            "side effect leaked through the rejected interpreter"
        )
        sentinel.unlink()
    return failures


def check_cwd_is_project_root() -> list[str]:
    """The snippet must execute with cwd == the real project root, not the OS
    temp dir the snippet file itself lives in."""
    failures: list[str] = []
    result = _dispatch("import os\nprint(os.getcwd())\n")
    if result.status != "success":
        failures.append(f"cwd snippet: expected status='success', got {result.status!r} body={result.body!r}")
        return failures

    body = result.body if isinstance(result.body, str) else str(result.body)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip() and ln.strip() != "--- stdout ---"]
    reported_cwd = lines[0] if lines else ""
    expected = os.path.realpath(str(REPO_ROOT))
    actual = os.path.realpath(reported_cwd) if reported_cwd else ""
    if actual != expected:
        failures.append(f"cwd snippet: expected cwd {expected!r}, snippet reported {actual!r} (body={body!r})")
    return failures


def check_no_temp_file_leak() -> list[str]:
    """The temp file created for a run must be gone afterward, whether the
    snippet passed or failed — captured via a real before/after directory
    listing filtered to the tool's own prefix, not a trust in the source."""
    failures: list[str] = []
    tmp_dir = Path(tempfile.gettempdir())

    def _snapshot() -> set[str]:
        return set(glob.glob(str(tmp_dir / "verify_scratch_*")))

    before = _snapshot()
    _dispatch("assert 1 == 1\n")
    _dispatch("assert False\n")
    after = _snapshot()

    leaked = after - before
    if leaked:
        failures.append(f"temp-file leak: verify_scratch_* files remained after runs: {sorted(leaked)}")
    return failures


CHECKS: list[tuple[str, Callable[[], list[str]]]] = [
    ("passing snippet -> success", check_success_snippet),
    ("failing snippet -> verification-failed with traceback in body", check_failure_snippet),
    ("disallowed interpreter -> invalid-interpreter, zero side effects", check_bad_interpreter),
    ("snippet runs with cwd == real project root", check_cwd_is_project_root),
    ("no leftover temp file after a run (pass or fail)", check_no_temp_file_leak),
]


def main() -> int:
    # main.py's real process is always launched with cwd already at the
    # project root (see evals/run.py: cwd=project_dir for live scenarios,
    # cwd=REPO_ROOT for inline ones) — verify_scratch.run() derives its
    # "project root" from Path.cwd() at call time. Reproduce that invariant
    # explicitly here instead of depending on whatever directory happened to
    # invoke this script directly.
    os.chdir(REPO_ROOT)

    from tools import registry

    # verify_scratch is declared by 'qa' mode (modes.py) — dispatch()
    # refuses any tool outside the active mode's fixed set.
    registry.activate_mode("qa")

    all_failures: list[str] = []
    for description, check in CHECKS:
        failures = check()
        if failures:
            print(f"FAIL: {description}", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
            all_failures.extend(failures)
        else:
            print(f"PASS: {description}")

    if all_failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
