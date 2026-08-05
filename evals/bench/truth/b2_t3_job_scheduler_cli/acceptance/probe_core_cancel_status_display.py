"""Core: cancelling a job that has NEVER executed creates a fresh state
entry for it, displayed by `status` as `status=cancelled` with the literal
`none` exit code -- the "never ran" counterpart to
probe_core_cancel_prevents_future_execution's "ran once, then cancelled"
case.

Requirement 10 happy path (status display, "none" convention).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(
        work,
        "jobs.txt",
        "x | 1 | 1 | - | true\n"
        "y | 1 | 1 | - | true\n",
    )

    # y is cancelled before jobsched has ever run at all -- no state file
    # exists yet, so `cancel` must create one from scratch.
    proc = _lib.run_cli(work, ["cancel", "jobs.txt", "y"])
    if proc.returncode != 0:
        _lib.fail(f"cancel: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
    if proc.stdout.strip() != "cancelled: y":
        _lib.fail(f"cancel: expected stdout 'cancelled: y', got {proc.stdout!r}")

    state = _lib.read_state(work)
    info = state["jobs"].get("y")
    if info is None:
        _lib.fail("expected a fresh 'y' entry in state.jobs after cancelling a never-run job")
    if info.get("cancelled") is not True:
        _lib.fail(f"expected cancelled == true, got {info}")
    if info["run_count"] != 0 or info["last_run_tick"] is not None or info["last_status"] is not None:
        _lib.fail(f"expected a never-executed entry (run_count=0, nulls), got {info}")

    proc = _lib.run_cli(work, ["status"])
    if proc.returncode != 0:
        _lib.fail(f"status: expected exit 0, got {proc.returncode}")
    expected_line = "y: status=cancelled run_count=0 last_exit_code=none"
    if expected_line not in proc.stdout:
        _lib.fail(f"expected status line {expected_line!r} in {proc.stdout!r}")
    # x was never executed and never cancelled -- must not appear at all.
    if "x:" in proc.stdout:
        _lib.fail(f"job 'x' was neither executed nor cancelled and must not be listed: {proc.stdout!r}")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
