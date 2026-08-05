"""Core: cancelling a job that has already executed once stops it from
executing again on subsequent ticks, without disturbing its historical
run_count/last_exit_code -- only its displayed status changes.

Requirement 10 happy path (direct effect of `cancel` on `run`).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "x | 1 | 1 | - | true\n")

    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if proc.returncode != 0:
        _lib.fail(f"initial run: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    proc = _lib.run_cli(work, ["cancel", "jobs.txt", "x"])
    if proc.returncode != 0:
        _lib.fail(f"cancel: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
    if proc.stdout.strip() != "cancelled: x":
        _lib.fail(f"cancel: expected stdout 'cancelled: x', got {proc.stdout!r}")

    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "3"])
    if proc.returncode != 0:
        _lib.fail(f"extended run: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log = _lib.read_log(work)
    by_tick = {}
    for line in log.splitlines():
        if not line:
            continue
        tick, job_id, status, exit_code, reason = line.split("\t")
        by_tick[int(tick)] = (job_id, status, exit_code, reason)

    if by_tick.get(0) != ("x", "completed", "0", "-"):
        _lib.fail(f"expected tick 0 = x completed, got {by_tick.get(0)}")
    for t in (1, 2):
        if by_tick.get(t) != ("x", "skipped", "-", "cancelled"):
            _lib.fail(f"expected tick {t} = x skipped/cancelled, got {by_tick.get(t)}")

    state = _lib.read_state(work)
    info = state["jobs"]["x"]
    if info.get("cancelled") is not True:
        _lib.fail(f"expected state jobs.x.cancelled == true, got {info}")
    if info["run_count"] != 1:
        _lib.fail(f"expected run_count to stay 1 (cancelled skips don't execute), got {info['run_count']}")
    if info["last_exit_code"] != 0:
        _lib.fail(f"expected last_exit_code to stay 0 from before cancellation, got {info['last_exit_code']}")

    proc = _lib.run_cli(work, ["status"])
    if proc.returncode != 0:
        _lib.fail(f"status: expected exit 0, got {proc.returncode}")
    expected_line = "x: status=cancelled run_count=1 last_exit_code=0"
    if expected_line not in proc.stdout:
        _lib.fail(f"expected status line {expected_line!r} in {proc.stdout!r}")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
