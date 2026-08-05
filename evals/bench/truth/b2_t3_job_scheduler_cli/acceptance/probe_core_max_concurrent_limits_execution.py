"""Core: --max-concurrent caps how many due jobs actually execute in a
single tick; the rest are skipped with reason budget-exceeded and are
retried at their own next due tick, exactly like waiting-on-dependency.

Requirement 3/10 happy path (--max-concurrent).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    # Three independent (no-dependency) jobs, all due every tick, distinct
    # priorities so due-order is fixed: a, b, c.
    _lib.write_file(
        work,
        "jobs.txt",
        "a | 3 | 1 | - | true\n"
        "b | 2 | 1 | - | true\n"
        "c | 1 | 1 | - | true\n",
    )

    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "2", "--max-concurrent", "2"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("tick ")]
    if lines[:2] != ["tick 0: 2 completed, 0 failed, 1 skipped", "tick 1: 2 completed, 0 failed, 1 skipped"]:
        _lib.fail(f"unexpected tick summary lines: {lines}")

    log = _lib.read_log(work)
    by_tick = {}
    for line in log.splitlines():
        if not line:
            continue
        tick, job_id, status, exit_code, reason = line.split("\t")
        by_tick.setdefault(int(tick), {})[job_id] = (status, reason)

    # a and b have the two highest priorities -- they run; c is bumped by
    # the budget every tick (it never becomes "eligible" via a dependency,
    # it is purely excluded by exhausted budget).
    for t in (0, 1):
        if by_tick[t]["a"][0] != "completed" or by_tick[t]["b"][0] != "completed":
            _lib.fail(f"expected a and b completed at tick {t}, got {by_tick[t]}")
        if by_tick[t]["c"] != ("skipped", "budget-exceeded"):
            _lib.fail(f"expected c skipped/budget-exceeded at tick {t}, got {by_tick[t]['c']}")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
