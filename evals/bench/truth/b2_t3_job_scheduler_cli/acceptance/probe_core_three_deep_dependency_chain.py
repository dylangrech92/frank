"""Core: a 3-deep dependency chain (c depends on b depends on a), where the
chain's tail has the HIGHEST priority.

This pins the priority/dependency interaction precisely: a single tick's due
jobs are walked once in priority order, not repeatedly to a fixed point. A
dependent whose dependency has not yet run *this tick* is skipped for this
tick, even if the dependent's priority would otherwise put it first — it is
not retried again until its own next due tick. With every job due every
tick (interval 1), the chain therefore cascades exactly one level per tick:
tick 0 only 'a' runs, tick 1 'a' and 'b' have run so 'b' now completes
alongside 'a', tick 2 all three have completed at least once.
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
        "a | 1 | 1 | -   | true\n"
        "b | 5 | 1 | a   | true\n"
        "c | 9 | 1 | b   | true\n",
    )
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "3"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log = _lib.read_log(work)
    by_tick = {}
    for line in log.splitlines():
        if not line:
            continue
        tick, job_id, status, exit_code, reason = line.split("\t")
        by_tick.setdefault(int(tick), {})[job_id] = status

    if by_tick[0] != {"a": "completed", "b": "skipped", "c": "skipped"}:
        _lib.fail(f"tick 0: expected only 'a' to complete, got {by_tick[0]}")
    if by_tick[1] != {"a": "completed", "b": "completed", "c": "skipped"}:
        _lib.fail(f"tick 1: expected 'a' and 'b' to complete, got {by_tick[1]}")
    if by_tick[2] != {"a": "completed", "b": "completed", "c": "completed"}:
        _lib.fail(f"tick 2: expected all three to complete, got {by_tick[2]}")

    state = _lib.read_state(work)
    for job_id in ("a", "b", "c"):
        if state["jobs"][job_id]["last_status"] != "completed":
            _lib.fail(f"job {job_id} not completed by end of run: {state['jobs'][job_id]}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
