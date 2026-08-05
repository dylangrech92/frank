"""Core: a dependent job due in the SAME tick as its not-yet-run dependency
is skipped for that tick only; once the dependency has completed (even in
that same earlier walk position), the dependent is eligible at every tick
after, since dependency satisfaction persists once achieved.

producer (interval 3, due at ticks 0 and 3) and consumer (interval 1, due
every tick, priority-tied with producer) both come due at tick 0; tie-break
sorts 'consumer' before 'producer' alphabetically, so consumer is checked
before producer has run this tick and is skipped. producer then runs and
succeeds, so from tick 1 onward consumer's dependency is already satisfied.

Requirement 4 happy path (dependency ordering).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    # consumer is due every tick (interval 1) but depends on producer, which
    # is only due (and only completes) at tick 3.
    _lib.write_file(
        work,
        "jobs.txt",
        "producer | 1 | 3 | -        | true\n"
        "consumer | 1 | 1 | producer | true\n",
    )
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "5"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log = _lib.read_log(work)
    by_tick = {}
    for line in log.splitlines():
        if not line:
            continue
        tick, job_id, status, exit_code, reason = line.split("\t")
        by_tick.setdefault(int(tick), {})[job_id] = status

    if by_tick.get(0, {}).get("consumer") != "skipped":
        _lib.fail(f"expected consumer skipped at tick 0, got {by_tick.get(0)}")
    if by_tick.get(0, {}).get("producer") != "completed":
        _lib.fail(f"expected producer completed at tick 0, got {by_tick.get(0)}")
    for t in (1, 2, 3, 4):
        if by_tick.get(t, {}).get("consumer") != "completed":
            _lib.fail(f"expected consumer completed at tick {t}, got {by_tick.get(t)}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
