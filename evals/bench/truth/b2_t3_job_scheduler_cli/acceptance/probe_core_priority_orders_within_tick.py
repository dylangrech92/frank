"""Core: within a tick, higher-priority due jobs run before lower-priority
ones — visible in the run log's per-tick line order.

Requirement 4 happy path.
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
        "low  | 1 | 1 | - | true\n"
        "high | 9 | 1 | - | true\n",
    )
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log = _lib.read_log(work)
    lines = [l for l in log.splitlines() if l]
    job_order = [l.split("\t")[1] for l in lines]
    if job_order != ["high", "low"]:
        _lib.fail(f"expected ['high', 'low'] execution order, got {job_order}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
