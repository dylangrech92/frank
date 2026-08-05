"""Core: a job only executes on ticks where t % interval == 0.

Requirement 3 happy path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "j | 1 | 3 | - | true\n")
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "10"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    state = _lib.read_state(work)
    info = state["jobs"].get("j")
    if info is None:
        _lib.fail("job 'j' never recorded in state")
    # due at ticks 0,3,6,9 -> 4 runs
    if info["run_count"] != 4:
        _lib.fail(f"expected run_count 4 (ticks 0,3,6,9), got {info['run_count']}")
    if info["last_run_tick"] != 9:
        _lib.fail(f"expected last_run_tick 9, got {info['last_run_tick']}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
