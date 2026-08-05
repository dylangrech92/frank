"""Core: re-running with the same --until after completion is a no-op.

Requirement 6 happy path (idempotent re-run).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")
    first = _lib.run_cli(work, ["run", "jobs.txt", "--until", "3"])
    if first.returncode != 0:
        _lib.fail(f"first run: expected exit 0, got {first.returncode}; stderr={first.stderr!r}")

    second = _lib.run_cli(work, ["run", "jobs.txt", "--until", "3"])
    if second.returncode != 0:
        _lib.fail(f"second run: expected exit 0, got {second.returncode}; stderr={second.stderr!r}")
    if "already complete: last_completed_tick=2" not in second.stdout:
        _lib.fail(f"expected 'already complete' message, got: {second.stdout!r}")

    state = _lib.read_state(work)
    if state["jobs"]["a"]["run_count"] != 3:
        _lib.fail(f"re-run must not re-execute jobs; run_count={state['jobs']['a']['run_count']}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
