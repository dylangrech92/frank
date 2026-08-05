"""Core: after a real run, the state file is valid JSON with the documented
top-level schema.

Requirement 6 happy path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "2"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    state = _lib.read_state(work)
    for key in ("version", "schedule_hash", "last_completed_tick", "in_progress_tick", "jobs"):
        if key not in state:
            _lib.fail(f"state.json missing key '{key}': {state}")
    if state["last_completed_tick"] != 1:
        _lib.fail(f"expected last_completed_tick 1, got {state['last_completed_tick']}")
    if state["in_progress_tick"] is not None:
        _lib.fail(f"expected in_progress_tick null after clean completion, got {state['in_progress_tick']}")
    if "a" not in state["jobs"]:
        _lib.fail(f"expected job 'a' recorded in state, got {state['jobs']}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
