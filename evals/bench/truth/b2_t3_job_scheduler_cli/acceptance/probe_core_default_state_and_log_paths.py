"""Core: with no --state-file/--log-dir flags, runtime files land exactly at
the documented default paths.

Requirement 6/7 happy path (default locations).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    expected_state = os.path.join(work, ".jobsched", "state.json")
    expected_log = os.path.join(work, ".jobsched", "logs", "run.log")
    if not os.path.isfile(expected_state):
        _lib.fail(f"expected default state file at {expected_state}")
    if not os.path.isfile(expected_log):
        _lib.fail(f"expected default log file at {expected_log}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
