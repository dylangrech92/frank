"""Edge: editing the schedule file between two `run` invocations changes its
SHA-256, so the second invocation must refuse (exit 5) without
--force-recover, and reset-and-proceed (exit 0, new hash recorded) with it.

Requirement 3 (state loading, schedule_hash comparison).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")
    first = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if first.returncode != 0:
        _lib.fail(f"first run: expected exit 0, got {first.returncode}; stderr={first.stderr!r}")

    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\nb | 1 | 1 | - | true\n")

    refused = _lib.run_cli(work, ["run", "jobs.txt", "--until", "2"])
    if refused.returncode != 5:
        _lib.fail(f"expected exit 5 without --force-recover, got {refused.returncode}; stderr={refused.stderr!r}")
    expected = "error: schedule has changed since last run (use --force-recover to reset state)"
    if refused.stderr.strip() != expected:
        _lib.fail(f"expected {expected!r}, got {refused.stderr!r}")

    recovered = _lib.run_cli(work, ["run", "jobs.txt", "--until", "2", "--force-recover"])
    if recovered.returncode != 0:
        _lib.fail(f"expected exit 0 with --force-recover, got {recovered.returncode}; stderr={recovered.stderr!r}")

    state = _lib.read_state(work)
    if "b" not in state["jobs"]:
        _lib.fail(f"expected fresh state (post schedule change) to include job 'b', got {state}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
