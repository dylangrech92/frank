"""Edge: a state file that is not valid JSON is refused (exit 4) without
--force-recover, and accepted-and-reset (fresh start, exit 0) with it.

Requirement 3 (state loading) / Requirement 8.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")
    os.makedirs(os.path.join(work, ".jobsched"), exist_ok=True)
    _lib.write_file(work, ".jobsched/state.json", "{not valid json,,,")

    refused = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if refused.returncode != 4:
        _lib.fail(f"expected exit 4 without --force-recover, got {refused.returncode}; stderr={refused.stderr!r}")
    expected_prefix = "error: state file corrupt:"
    if not refused.stderr.strip().startswith(expected_prefix):
        _lib.fail(f"expected stderr to start with {expected_prefix!r}, got {refused.stderr!r}")
    if "--force-recover" not in refused.stderr:
        _lib.fail(f"expected recovery hint in stderr, got {refused.stderr!r}")

    recovered = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1", "--force-recover"])
    if recovered.returncode != 0:
        _lib.fail(f"expected exit 0 with --force-recover, got {recovered.returncode}; stderr={recovered.stderr!r}")

    state = _lib.read_state(work)
    if state["last_completed_tick"] != 0:
        _lib.fail(f"expected fresh state to reach last_completed_tick=0, got {state}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
