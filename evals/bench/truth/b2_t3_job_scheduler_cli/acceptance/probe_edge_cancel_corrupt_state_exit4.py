"""Edge: `cancel` follows the exact same corrupt-state recovery rules as
`run` -- exit 4 with the standard message when the state file is corrupt
and --force-recover is not given, and a clean force-recovered cancellation
when it is.

Requirement 10 / Section 8 edge case (cancel shares run's state-recovery
contract).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "x | 1 | 1 | - | true\n")

    os.makedirs(os.path.join(work, ".jobsched"), exist_ok=True)
    state_path = os.path.join(work, ".jobsched", "state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        f.write("{ not valid json")

    proc = _lib.run_cli(work, ["cancel", "jobs.txt", "x"])
    if proc.returncode != 4:
        _lib.fail(f"expected exit 4, got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    expected_prefix = "error: state file corrupt:"
    expected_suffix = "(use --force-recover to reset)"
    stderr = proc.stderr.strip()
    if not (stderr.startswith(expected_prefix) and stderr.endswith(expected_suffix)):
        _lib.fail(f"expected prefix {expected_prefix!r} and suffix {expected_suffix!r}, got {stderr!r}")
    with open(state_path, "r", encoding="utf-8") as f:
        if f.read() != "{ not valid json":
            _lib.fail("cancel must not touch the corrupt state file when --force-recover is absent")

    proc = _lib.run_cli(work, ["cancel", "jobs.txt", "x", "--force-recover"])
    if proc.returncode != 0:
        _lib.fail(f"force-recover: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
    if proc.stdout.strip() != "cancelled: x":
        _lib.fail(f"force-recover: expected stdout 'cancelled: x', got {proc.stdout!r}")

    state = _lib.read_state(work)
    if state["last_completed_tick"] is not None or state["in_progress_tick"] is not None:
        _lib.fail(f"expected a fresh state after force-recover, got {state}")
    info = state["jobs"].get("x")
    if not info or info.get("cancelled") is not True:
        _lib.fail(f"expected 'x' cancelled in the recovered state, got {state['jobs']}")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
