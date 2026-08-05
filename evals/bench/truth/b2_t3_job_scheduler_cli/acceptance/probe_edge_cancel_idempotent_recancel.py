"""Edge: cancelling an already-cancelled job succeeds again with no error
(not a usage error, not a crash) and leaves state coherent -- a single
entry, still cancelled, with its other fields untouched by the re-cancel.

Requirement 10 edge case (idempotency).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "x | 1 | 1 | - | true\n")

    for _ in range(2):
        proc = _lib.run_cli(work, ["cancel", "jobs.txt", "x"])
        if proc.returncode != 0:
            _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
        if proc.stdout.strip() != "cancelled: x":
            _lib.fail(f"expected stdout 'cancelled: x', got {proc.stdout!r}")

    state = _lib.read_state(work)
    if list(state["jobs"].keys()) != ["x"]:
        _lib.fail(f"expected exactly one job entry ('x'), got {list(state['jobs'].keys())}")
    info = state["jobs"]["x"]
    if info.get("cancelled") is not True:
        _lib.fail(f"expected cancelled == true after re-cancel, got {info}")
    if info["run_count"] != 0 or info["last_status"] is not None:
        _lib.fail(f"re-cancelling must not fabricate execution history, got {info}")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
