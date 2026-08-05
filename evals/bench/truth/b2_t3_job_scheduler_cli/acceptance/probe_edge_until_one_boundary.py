"""Edge: `--until 1` is the smallest legal value and simulates exactly one
tick (tick 0 only).

Requirement 3 (ticks are the integer range 0..N-1 inclusive).
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
    if "tick 0:" not in proc.stdout:
        _lib.fail(f"expected a tick 0 line, got {proc.stdout!r}")
    if "tick 1:" in proc.stdout:
        _lib.fail(f"--until 1 must simulate only tick 0, got {proc.stdout!r}")
    if "done: 1 ticks simulated" not in proc.stdout:
        _lib.fail(f"expected 'done: 1 ticks simulated', got {proc.stdout!r}")

    state = _lib.read_state(work)
    if state["last_completed_tick"] != 0:
        _lib.fail(f"expected last_completed_tick=0, got {state['last_completed_tick']}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
