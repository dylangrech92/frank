"""Edge: --dry-run honors --max-concurrent identically to a real run (same
budget, same precedence), while still making zero filesystem changes.

Requirement 3 edge case (dry-run/budget interaction).
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
        "a | 3 | 1 | - | true\n"
        "b | 2 | 1 | - | true\n"
        "c | 1 | 1 | - | true\n",
    )

    before = _lib.snapshot_paths(work)
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1", "--dry-run", "--max-concurrent", "2"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    expected_line = "tick 0 (dry-run): 2 would-run, 1 would-skip"
    if expected_line not in proc.stdout:
        _lib.fail(f"expected {expected_line!r} in stdout, got {proc.stdout!r}")

    after = _lib.snapshot_paths(work)
    if before != after:
        _lib.fail(f"--dry-run with --max-concurrent must make zero filesystem changes: before={before} after={after}")
    if os.path.exists(os.path.join(work, ".jobsched")):
        _lib.fail("--dry-run must never create .jobsched/")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
