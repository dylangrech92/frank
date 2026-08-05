"""Edge: --max-concurrent, when given, must be validated the same way
--until is -- a non-positive-integer value is a usage error (exit 6), not a
crash and not silently treated as unlimited.

Requirement 3/9 edge case (usage validation).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "x | 1 | 1 | - | true\n")

    for bad_value in ("0", "-1", "abc"):
        proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1", "--max-concurrent", bad_value])
        if proc.returncode != 6:
            _lib.fail(
                f"--max-concurrent {bad_value!r}: expected exit 6, got {proc.returncode}; "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        expected = "error: --max-concurrent requires a positive integer"
        if proc.stderr.strip() != expected:
            _lib.fail(f"--max-concurrent {bad_value!r}: expected {expected!r}, got {proc.stderr!r}")

    if os.path.exists(os.path.join(work, ".jobsched")):
        _lib.fail("a usage error must not create .jobsched/")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
