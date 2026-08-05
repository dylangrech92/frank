"""Edge: a job line with too few '|'-separated fields is rejected by rule 1
with the exact 'got N' count, before any other rule is even considered.

Requirement 1, rule 1.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1\n")
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 2:
        _lib.fail(f"expected exit 2, got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    expected = "error: line 1: expected 5 fields separated by '|', got 3"
    if proc.stderr.strip() != expected:
        _lib.fail(f"expected {expected!r}, got {proc.stderr!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
