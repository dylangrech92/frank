"""Core: a well-formed multi-job schedule validates cleanly.

Requirement 1 (schedule parsing) happy path.
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
        "backup  | 5 | 2 | -      | true\n"
        "report  | 3 | 2 | backup | true\n"
        "cleanup | 1 | 4 | report | true\n",
    )
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
    if proc.stdout.strip() != "OK: 3 jobs, 2 edges":
        _lib.fail(f"unexpected stdout: {proc.stdout!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
