"""Core: a diamond dependency graph (no cycle) validates cleanly.

Requirement 2 happy path.
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
        "top    | 1 | 1 | -          | true\n"
        "left   | 1 | 1 | top        | true\n"
        "right  | 1 | 1 | top        | true\n"
        "bottom | 1 | 1 | left,right | true\n",
    )
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
    if proc.stdout.strip() != "OK: 4 jobs, 4 edges":
        _lib.fail(f"unexpected stdout: {proc.stdout!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
