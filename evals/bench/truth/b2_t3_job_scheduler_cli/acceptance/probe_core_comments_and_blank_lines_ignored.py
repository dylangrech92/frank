"""Core: comment and blank lines never count as jobs or affect edge counts.

Requirement 1 happy path.
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
        "# leading comment\n"
        "\n"
        "   \n"
        "a | 1 | 1 | - | true\n"
        "# a mid-file comment\n"
        "b | 1 | 1 | a | true\n"
        "\n",
    )
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
    if proc.stdout.strip() != "OK: 2 jobs, 1 edges":
        _lib.fail(f"unexpected stdout: {proc.stdout!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
