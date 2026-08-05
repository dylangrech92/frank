"""Core: `validate` on a well-formed schedule prints exactly
`OK: <n> jobs, <m> edges` to stdout and nothing else, exit 0.

Requirement 1 happy path (validate subcommand contract).
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
        "a | 1 | 1 | -   | true\n"
        "b | 2 | 1 | a   | true\n"
        "c | 3 | 1 | a   | true\n",
    )
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    stdout = proc.stdout.strip()
    if stdout != "OK: 3 jobs, 2 edges":
        _lib.fail(f"expected exact 'OK: 3 jobs, 2 edges', got {stdout!r}")
    if proc.stderr.strip() != "":
        _lib.fail(f"expected empty stderr on successful validate, got {proc.stderr!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
