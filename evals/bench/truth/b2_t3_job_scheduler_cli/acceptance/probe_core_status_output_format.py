"""Core: `status` output matches the documented format exactly.

Requirement 8 happy path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")
    run_proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if run_proc.returncode != 0:
        _lib.fail(f"setup run failed: {run_proc.stderr!r}")

    proc = _lib.run_cli(work, ["status"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    lines = proc.stdout.splitlines()
    if not lines or lines[0] != "last_completed_tick: 0":
        _lib.fail(f"unexpected first line: {lines[:1]!r}")
    if "a: status=completed run_count=1 last_exit_code=0" not in lines:
        _lib.fail(f"missing expected job status line, got: {lines!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
