"""Core: `history` prints newest-first, in the documented format, bounded by
--limit.

Requirement 8 happy path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")
    run_proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "3"])
    if run_proc.returncode != 0:
        _lib.fail(f"setup run failed: {run_proc.stderr!r}")

    proc = _lib.run_cli(work, ["history", "--limit", "2"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    lines = [l for l in proc.stdout.splitlines() if l]
    if len(lines) != 2:
        _lib.fail(f"expected 2 lines with --limit 2, got {lines}")
    expected0 = "tick=2 job=a status=completed exit_code=0 reason=-"
    expected1 = "tick=1 job=a status=completed exit_code=0 reason=-"
    if lines[0] != expected0 or lines[1] != expected1:
        _lib.fail(f"unexpected/misordered lines: {lines}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
