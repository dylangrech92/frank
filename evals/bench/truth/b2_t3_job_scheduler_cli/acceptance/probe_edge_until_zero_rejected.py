"""Edge: `--until 0` is syntactically an integer but fails the '>= 1'
requirement, so it must hit the same usage-error path (exit 6) as a
non-integer value, not exit 2 or a Python-level crash.

Requirement 3 / Section 9 exit-code contract.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "0"])
    if proc.returncode != 6:
        _lib.fail(f"expected exit 6, got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    if proc.stderr.strip() != "error: --until requires a positive integer":
        _lib.fail(f"unexpected stderr: {proc.stderr!r}")
    if os.path.exists(os.path.join(work, ".jobsched")):
        _lib.fail("--until 0 must not create .jobsched/")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
