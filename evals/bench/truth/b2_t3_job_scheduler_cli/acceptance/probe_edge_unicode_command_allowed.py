"""Edge: the command field is exempt from the job_id ASCII restriction —
non-ASCII bytes in `command` must parse cleanly, since only fields 1-4 are
validated by Requirement 1's rules; `command` is everything remaining on the
line after the 4th '|'.

Requirement 1 (field-rule scope).
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
        "greet | 1 | 1 | - | echo héllo 日本語 ☃\n",
    )
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
    if proc.stdout.strip() != "OK: 1 jobs, 0 edges":
        _lib.fail(f"expected 'OK: 1 jobs, 0 edges', got {proc.stdout!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
