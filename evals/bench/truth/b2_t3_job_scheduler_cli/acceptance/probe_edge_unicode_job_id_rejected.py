"""Edge: a job_id containing a non-ASCII character is rejected with the
exact parse-error message and exit code, even though it would satisfy a
naive '\\w+'-style regex in a locale/unicode-aware engine.

Requirement 1, rule 2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "jöb | 1 | 1 | - | true\n")
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 2:
        _lib.fail(f"expected exit 2, got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    expected = "error: line 1: invalid job id 'jöb' (must match [A-Za-z0-9_-]+)"
    if proc.stderr.strip() != expected:
        _lib.fail(f"expected {expected!r}, got {proc.stderr!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
