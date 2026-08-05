"""Edge: an empty schedule file (zero bytes) is valid — zero jobs, zero
edges, exit 0.

Requirement 1, final paragraph.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "")
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
    if proc.stdout.strip() != "OK: 0 jobs, 0 edges":
        _lib.fail(f"expected 'OK: 0 jobs, 0 edges', got {proc.stdout!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
