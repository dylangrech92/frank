"""Edge: a non-integer priority field is rejected by rule 3 with the exact
message, distinct from the interval rule (rule 4) and the field-count rule
(rule 1).

Requirement 1, rule 3.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | high | 1 | - | true\n")
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 2:
        _lib.fail(f"expected exit 2, got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    expected = "error: line 1: priority must be an integer"
    if proc.stderr.strip() != expected:
        _lib.fail(f"expected {expected!r}, got {proc.stderr!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
