"""Adversarial: duplicate job id detection is reported at the LATER line's
number (Section 1, rule 6), scanning top-down -- not the first occurrence,
not the last line in the file, and only after a distinct, non-duplicate
line in between (so a naive "adjacent-lines-only" check would also pass
this by accident; this schedule specifically defeats that).

Requirement 1 adversarial case (exact message + line number).
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
        "dup   | 1 | 1 | - | true\n"
        "other | 1 | 1 | - | true\n"
        "dup   | 2 | 1 | - | true\n",
    )

    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 2:
        _lib.fail(f"expected exit 2, got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    expected = "error: line 3: duplicate job id 'dup'"
    if proc.stderr.strip() != expected:
        _lib.fail(f"expected {expected!r}, got {proc.stderr!r}")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
