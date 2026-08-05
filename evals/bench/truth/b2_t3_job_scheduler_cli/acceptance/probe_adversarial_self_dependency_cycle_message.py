"""Adversarial: a job that depends on itself is a 1-node cycle. The cycle
re-rooting rule (start at the lexicographically smallest participant,
Section 2) must degrade correctly to this minimal case: 'x -> x', not a
crash (e.g. from an empty stack.index result or an off-by-one in the walk)
and not silently treated as a valid, acyclic 0-edge-among-others graph.

Requirement 2 adversarial case (self-dependency).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "x | 1 | 1 | x | true\n")

    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 3:
        _lib.fail(f"expected exit 3, got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    expected = "error: dag: cycle detected: x -> x"
    if proc.stderr.strip() != expected:
        _lib.fail(f"expected {expected!r}, got {proc.stderr!r}")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
