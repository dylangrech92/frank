"""Adversarial: cycle reporting must re-root at the lexicographically
smallest participating job id and follow depends_on edges (job -> its
dependency) exactly once around, REGARDLESS of the order jobs appear in the
file. A naive implementation that reports the cycle starting from whichever
job it happens to visit first in a DFS (e.g. file order, or dict iteration
order) will fail this: the file below defines the cycle starting at 'zeta',
but the smallest id in the cycle is 'alpha'.

Requirement 2 (cycle detection, exact message format).
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
        "zeta  | 1 | 1 | mu    | true\n"
        "mu    | 1 | 1 | alpha | true\n"
        "alpha | 1 | 1 | zeta  | true\n",
    )
    proc = _lib.run_cli(work, ["validate", "jobs.txt"])
    if proc.returncode != 3:
        _lib.fail(f"expected exit 3, got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    expected = "error: dag: cycle detected: alpha -> zeta -> mu -> alpha"
    if proc.stderr.strip() != expected:
        _lib.fail(f"expected {expected!r}, got {proc.stderr!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
