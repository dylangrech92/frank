"""Adversarial: `status` and `history`, run against a tree with NO prior
.jobsched/ at all, must print their 'nothing here' message and exit 0
WITHOUT creating .jobsched/ (or state.json/logs/) as a side effect — a
naive implementation that shares a "resolve default path, then
mkdir(parents=True) it before checking existence" helper with `run` would
fail this, since `run` legitimately creates that directory lazily but the
query commands must not.

Requirement 5 (lazy directory creation), Requirement 7 ("neither command
writes anything, ever").
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    before = _lib.snapshot_paths(work)

    status_proc = _lib.run_cli(work, ["status"])
    if status_proc.returncode != 0:
        _lib.fail(f"status: expected exit 0, got {status_proc.returncode}; stderr={status_proc.stderr!r}")
    if status_proc.stdout.strip() != "no state":
        _lib.fail(f"status: expected 'no state', got {status_proc.stdout!r}")

    history_proc = _lib.run_cli(work, ["history"])
    if history_proc.returncode != 0:
        _lib.fail(f"history: expected exit 0, got {history_proc.returncode}; stderr={history_proc.stderr!r}")
    if history_proc.stdout.strip() != "no history":
        _lib.fail(f"history: expected 'no history', got {history_proc.stdout!r}")

    if os.path.exists(os.path.join(work, ".jobsched")):
        _lib.fail("status/history created .jobsched/ as a side effect")

    after = _lib.snapshot_paths(work)
    if before != after:
        _lib.fail(f"status/history changed the tree: before={before} after={after}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
