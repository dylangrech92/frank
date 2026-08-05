"""Adversarial: --dry-run must leave the ENTIRE working tree byte-identical,
including a pre-existing real state file and log from a prior non-dry-run
invocation. This is a stricter version of the core dry-run probe (which only
checks that .jobsched/ isn't created from nothing) — here .jobsched/ already
exists with real content, and a naive implementation that reuses the real
state-loading/persist code path with a 'just don't call replace() at the
very end' shortcut could still mutate an in-memory-then-serialized copy, or
touch file mtimes, or partially write before bailing. Full content hashes of
every file are compared before and after.

Requirement 3, step 3, dry-run sub-bullet.
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
        "a | 1 | 1 | -   | true\n"
        "b | 5 | 1 | a   | true\n"
        "c | 9 | 1 | b   | true\n",
    )
    seed = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if seed.returncode != 0:
        _lib.fail(f"seed run failed: {seed.returncode}; stderr={seed.stderr!r}")

    before_paths = _lib.snapshot_paths(work)
    before_hashes = _lib.snapshot_hashes(work)

    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "5", "--dry-run"])
    if proc.returncode != 0:
        _lib.fail(f"dry-run failed: {proc.returncode}; stderr={proc.stderr!r}")
    if "would-run" not in proc.stdout:
        _lib.fail(f"expected a dry-run plan, got: {proc.stdout!r}")

    after_paths = _lib.snapshot_paths(work)
    after_hashes = _lib.snapshot_hashes(work)

    if before_paths != after_paths:
        _lib.fail(f"dry-run changed the set of files on disk: before={before_paths} after={after_paths}")
    if before_hashes != after_hashes:
        changed = {
            k for k in before_hashes
            if before_hashes.get(k) != after_hashes.get(k)
        }
        _lib.fail(f"dry-run mutated file content: changed files = {sorted(changed)}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
