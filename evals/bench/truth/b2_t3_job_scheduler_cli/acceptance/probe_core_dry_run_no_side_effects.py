"""Core: --dry-run prints a plan but creates no runtime files.

Requirement 5 happy path (also see the adversarial band for a stricter
byte-identical filesystem check).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\nb | 1 | 1 | a | true\n")
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "3", "--dry-run"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")
    if "would-run" not in proc.stdout:
        _lib.fail(f"expected a dry-run plan mentioning 'would-run', got: {proc.stdout!r}")
    if "done: 3 ticks planned (dry-run, no changes made)" not in proc.stdout:
        _lib.fail(f"missing dry-run summary line, got: {proc.stdout!r}")
    if os.path.exists(os.path.join(work, ".jobsched")):
        _lib.fail("--dry-run created .jobsched/")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
