"""Core: the run log rotates once it exceeds 10000 bytes.

Requirement 7 happy path (rotation).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "j | 1 | 1 | - | true\n")
    # ~20 bytes/line * 600 ticks comfortably exceeds the 10000-byte threshold.
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "600"], timeout=180)
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log_dir = os.path.join(work, ".jobsched", "logs")
    rotated = os.path.join(log_dir, "run.log.1")
    if not os.path.exists(rotated):
        _lib.fail("expected run.log.1 to exist after exceeding the rotation threshold")

    current = os.path.join(log_dir, "run.log")
    if os.path.getsize(current) > 10000:
        _lib.fail(f"run.log should have rotated, but is {os.path.getsize(current)} bytes")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
