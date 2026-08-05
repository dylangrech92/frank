"""Edge: `cancel` validates job_id against the SCHEDULE, not the state --
an id that has never appeared in state but is a real schedule job must be
cancellable (see probe_core_cancel_status_display), while an id that is not
defined anywhere in the schedule is a usage error (exit 6), not a
silent no-op and not an internal crash.

Requirement 10 edge case.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")

    proc = _lib.run_cli(work, ["cancel", "jobs.txt", "bogus"])
    if proc.returncode != 6:
        _lib.fail(f"expected exit 6, got {proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}")
    expected = "error: cancel: unknown job id 'bogus'"
    if proc.stderr.strip() != expected:
        _lib.fail(f"expected {expected!r}, got {proc.stderr!r}")

    if os.path.exists(os.path.join(work, ".jobsched")):
        _lib.fail("cancelling an unknown job id must not create .jobsched/")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
