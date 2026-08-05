"""Core: validate -> run -> status -> history all succeed in sequence and
agree with each other.

Integration happy path across Requirements 1, 3, 8.
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
        "fetch   | 3 | 1 | -      | true\n"
        "process | 2 | 1 | fetch  | true\n"
        "publish | 1 | 2 | process | true\n",
    )

    v = _lib.run_cli(work, ["validate", "jobs.txt"])
    if v.returncode != 0:
        _lib.fail(f"validate failed: {v.returncode} {v.stderr!r}")

    r = _lib.run_cli(work, ["run", "jobs.txt", "--until", "4"])
    if r.returncode != 0:
        _lib.fail(f"run failed: {r.returncode} {r.stderr!r}")

    s = _lib.run_cli(work, ["status"])
    if s.returncode != 0:
        _lib.fail(f"status failed: {s.returncode} {s.stderr!r}")
    if "last_completed_tick: 3" not in s.stdout:
        _lib.fail(f"status did not report tick 3 complete: {s.stdout!r}")

    h = _lib.run_cli(work, ["history"])
    if h.returncode != 0:
        _lib.fail(f"history failed: {h.returncode} {h.stderr!r}")
    if "job=fetch" not in h.stdout or "job=publish" not in h.stdout:
        _lib.fail(f"history missing expected job events: {h.stdout!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
