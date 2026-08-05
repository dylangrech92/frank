"""Core: a job command exiting non-zero does not change `run`'s own exit
code — the defining partial-failure-semantics behavior of Requirement 9.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | false\n")
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if proc.returncode != 0:
        _lib.fail(f"run must exit 0 even though the job failed; got {proc.returncode}")
    if "1 failed" not in proc.stdout:
        _lib.fail(f"expected tick summary to report the failure, got: {proc.stdout!r}")

    state = _lib.read_state(work)
    if state["jobs"]["a"]["last_status"] != "failed":
        _lib.fail(f"expected job 'a' recorded as failed, got {state['jobs']['a']}")
    if state["jobs"]["a"]["last_exit_code"] != 1:
        _lib.fail(f"expected last_exit_code 1, got {state['jobs']['a']['last_exit_code']}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
