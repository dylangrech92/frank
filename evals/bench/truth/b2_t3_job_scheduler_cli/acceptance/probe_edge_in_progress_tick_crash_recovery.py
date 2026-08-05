"""Edge: a state file left with `in_progress_tick` set (simulating a process
killed after step 1's pre-execution write but before step 4's completion
write) causes the next `run` to resume AT that tick, not skip past it or
restart from tick 0.

Requirement 5, final paragraph (crash recovery).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")

    first = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if first.returncode != 0:
        _lib.fail(f"first run: expected exit 0, got {first.returncode}; stderr={first.stderr!r}")

    state = _lib.read_state(work)
    if state["last_completed_tick"] != 0:
        _lib.fail(f"expected last_completed_tick=0 after first run, got {state}")

    # Hand-craft a crash: mark tick 1 as in-progress, as step 1 of the tick
    # loop would have written right before a kill, with no matching step-4
    # completion write.
    import json
    state["in_progress_tick"] = 1
    with open(os.path.join(work, ".jobsched", "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)

    second = _lib.run_cli(work, ["run", "jobs.txt", "--until", "3"])
    if second.returncode != 0:
        _lib.fail(f"recovery run: expected exit 0, got {second.returncode}; stderr={second.stderr!r}")
    if "tick 1:" not in second.stdout:
        _lib.fail(f"recovery run must re-attempt tick 1, got stdout={second.stdout!r}")
    if "tick 0:" in second.stdout:
        _lib.fail(f"recovery run must NOT restart from tick 0, got stdout={second.stdout!r}")

    final_state = _lib.read_state(work)
    if final_state["last_completed_tick"] != 2:
        _lib.fail(f"expected last_completed_tick=2 after recovery run to --until 3, got {final_state}")
    if final_state["in_progress_tick"] is not None:
        _lib.fail(f"expected in_progress_tick cleared, got {final_state}")
    if final_state["jobs"]["a"]["run_count"] != 3:
        _lib.fail(f"expected run_count=3 (ticks 0,1,2 each due), got {final_state['jobs']['a']}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
