#!/usr/bin/env python3
"""Core acceptance probe for b1_t1_restart_crash.

The reported bug: restarting the process after some jobs have completed
and been purged, then enqueuing more work, crashes with a TypeError before
anything gets scheduled. This probe reproduces exactly that sequence and
asserts the queue drains cleanly.

Usage: python3 probe_core_restart_no_crash.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    """Copy tree_path into a scratch dir and import queueworks from the copy.

    The probe never touches tree_path itself -- everything (module import
    caches, the on-disk queue state the scenario writes) happens inside a
    disposable copy.
    """

    base = tempfile.mkdtemp(prefix="b1-t1-core-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_core_restart_no_crash.py <tree_path>", file=sys.stderr)
        return 2
    tree_path = os.path.abspath(sys.argv[1])
    state_dir = _import_queueworks_from(tree_path)

    try:
        from queueworks.demo import scenario_restart_seq_collision
    except Exception as exc:
        print(f"FAIL: could not import queueworks.demo: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        result = scenario_restart_seq_collision(state_dir)
    except Exception as exc:
        print(
            f"FAIL: restart scenario raised {type(exc).__name__}: {exc} "
            "(expected it to run to completion without crashing)",
            file=sys.stderr,
        )
        return 1

    statuses = result["final_statuses"]
    if len(statuses) != 3:
        print(f"FAIL: expected 3 jobs processed after restart, got {len(statuses)}", file=sys.stderr)
        return 1
    if any(s != "completed" for s in statuses):
        print(f"FAIL: expected all jobs to complete, got statuses {statuses}", file=sys.stderr)
        return 1

    print("PASS: restart scenario completed 3/3 jobs without crashing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
