#!/usr/bin/env python3
"""Edge acceptance probe for b1_t1_restart_crash.

Exercises the same restart-after-purge shape as the core probe but with a
bigger gap between how many jobs remain in the store and how many sequence
numbers have actually been handed out (6 enqueued, 4 completed and purged,
then 4 more enqueued after "restart"). A fix that only happens to work for
a gap of one job should still be caught here.

Usage: python3 probe_edge_larger_gap.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t1-edge-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_edge_larger_gap.py <tree_path>", file=sys.stderr)
        return 2
    tree_path = os.path.abspath(sys.argv[1])
    state_dir = _import_queueworks_from(tree_path)

    try:
        from queueworks.manager import QueueManager
        from queueworks.models import Priority
        from queueworks.worker import Worker
        from queueworks import sample_tasks  # noqa: F401 -- registers "noop"
    except Exception as exc:
        print(f"FAIL: could not import queueworks: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    state_path = os.path.join(state_dir, "queue_state.json")

    try:
        manager = QueueManager(state_path)
        for _ in range(6):
            manager.enqueue("noop", priority=Priority.NORMAL)

        worker = Worker(manager.queue, manager.store)
        for _ in range(4):
            worker.run_one()
        manager.store.purge_completed()

        resumed = QueueManager(state_path)
        for _ in range(4):
            resumed.enqueue("noop", priority=Priority.NORMAL)

        processed = resumed.run_pending()
    except Exception as exc:
        print(
            f"FAIL: larger-gap restart scenario raised {type(exc).__name__}: {exc} "
            "(expected it to run to completion without crashing)",
            file=sys.stderr,
        )
        return 1

    if len(processed) != 6:
        print(f"FAIL: expected 6 jobs processed after restart, got {len(processed)}", file=sys.stderr)
        return 1
    statuses = [j.status.value for j in processed]
    if any(s != "completed" for s in statuses):
        print(f"FAIL: expected all 6 jobs to complete, got statuses {statuses}", file=sys.stderr)
        return 1

    print("PASS: larger-gap restart scenario completed 6/6 jobs without crashing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
