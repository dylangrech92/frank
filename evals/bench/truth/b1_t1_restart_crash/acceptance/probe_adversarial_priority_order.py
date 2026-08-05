#!/usr/bin/env python3
"""Adversarial acceptance probe for b1_t1_restart_crash.

Guards against a shallow fix (e.g. swallowing the TypeError and dropping
whichever job collided) by checking that after a collision-triggering
restart, every job is still processed exactly once and in strict priority
order -- not just that the process didn't crash.

Usage: python3 probe_adversarial_priority_order.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t1-adv-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_adversarial_priority_order.py <tree_path>", file=sys.stderr)
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
        m1 = QueueManager(state_path)
        ids = {}
        ids["normal_a"] = m1.enqueue("noop", priority=Priority.NORMAL).id
        ids["normal_b"] = m1.enqueue("noop", priority=Priority.NORMAL).id
        ids["normal_c"] = m1.enqueue("noop", priority=Priority.NORMAL).id

        worker = Worker(m1.queue, m1.store)
        worker.run_one()  # completes normal_a
        m1.store.purge_completed()

        # Restart: the store now resumes its sequence counter from only
        # what's left on disk (normal_b, normal_c). The first job enqueued
        # below shares a priority with normal_c and, under the bug, ends
        # up with the exact same sequence number.
        m2 = QueueManager(state_path)
        ids["normal_new"] = m2.enqueue("noop", priority=Priority.NORMAL).id
        ids["critical_new"] = m2.enqueue("noop", priority=Priority.CRITICAL).id
        ids["high_new"] = m2.enqueue("noop", priority=Priority.HIGH).id

        processed = m2.run_pending()
    except Exception as exc:
        print(
            f"FAIL: priority-order restart scenario raised {type(exc).__name__}: {exc} "
            "(expected it to run to completion without crashing)",
            file=sys.stderr,
        )
        return 1

    order = [j.id for j in processed]
    expected = [
        ids["critical_new"],
        ids["high_new"],
        ids["normal_b"],
        ids["normal_c"],
        ids["normal_new"],
    ]

    if set(order) != set(expected):
        print(
            f"FAIL: expected exactly these jobs processed once each: {sorted(expected)}, "
            f"got: {sorted(order)}",
            file=sys.stderr,
        )
        return 1

    if order != expected:
        print(f"FAIL: wrong processing order.\n  expected: {expected}\n  actual:   {order}", file=sys.stderr)
        return 1

    print("PASS: all 5 jobs processed exactly once, in strict priority order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
