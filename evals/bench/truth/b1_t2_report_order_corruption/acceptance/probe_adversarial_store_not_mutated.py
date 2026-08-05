#!/usr/bin/env python3
"""Adversarial acceptance probe for b1_t2_report_order_corruption.

Guards against a shallow fix that only re-sorts inside
``reporting.chronological_report()`` before formatting, without touching
the actual shared state. That would make the report *look* right while
leaving the underlying recent-completions feed corrupted for any other
consumer. This probe bypasses reporting.py entirely and reads
``store.get_completion_log()`` directly after the same stats-then-report
trigger sequence.

Usage: python3 probe_adversarial_store_not_mutated.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t2-adv-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_adversarial_store_not_mutated.py <tree_path>", file=sys.stderr)
        return 2
    tree_path = os.path.abspath(sys.argv[1])
    state_dir = _import_queueworks_from(tree_path)

    try:
        from queueworks import sample_tasks  # noqa: F401 -- registers "timed_work"
        from queueworks import stats, store
        from queueworks.manager import QueueManager
        from queueworks.models import Priority
        from queueworks.reporting import chronological_report
    except Exception as exc:
        print(f"FAIL: could not import queueworks: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    store.reset_completion_log()
    state_path = os.path.join(state_dir, "queue_state.json")

    try:
        manager = QueueManager(state_path)
        durations = [0.10, 0.02, 0.07, 0.03, 0.09]
        true_order = []
        for i, d in enumerate(durations):
            job = manager.enqueue(
                "timed_work", kwargs={"duration": d, "label": f"job-{i}"}, priority=Priority.NORMAL
            )
            true_order.append(job.id)
        manager.run_pending()

        stats.slowest_jobs(n=3)
        chronological_report()  # rendered once, as a real dashboard would
        log_order = [r.job_id for r in store.get_completion_log()]
    except Exception as exc:
        print(f"FAIL: dashboard scenario raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if log_order != true_order:
        print(
            "FAIL: store.get_completion_log() is still out of true completion order after a "
            "stats call -- the underlying shared state is corrupted, not just the report's "
            "rendering.\n"
            f"  expected: {true_order}\n"
            f"  actual:   {log_order}",
            file=sys.stderr,
        )
        return 1

    print("PASS: the underlying completion log is unmutated by generating stats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
