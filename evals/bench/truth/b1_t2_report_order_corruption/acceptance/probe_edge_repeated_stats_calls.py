#!/usr/bin/env python3
"""Edge acceptance probe for b1_t2_report_order_corruption.

Some dashboards re-render their "slowest jobs" panel more than once per
page view (a refresh, a second widget using the same data). This probe
calls the stats panel twice before rendering the chronological report and
checks the report is still in true completion order -- a fix that happens
to work for exactly one stats call should still be caught here.

Usage: python3 probe_edge_repeated_stats_calls.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t2-edge-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_edge_repeated_stats_calls.py <tree_path>", file=sys.stderr)
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
        stats.slowest_jobs(n=2)  # a second panel render
        report_lines = chronological_report()
    except Exception as exc:
        print(f"FAIL: dashboard scenario raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    report_order = [line.split()[1] for line in report_lines]
    if report_order != true_order:
        print(
            "FAIL: chronological report is out of order after two stats panel renders.\n"
            f"  expected: {true_order}\n"
            f"  actual:   {report_order}",
            file=sys.stderr,
        )
        return 1

    print("PASS: chronological report survives repeated stats panel renders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
