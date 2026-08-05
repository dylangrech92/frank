#!/usr/bin/env python3
"""Core acceptance probe for b1_t2_report_order_corruption.

The reported bug: the "completed jobs (chronological)" report shows jobs
out of order whenever the "slowest jobs" panel was rendered first. This
probe reproduces exactly that dashboard sequence (5 jobs with distinct
durations, slowest-jobs panel, then the chronological report) and checks
the report's job order against true completion order.

Usage: python3 probe_core_chronological_order.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t2-core-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def _extract_job_id(report_line: str) -> str:
    # Report lines look like: "<finished_at>  <job_id>  <task_name>  <status>  <duration>s"
    return report_line.split()[1]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_core_chronological_order.py <tree_path>", file=sys.stderr)
        return 2
    tree_path = os.path.abspath(sys.argv[1])
    state_dir = _import_queueworks_from(tree_path)

    try:
        from queueworks.demo import scenario_dashboard
    except Exception as exc:
        print(f"FAIL: could not import queueworks.demo: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        result = scenario_dashboard(state_dir)
    except Exception as exc:
        print(f"FAIL: dashboard scenario raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    true_order = result["true_order"]
    report_lines = result["report_lines"]
    if len(report_lines) != len(true_order):
        print(
            f"FAIL: expected {len(true_order)} lines in the chronological report, got {len(report_lines)}",
            file=sys.stderr,
        )
        return 1

    report_order = [_extract_job_id(line) for line in report_lines]
    if report_order != true_order:
        print(
            "FAIL: chronological_report() is not in true completion order.\n"
            f"  expected: {true_order}\n"
            f"  actual:   {report_order}",
            file=sys.stderr,
        )
        return 1

    print("PASS: chronological report matches true completion order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
