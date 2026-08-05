"""Runnable demo scenarios for queueworks.

Usage::

    python3 -m queueworks.demo restart
    python3 -m queueworks.demo dashboard
    python3 -m queueworks.demo retries

Each scenario is also exposed as a plain function (``scenario_*``) that
takes a working directory and returns a small dict describing what
happened, so the same code path used for the human-facing demo can be
driven programmatically (e.g. from an integration test).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List

from . import sample_tasks  # noqa: F401 -- registers the demo task functions
from . import stats, store
from .manager import QueueManager
from .models import Priority
from .reporting import chronological_report
from .retry import RetryPolicy
from .worker import Worker


def scenario_restart_seq_collision(work_dir: str) -> dict:
    """Enqueue jobs, process some, restart, and enqueue more.

    This mirrors an application that keeps its queue state on disk so it
    can resume after a process restart: a batch of jobs gets enqueued, a
    couple finish and are cleaned up, the process comes back up and reads
    the state file back in, and then more work is enqueued on top of
    whatever was still pending.
    """

    state_path = os.path.join(work_dir, "queue_state.json")

    manager = QueueManager(state_path)
    for i in range(3):
        manager.enqueue("noop", priority=Priority.NORMAL)

    # Process the oldest job, then clean up finished jobs the way a
    # long-running deployment periodically would.
    worker = Worker(manager.queue, manager.store)
    worker.run_one()
    removed = manager.store.purge_completed()

    # Simulate a restart: a brand new process picks the state file back up.
    resumed = QueueManager(state_path)
    resumed.enqueue("noop", priority=Priority.NORMAL)

    processed = resumed.run_pending()
    return {
        "purged_on_shutdown": removed,
        "final_statuses": [j.status.value for j in processed],
        "processed_ids": [j.id for j in processed],
    }


def scenario_dashboard(work_dir: str) -> dict:
    """Run a handful of jobs with distinct durations, then render two panels.

    First the "slowest jobs" panel (:func:`queueworks.stats.slowest_jobs`),
    then the "completed jobs, chronological" panel
    (:func:`queueworks.reporting.chronological_report`) -- the order a real
    ops dashboard renders them in, slowest-first summary at the top and the
    full timeline below.
    """

    state_path = os.path.join(work_dir, "queue_state.json")
    store.reset_completion_log()

    manager = QueueManager(state_path)
    durations = [0.30, 0.05, 0.20, 0.10, 0.25]
    enqueued_ids = []
    for i, d in enumerate(durations):
        job = manager.enqueue(
            "timed_work",
            kwargs={"duration": d, "label": f"job-{i}"},
            priority=Priority.NORMAL,
        )
        enqueued_ids.append(job.id)

    # A single worker draining a FIFO-ordered queue finishes jobs in the
    # order they were enqueued, so enqueue order is also true completion
    # (chronological) order here.
    manager.run_pending()
    true_order = enqueued_ids

    slowest = stats.slowest_jobs(n=3)
    report_lines = chronological_report()
    log_order = [rec.job_id for rec in store.get_completion_log()]

    return {
        "true_order": true_order,
        "slowest_job_ids": [r.job_id for r in slowest],
        "report_lines": report_lines,
        "log_order_after_dashboard": log_order,
    }


def scenario_retry_exhaustion(work_dir: str, sleep_fn=None) -> dict:
    """Run a job whose task always fails, with a fixed retry policy.

    ``sleep_fn`` defaults to ``time.sleep``; callers that want the scenario
    to run instantly (e.g. acceptance probes) can pass a no-op.
    """

    import time as _time

    state_path = os.path.join(work_dir, "queue_state.json")
    policy = RetryPolicy(max_retries=3, base_delay=0.05, backoff_factor=4.0, max_delay=1.0)
    manager = QueueManager(state_path, retry_policy=policy)
    job = manager.enqueue(
        "always_fails",
        kwargs={"message": "simulated failure"},
        priority=Priority.NORMAL,
    )

    worker = Worker(
        manager.queue,
        manager.store,
        retry_policy=policy,
        sleep_fn=sleep_fn or _time.sleep,
    )
    worker.run_one()

    finished = manager.store.get(job.id)
    return {
        "final_status": finished.status.value,
        "total_attempts": finished.attempts,
        "retry_delays": list(finished.retry_delays),
        "max_delay": policy.max_delay,
        "max_retries": policy.max_retries,
    }


def _print_lines(lines: List[str]) -> None:
    for line in lines:
        print(line)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m queueworks.demo",
        description="Run a queueworks demo scenario against a scratch state directory.",
    )
    parser.add_argument(
        "scenario",
        choices=["restart", "dashboard", "retries"],
        help="which scenario to run",
    )
    args = parser.parse_args(argv)

    work_dir = tempfile.mkdtemp(prefix="queueworks-demo-")
    try:
        if args.scenario == "restart":
            print("Enqueuing 3 jobs, processing 1, purging it, then restarting...")
            result = scenario_restart_seq_collision(work_dir)
            print(f"Purged {result['purged_on_shutdown']} completed job(s) before restart.")
            print("Resumed process enqueued 1 more job and ran the queue to completion:")
            for status, job_id in zip(result["final_statuses"], result["processed_ids"]):
                print(f"  {job_id}: {status}")

        elif args.scenario == "dashboard":
            print("Running 5 jobs with different durations...")
            result = scenario_dashboard(work_dir)
            print("\n--- Slowest jobs ---")
            for job_id in result["slowest_job_ids"]:
                print(f"  {job_id}")
            print("\n--- Completed jobs (chronological) ---")
            _print_lines(result["report_lines"])
            print("\ntrue completion order:", result["true_order"])
            print("report/log order:     ", result["log_order_after_dashboard"])

        elif args.scenario == "retries":
            print("Running a task that always fails, with max_retries=3...")
            result = scenario_retry_exhaustion(work_dir)
            print(f"Final status: {result['final_status']}")
            print(f"Total attempts made: {result['total_attempts']} (expected 4)")
            print("Backoff delays used before each retry:")
            for i, d in enumerate(result["retry_delays"], start=1):
                print(f"  retry {i}: waited {d:.3f}s (max_delay={result['max_delay']:.3f}s)")

        return 0
    finally:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
