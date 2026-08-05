"""Command-line entry point for everyday queue operations.

This is the operational counterpart to :mod:`queueworks.demo` -- where the
demo module runs fixed scenarios for illustration, this module lets a real
caller enqueue work, drain the queue, and inspect what happened against a
state file of their own choosing::

    python3 -m queueworks.cli enqueue noop
    python3 -m queueworks.cli run
    python3 -m queueworks.cli status
    python3 -m queueworks.cli query --status failed
    python3 -m queueworks.cli purge
    python3 -m queueworks.cli stats --chronological
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import queries
from . import sample_tasks  # noqa: F401 -- registers the built-in demo tasks
from . import stats as stats_mod
from .config import QueueConfig
from .manager import QueueManager
from .models import JobStatus, Priority
from .reporting import chronological_report


def _priority_from_name(name: str) -> Priority:
    try:
        return Priority[name.upper()]
    except KeyError as exc:
        choices = ", ".join(p.name for p in Priority)
        raise SystemExit(f"unknown priority {name!r}; choose one of: {choices}") from exc


def _status_from_name(name: str) -> JobStatus:
    try:
        return JobStatus(name.lower())
    except ValueError as exc:
        choices = ", ".join(s.value for s in JobStatus)
        raise SystemExit(f"unknown status {name!r}; choose one of: {choices}") from exc


def _cmd_enqueue(manager: QueueManager, args: argparse.Namespace) -> int:
    kwargs = json.loads(args.kwargs) if args.kwargs else {}
    job = manager.enqueue(
        args.task_name,
        kwargs=kwargs,
        priority=_priority_from_name(args.priority),
    )
    print(f"enqueued {job.id} (task={job.task_name!r}, priority={job.priority.name})")
    return 0


def _cmd_run(manager: QueueManager, args: argparse.Namespace) -> int:
    processed = manager.run_pending()
    for job in processed:
        print(f"{job.id}: {job.status.value} (attempts={job.attempts})")
    print(f"processed {len(processed)} job(s), {manager.pending_count()} still pending")
    return 0


def _cmd_status(manager: QueueManager, args: argparse.Namespace) -> int:
    jobs = manager.store.all_jobs()
    if not jobs:
        print("no jobs in the store")
        return 0
    for job in sorted(jobs, key=lambda j: j.seq):
        print(f"{job.id}: {job.status.value} task={job.task_name!r} attempts={job.attempts}")
    return 0


def _cmd_query(manager: QueueManager, args: argparse.Namespace) -> int:
    if args.status:
        matches = queries.jobs_by_status(manager.store, _status_from_name(args.status))
    elif args.task:
        matches = queries.jobs_by_task(manager.store, args.task)
    else:
        counts = queries.status_counts(manager.store)
        if not counts:
            print("no jobs in the store")
            return 0
        for status_value, count in sorted(counts.items()):
            print(f"{status_value}: {count}")
        return 0

    for job in matches:
        print(f"{job.id}: {job.status.value} task={job.task_name!r} attempts={job.attempts}")
    return 0


def _cmd_purge(manager: QueueManager, args: argparse.Namespace) -> int:
    removed = manager.store.purge_completed()
    print(f"purged {removed} completed/failed job(s)")
    return 0


def _cmd_stats(manager: QueueManager, args: argparse.Namespace) -> int:
    slowest = stats_mod.slowest_jobs(n=args.top)
    print(f"success rate: {stats_mod.success_rate():.1%}")
    print(f"average duration: {stats_mod.average_duration():.3f}s")
    print(f"slowest {len(slowest)} job(s):")
    for record in slowest:
        print(f"  {record.job_id}: {record.duration:.3f}s ({record.task_name})")
    if args.chronological:
        print("chronological report:")
        for line in chronological_report():
            print(f"  {line}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="queueworks", description=__doc__)
    parser.add_argument(
        "--state-path",
        default=None,
        help="path to the JSON state file (default: $QUEUEWORKS_STATE_PATH or queueworks_state.json)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_enqueue = sub.add_parser("enqueue", help="enqueue a new job")
    p_enqueue.add_argument("task_name", help="name of a registered task")
    p_enqueue.add_argument("--kwargs", default=None, help="JSON object of keyword arguments")
    p_enqueue.add_argument("--priority", default="NORMAL", help="CRITICAL, HIGH, NORMAL, or LOW")
    p_enqueue.set_defaults(func=_cmd_enqueue)

    p_run = sub.add_parser("run", help="synchronously drain every pending job")
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="list every job the store knows about")
    p_status.set_defaults(func=_cmd_status)

    p_query = sub.add_parser("query", help="filter jobs by status or task, or summarize counts")
    p_query.add_argument("--status", default=None, help="pending, running, retry_scheduled, completed, or failed")
    p_query.add_argument("--task", default=None, help="a registered task name")
    p_query.set_defaults(func=_cmd_query)

    p_purge = sub.add_parser("purge", help="drop completed/failed jobs from the store")
    p_purge.set_defaults(func=_cmd_purge)

    p_stats = sub.add_parser("stats", help="print recent-completion statistics")
    p_stats.add_argument("--top", type=int, default=5, help="how many slowest jobs to show")
    p_stats.add_argument(
        "--chronological", action="store_true", help="also print the chronological report"
    )
    p_stats.set_defaults(func=_cmd_stats)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = QueueConfig.from_env()
    state_path = args.state_path or config.state_path
    manager = QueueManager(state_path, retry_policy=config.retry_policy())

    return args.func(manager, args)


if __name__ == "__main__":
    sys.exit(main())
