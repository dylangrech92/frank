"""Read-only helpers for slicing a job store's contents.

These are convenience wrappers around :meth:`JobStore.all_jobs` for the
common ways an operator or a dashboard wants to slice the current job
population -- by status, by task, or just "what's currently broken".
"""

from __future__ import annotations

from typing import Dict, List

from .models import Job, JobStatus
from .store import JobStore


def jobs_by_status(store: JobStore, status: JobStatus) -> List[Job]:
    """Return every job currently in ``status``, oldest-enqueued first."""

    matches = [job for job in store.all_jobs() if job.status == status]
    matches.sort(key=lambda j: j.seq)
    return matches


def jobs_by_task(store: JobStore, task_name: str) -> List[Job]:
    """Return every job for ``task_name``, oldest-enqueued first."""

    matches = [job for job in store.all_jobs() if job.task_name == task_name]
    matches.sort(key=lambda j: j.seq)
    return matches


def status_counts(store: JobStore) -> Dict[str, int]:
    """Return a ``{status_value: count}`` breakdown of every job in the store."""

    counts: Dict[str, int] = {}
    for job in store.all_jobs():
        counts[job.status.value] = counts.get(job.status.value, 0) + 1
    return counts


def failed_job_summaries(store: JobStore) -> List[str]:
    """Return one human-readable line per currently-failed job."""

    lines = []
    for job in jobs_by_status(store, JobStatus.FAILED):
        reason = job.error or "unknown error"
        lines.append(
            f"{job.id} ({job.task_name}): failed after {job.attempts} attempt(s) -- {reason}"
        )
    return lines
