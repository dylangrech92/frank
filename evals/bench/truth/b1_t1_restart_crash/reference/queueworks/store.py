"""Persistent storage for jobs, plus a small in-memory completion feed.

:class:`JobStore` is the single source of truth for job state. It keeps an
in-memory index of every job it knows about and mirrors that index to a JSON
file on disk so a process can resume its queue after a restart.

Separately, this module keeps a small module-level cache of recently
finished jobs (:data:`_recent_completions`). It exists so lightweight
dashboards and stats helpers can look at "what just finished" without
re-reading the full job history from disk on every call. It is capped so it
never grows unbounded, and it is process-wide rather than per-``JobStore``
instance because multiple parts of an application (a web dashboard, a CLI
report, a metrics exporter) may all want the same recent-activity feed
without wiring a shared store object through every layer.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, Iterable, List, Optional

from .models import CompletionRecord, Job, JobStatus

_TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})

# Process-wide feed of recently finished jobs, newest last. Capped by
# _MAX_COMPLETION_LOG so long-running processes don't leak memory.
_recent_completions: List[CompletionRecord] = []
_MAX_COMPLETION_LOG = 200


def record_completion(record: CompletionRecord) -> None:
    """Append ``record`` to the process-wide recent-completions feed."""

    _recent_completions.append(record)
    if len(_recent_completions) > _MAX_COMPLETION_LOG:
        del _recent_completions[: len(_recent_completions) - _MAX_COMPLETION_LOG]


def get_completion_log() -> List[CompletionRecord]:
    """Return the recent-completions feed, oldest first."""

    return _recent_completions


def reset_completion_log() -> None:
    """Clear the recent-completions feed. Mainly useful for demos/tests."""

    _recent_completions.clear()


class JobStore:
    """JSON-file-backed persistence for :class:`~queueworks.models.Job`.

    The store keeps every job it has ever seen in memory (``self._jobs``)
    until it is explicitly purged, and mirrors the non-purged set to disk
    on every mutation so a fresh :class:`JobStore` pointed at the same path
    picks up where the last one left off.
    """

    def __init__(self, path: str):
        self.path = path
        self._jobs: Dict[str, Job] = {}
        self._seq_counter = 0
        if os.path.exists(path):
            self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for raw in data.get("jobs", []):
            job = Job.from_dict(raw)
            self._jobs[job.id] = job
        # Resume the sequence counter so newly enqueued jobs never reuse a
        # seq value already held by a job still sitting in the store. The
        # counter itself is not persisted (older state files predate it),
        # so it is reconstructed from what is currently on disk.
        self._seq_counter = max((job.seq for job in self._jobs.values()), default=0)

    def _save(self) -> None:
        payload = {"jobs": [job.to_dict() for job in self._jobs.values()]}
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".queueworks-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # -- sequence numbers --------------------------------------------------

    def next_seq(self) -> int:
        """Return the next sequence number, for priority-queue tie-breaking."""

        self._seq_counter += 1
        return self._seq_counter

    # -- job lifecycle -----------------------------------------------------

    def add(self, job: Job) -> None:
        """Register a newly created job and persist it."""

        self._jobs[job.id] = job
        self._save()

    def update(self, job: Job) -> None:
        """Persist an in-place update to a job already known to the store."""

        self._jobs[job.id] = job
        self._save()

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def all_jobs(self) -> List[Job]:
        return list(self._jobs.values())

    def pending_jobs(self) -> List[Job]:
        return [
            j
            for j in self._jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.RETRY_SCHEDULED)
        ]

    def purge_completed(self) -> int:
        """Drop terminal (completed/failed) jobs from the store.

        Real deployments call this periodically so the state file does not
        grow forever. Returns the number of jobs removed.
        """

        to_remove = [jid for jid, j in self._jobs.items() if j.status in _TERMINAL_STATUSES]
        for jid in to_remove:
            del self._jobs[jid]
        if to_remove:
            self._save()
        return len(to_remove)

    def mark_completed(self, job: Job, result) -> None:
        job.status = JobStatus.COMPLETED
        job.result = result
        job.error = None
        self.update(job)
        record_completion(
            CompletionRecord(
                job_id=job.id,
                task_name=job.task_name,
                status=job.status.value,
                finished_at=job.finished_at or 0.0,
                duration=job.duration or 0.0,
            )
        )

    def mark_failed(self, job: Job, error: str) -> None:
        job.status = JobStatus.FAILED
        job.error = error
        self.update(job)
        record_completion(
            CompletionRecord(
                job_id=job.id,
                task_name=job.task_name,
                status=job.status.value,
                finished_at=job.finished_at or 0.0,
                duration=job.duration or 0.0,
            )
        )
