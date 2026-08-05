"""Core data types shared across the queueworks package."""

from __future__ import annotations

import dataclasses
import enum
import itertools
import time
import uuid
from typing import Any, NamedTuple, Optional


class Priority(enum.IntEnum):
    """Job priority. Lower numeric value is scheduled first."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class JobStatus(str, enum.Enum):
    """Lifecycle states a :class:`Job` moves through."""

    PENDING = "pending"
    RUNNING = "running"
    RETRY_SCHEDULED = "retry_scheduled"
    COMPLETED = "completed"
    FAILED = "failed"


_id_counter = itertools.count(1)


def _new_job_id() -> str:
    return f"job-{next(_id_counter)}-{uuid.uuid4().hex[:8]}"


@dataclasses.dataclass
class Job:
    """A unit of work tracked by the queue.

    ``task_name`` is a lookup key into the task registry (see
    :mod:`queueworks.registry`) rather than a pickled callable, so a job
    can be serialized to JSON and safely reloaded in a new process.

    ``seq`` is assigned once, at enqueue time, and is used purely to break
    ties between jobs that share the same priority so the queue behaves as
    a stable FIFO within a priority band.
    """

    task_name: str
    args: list = dataclasses.field(default_factory=list)
    kwargs: dict = dataclasses.field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    id: str = dataclasses.field(default_factory=_new_job_id)
    seq: int = 0
    status: JobStatus = JobStatus.PENDING
    attempts: int = 0
    created_at: float = dataclasses.field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    result: Any = None
    retry_delays: list = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["priority"] = int(self.priority)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        data = dict(data)
        data["priority"] = Priority(data["priority"])
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class CompletionRecord(NamedTuple):
    """A lightweight record of one finished job, used for stats/reporting."""

    job_id: str
    task_name: str
    status: str
    finished_at: float
    duration: float
