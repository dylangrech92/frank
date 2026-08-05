"""High-level facade wiring the store, queue, and workers together.

Most applications only need :class:`QueueManager`: it owns a
:class:`~queueworks.store.JobStore` and a
:class:`~queueworks.queue.PriorityQueue`, loads any pending work left over
from a previous run, and exposes ``enqueue``/``run_pending`` so callers do
not have to wire the pieces together themselves.
"""

from __future__ import annotations

from typing import List, Optional

from .models import Job, Priority
from .queue import PriorityQueue
from .retry import RetryPolicy
from .store import JobStore
from .worker import Worker


class QueueManager:
    """Ties a :class:`JobStore` and :class:`PriorityQueue` together.

    On construction, any jobs left pending or mid-retry from a previous run
    (as recorded in the store at ``state_path``) are loaded back onto the
    in-memory queue, so a restarted process picks up where it left off.
    """

    def __init__(self, state_path: str, retry_policy: Optional[RetryPolicy] = None):
        self.store = JobStore(state_path)
        self.queue = PriorityQueue()
        self.retry_policy = retry_policy or RetryPolicy()
        for job in self.store.pending_jobs():
            self.queue.push(job)

    def enqueue(
        self,
        task_name: str,
        *,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
        priority: Priority = Priority.NORMAL,
    ) -> Job:
        """Create, persist, and schedule a new job."""

        job = Job(
            task_name=task_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
        )
        job.seq = self.store.next_seq()
        self.store.add(job)
        self.queue.push(job)
        return job

    def run_pending(self) -> List[Job]:
        """Synchronously process every job currently on the queue.

        Returns the jobs in the order they were processed. Suitable for
        small/batch workloads and for demos; long-running services should
        use :class:`~queueworks.worker.WorkerPool` instead.
        """

        worker = Worker(self.queue, self.store, self.retry_policy)
        return worker.drain()

    def pending_count(self) -> int:
        return len(self.queue)
