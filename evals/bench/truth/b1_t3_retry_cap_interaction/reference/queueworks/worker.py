"""Workers: pull jobs off a queue, execute them, and drive retries.

A single :class:`Worker` runs jobs synchronously and is the building block
used by both the demo scenarios and :class:`WorkerPool`, which runs several
workers on background threads for a continuously-running queue.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

from . import registry
from .models import Job, JobStatus
from .queue import PriorityQueue
from .retry import RetryPolicy
from .store import JobStore


class Worker:
    """Executes jobs pulled from a :class:`PriorityQueue`, one at a time.

    Retries are handled synchronously: on failure, if the retry policy has
    an attempt left, the worker waits out the backoff delay and re-runs the
    job in place rather than re-enqueueing it. This keeps ordering simple
    for a library this size, at the cost of tying up the worker for the
    duration of the backoff.
    """

    def __init__(
        self,
        queue: PriorityQueue,
        store: JobStore,
        retry_policy: Optional[RetryPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.queue = queue
        self.store = store
        self.retry_policy = retry_policy or RetryPolicy()
        self.clock = clock
        self.sleep_fn = sleep_fn

    def run_one(self) -> Optional[Job]:
        """Pop and fully process a single job. Returns None if the queue is empty."""

        job = self.queue.pop()
        if job is None:
            return None
        self._run_with_retries(job)
        return job

    def drain(self) -> List[Job]:
        """Run jobs one at a time until the queue is empty."""

        processed = []
        while True:
            job = self.run_one()
            if job is None:
                break
            processed.append(job)
        return processed

    def _run_with_retries(self, job: Job) -> None:
        fn = registry.get_task(job.task_name)

        while True:
            job.status = JobStatus.RUNNING
            job.started_at = self.clock()
            job.attempts += 1
            try:
                result = fn(*job.args, **job.kwargs)
            except Exception as exc:  # a task's own failure is data, not a worker bug
                job.finished_at = self.clock()
                job.duration = job.finished_at - job.started_at
                job.error = str(exc)

                if self.retry_policy.should_retry(job):
                    raw_delay = self.retry_policy.compute_backoff(job.attempts)
                    delay = min(raw_delay, self.retry_policy.max_delay)
                    job.retry_delays.append(delay)
                    job.status = JobStatus.RETRY_SCHEDULED
                    self.store.update(job)
                    self.sleep_fn(delay)
                    continue

                self.store.mark_failed(job, str(exc))
                return
            else:
                job.finished_at = self.clock()
                job.duration = job.finished_at - job.started_at
                self.store.mark_completed(job, result)
                return


class WorkerPool:
    """Runs several :class:`Worker` instances on background threads.

    Each thread loops ``Worker.run_one()`` until the shared queue is empty
    and :meth:`stop` has been called (or the pool is used as a context
    manager). Intended for a long-running process; the demo scenarios use a
    single synchronous :class:`Worker` instead so their output is
    deterministic.
    """

    def __init__(
        self,
        queue: PriorityQueue,
        store: JobStore,
        size: int = 4,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        self.queue = queue
        self.store = store
        self.size = size
        self.retry_policy = retry_policy or RetryPolicy()
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("WorkerPool already started")
        self._stop.clear()
        for i in range(self.size):
            t = threading.Thread(target=self._loop, name=f"queueworks-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def _loop(self) -> None:
        worker = Worker(self.queue, self.store, self.retry_policy)
        while not self._stop.is_set():
            with self._lock:
                job = self.queue.pop()
            if job is None:
                time.sleep(0.01)
                continue
            worker._run_with_retries(job)

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        if wait:
            for t in self._threads:
                t.join(timeout=5)
        self._threads = []

    def __enter__(self) -> "WorkerPool":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
