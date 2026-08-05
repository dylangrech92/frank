"""An in-memory priority queue of jobs, ordered by (priority, seq).

The queue itself holds no persistence logic -- it is a thin ``heapq``
wrapper that a :class:`~queueworks.manager.QueueManager` refills from the
:class:`~queueworks.store.JobStore` at startup and keeps in sync as jobs are
enqueued and dequeued during a run.
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

from .models import Job, Priority


class PriorityQueue:
    """A min-heap of jobs ordered by ``(priority, seq)``.

    ``seq`` is assigned once per job by :meth:`JobStore.next_seq` at enqueue
    time and exists purely to keep jobs of equal priority in FIFO order --
    it is expected to be unique across every job the queue ever sees.
    """

    def __init__(self):
        self._heap: List[Tuple[int, int, Job]] = []

    def push(self, job: Job) -> None:
        heapq.heappush(self._heap, (int(job.priority), job.seq, job))

    def pop(self) -> Optional[Job]:
        if not self._heap:
            return None
        _, _, job = heapq.heappop(self._heap)
        return job

    def peek(self) -> Optional[Job]:
        if not self._heap:
            return None
        return self._heap[0][2]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def drain_in_order(self) -> List[Job]:
        """Pop every job off the queue and return them in priority order."""

        drained = []
        while self._heap:
            drained.append(self.pop())
        return drained
