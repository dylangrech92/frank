## Diagnosis

Root cause: `queueworks/store.py`'s `JobStore._load()` resumes `_seq_counter`
by counting how many jobs are currently in the store
(`self._seq_counter = len(self._jobs)`) instead of resuming from the highest
`seq` value actually present. After a job is processed and
`purge_completed()` removes it, the store's job count drops -- so the next
time a process starts up and calls `_load()`, `_seq_counter` gets
reinitialized too low. The next `enqueue()` call then hands out a `seq` that
duplicates one still held by a job left `PENDING` in the store.

When two jobs end up sharing the same `(priority, seq)` pair, the priority
queue's heap can end up with both `Job` objects as heap siblings.
`heapq.heappush` doesn't always need to compare them directly (its sift-up
only walks ancestors), so the collision can silently succeed on push -- but
`heapq.heappop`'s sift-down does compare sibling children directly, and
because `Job` has no ordering defined, that pop raises `TypeError: '<' not
supported between instances of 'Job' and 'Job'`. That's why the traceback
points into `queue.py`'s `pop()` even though the actual defect is the seq
counter resume logic in `store.py`.

## Fix

Resume `_seq_counter` from the maximum `seq` still present in the loaded
jobs (falling back to 0 when the store is empty), e.g.
`max((job.seq for job in self._jobs.values()), default=0)`, so newly
enqueued jobs never reuse a seq value already held by a job still sitting
in the store.
