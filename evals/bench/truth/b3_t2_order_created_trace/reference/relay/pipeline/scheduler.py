"""A minimal in-process scheduler for retried events.

`schedule_retry` stores a callback alongside the event it belongs to and
returns immediately; the callback runs later, from `drain_ready`, whenever
something chooses to call it. `expire_stale` is the scheduler's other exit
door: a retry that has been waiting too long for its turn is given up on
directly, without ever running its callback.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from relay.errors import RetryExpiredError
from relay.sinks import deadletter, log_sink

logger = logging.getLogger(__name__)


@dataclass
class _ScheduledRetry:
    event: object
    run_again: Callable[[object], object]
    ready_at: float
    scheduled_at: float


class Scheduler:
    def __init__(self) -> None:
        self._pending: list[_ScheduledRetry] = []

    def schedule_retry(self, event, run_again: Callable[[object], object], delay_s: float) -> None:
        """Queue `run_again(event)` to run once `delay_s` has elapsed."""
        now = time.time()
        self._pending.append(
            _ScheduledRetry(event=event, run_again=run_again, ready_at=now + delay_s, scheduled_at=now)
        )
        log_sink.log_retry_scheduled(event, delay_s)

    def pending_count(self) -> int:
        return len(self._pending)

    def drain_ready(self, now: float | None = None) -> int:
        """Invoke every scheduled callback whose delay has elapsed.

        Returns the number of callbacks invoked. A callback failing
        unexpectedly is logged and skipped rather than allowed to stop
        the rest of the batch from draining - `run_again` owns deciding
        what a failure of its own event means (retry again, dead-letter,
        or something else) and normally handles that itself.
        """
        now = time.time() if now is None else now
        ready: list[_ScheduledRetry] = []
        still_pending: list[_ScheduledRetry] = []
        for item in self._pending:
            (ready if item.ready_at <= now else still_pending).append(item)
        self._pending = still_pending

        for item in ready:
            try:
                item.run_again(item.event)
            except Exception:  # noqa: BLE001 - one bad scheduled retry shouldn't block its siblings
                logger.exception("scheduled retry callback failed for %s", item.event.event_id)
        return len(ready)

    def expire_stale(self, max_pending_s: float, now: float | None = None) -> int:
        """Give up on any retry that has been sitting in the queue for
        longer than `max_pending_s`, whether or not its delay has
        elapsed yet. Each one is dead-lettered directly, without ever
        running its callback. Returns the number expired.
        """
        now = time.time() if now is None else now
        cutoff = now - max_pending_s
        expiring: list[_ScheduledRetry] = []
        still_pending: list[_ScheduledRetry] = []
        for item in self._pending:
            (expiring if item.scheduled_at <= cutoff else still_pending).append(item)
        self._pending = still_pending

        for item in expiring:
            error = RetryExpiredError(
                f"retry for {item.event.event_id} expired after {max_pending_s}s pending"
            )
            deadletter.write_dead_letter(item.event, error)
        return len(expiring)


scheduler = Scheduler()
