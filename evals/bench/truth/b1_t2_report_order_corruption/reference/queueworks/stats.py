"""Summary statistics over recently completed jobs.

These helpers read from :func:`queueworks.store.get_completion_log`, the
process-wide feed of recently finished jobs, rather than re-scanning the
persistent store on every call.
"""

from __future__ import annotations

from typing import List

from .models import CompletionRecord
from .store import get_completion_log


def slowest_jobs(n: int = 5) -> List[CompletionRecord]:
    """Return the ``n`` slowest recently-completed jobs, slowest first."""

    records = sorted(get_completion_log(), key=lambda r: r.duration, reverse=True)
    return records[:n]


def success_rate() -> float:
    """Fraction of recently-completed jobs that finished successfully."""

    records = get_completion_log()
    if not records:
        return 1.0
    successes = sum(1 for r in records if r.status == "completed")
    return successes / len(records)


def average_duration() -> float:
    """Mean duration, in seconds, across recently-completed jobs."""

    records = get_completion_log()
    if not records:
        return 0.0
    return sum(r.duration for r in records) / len(records)
