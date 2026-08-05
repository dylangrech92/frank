"""Human-readable reports over recently completed jobs.

Unlike :mod:`queueworks.stats`, which distills the recent-completions feed
down to single numbers, this module renders the feed itself for a human to
read -- most notably :func:`chronological_report`, which operations staff
use to see what the queue has been doing in the order it actually happened.
"""

from __future__ import annotations

from typing import List

from .models import CompletionRecord
from .store import get_completion_log


def chronological_report() -> List[str]:
    """Render recently-completed jobs in true completion order, oldest first.

    Returns one formatted line per job; callers that just want the ordering
    for their own checks can read ``.job_id`` off
    :func:`queueworks.store.get_completion_log` directly instead.
    """

    records = get_completion_log()
    lines = []
    for r in records:
        lines.append(
            f"{r.finished_at:.3f}  {r.job_id:<20} {r.task_name:<20} "
            f"{r.status:<10} {r.duration:.3f}s"
        )
    return lines


def summary_report() -> List[str]:
    """Render a short human-readable summary of recent completions."""

    records = get_completion_log()
    total = len(records)
    failed = sum(1 for r in records if r.status == "failed")
    lines = [f"{total} jobs completed recently, {failed} failed."]
    for r in records:
        lines.append(f"  - {r.job_id}: {r.task_name} -> {r.status}")
    return lines
