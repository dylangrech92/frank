"""Assembles a human-readable operator report from the pipeline's
various counters and stores.

This module only reads from `state`, `metrics`, `deadletter`, and
`scheduler` - it never mutates the pipeline, so building a report is
always safe to call mid-run.
"""
from __future__ import annotations

from relay.pipeline.metrics import metrics
from relay.pipeline.scheduler import scheduler
from relay.sinks import deadletter
from relay.storage.state import StateStore
from relay.utils.formatting import format_count, format_duration_s, format_percent


def build_report(state: StateStore) -> str:
    """Render a multi-line operator report covering throughput, the
    slowest stages, and outstanding retries."""
    counters = state.counters()
    delivered = counters["delivered"]
    dead_lettered = counters["dead_lettered"]
    retried = counters["retried"]
    total = delivered + dead_lettered

    lines = ["relay operator report", "=" * 22]
    lines.append(f"delivered:      {format_count(delivered, 'event')}")
    lines.append(f"dead-lettered:  {format_count(dead_lettered, 'event')}")
    lines.append(f"retried:        {format_count(retried, 'attempt')}")
    lines.append(f"delivery rate:  {format_percent(delivered, total)}")
    lines.append(f"dead-letter file backlog: {format_count(deadletter.count_dead_letters(), 'record')}")
    lines.append(f"pending retries: {scheduler.pending_count()}")

    slowest = metrics.slowest_stages(limit=3)
    if slowest:
        lines.append("")
        lines.append("slowest stages (mean duration):")
        for name, mean_seconds in slowest:
            lines.append(f"  {name}: {format_duration_s(mean_seconds)}")

    history = state.recent_history()
    if history:
        lines.append("")
        lines.append(f"most recently delivered ({len(history)}):")
        for event_id in history[-5:]:
            lines.append(f"  {event_id}")

    return "\n".join(lines)
