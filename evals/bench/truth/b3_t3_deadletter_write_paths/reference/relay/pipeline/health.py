"""A read-only health check: compares the pipeline's current backlog
against the thresholds in `config` and reports which, if any, are
breached. Never raises and never mutates anything - a breach is only
ever reported, not acted on.
"""
from __future__ import annotations

from dataclasses import dataclass

from relay import config
from relay.pipeline.scheduler import scheduler
from relay.sinks import deadletter


@dataclass
class HealthCheck:
    name: str
    value: int
    threshold: int

    @property
    def breached(self) -> bool:
        return self.value > self.threshold

    def __str__(self) -> str:
        status = "BREACH" if self.breached else "ok"
        return f"[{status}] {self.name}: {self.value} (threshold {self.threshold})"


def run_health_checks() -> list[HealthCheck]:
    """Return one `HealthCheck` per monitored backlog."""
    return [
        HealthCheck(
            name="dead_letter_backlog",
            value=deadletter.count_dead_letters(),
            threshold=config.HEALTH_MAX_DEAD_LETTER_BACKLOG,
        ),
        HealthCheck(
            name="pending_retries",
            value=scheduler.pending_count(),
            threshold=config.HEALTH_MAX_PENDING_RETRIES,
        ),
    ]


def is_healthy() -> bool:
    """True if every monitored backlog is within its threshold."""
    return not any(check.breached for check in run_health_checks())
