"""Human-readable operational logging for the pipeline."""
from __future__ import annotations

import logging

logger = logging.getLogger("relay.pipeline")


def log_delivered(event) -> None:
    logger.info("delivered %s (%d attempt(s))", event.event_id, event.attempts + 1)


def log_dead_lettered(event, error: Exception) -> None:
    """Emit a warning log line for an event that has been dead-lettered."""
    logger.warning("event %s dead-lettered: %s: %s", event.event_id, type(error).__name__, error)


def log_retry_scheduled(event, delay_s: float) -> None:
    logger.info("retry %d scheduled for %s in %.2fs", event.attempts, event.event_id, delay_s)
