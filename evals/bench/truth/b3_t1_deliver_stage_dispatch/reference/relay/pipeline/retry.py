"""Retry policy: decides whether a stage failure should be retried."""
from __future__ import annotations

from relay import config
from relay.errors import DeliveryError, PluginLoadError, RelayError

# Errors judged transient enough to retry. Validation and enrichment
# failures observed in production are shaped by the payload itself, so
# retrying them unchanged just fails again the same way.
RETRYABLE_EXCEPTIONS: tuple[type, ...] = (PluginLoadError, DeliveryError)


def should_retry(exc: RelayError, attempt: int) -> bool:
    """Return True if `exc`, raised on attempt number `attempt`
    (0-indexed), should be retried."""
    if not config.RETRY_ENABLED:
        return False
    if attempt >= config.MAX_RETRIES:
        return False
    return isinstance(exc, RETRYABLE_EXCEPTIONS)


def backoff_for(attempt: int) -> float:
    """Exponential backoff in seconds for the given (0-indexed) attempt,
    capped at `config.RETRY_BACKOFF_MAX_S`."""
    return min(config.RETRY_BACKOFF_BASE_S * (2**attempt), config.RETRY_BACKOFF_MAX_S)


def attempts_remaining(attempt: int) -> int:
    """How many more retries are available after attempt number
    `attempt`, ignoring `config.RETRY_ENABLED`."""
    return max(config.MAX_RETRIES - attempt, 0)
