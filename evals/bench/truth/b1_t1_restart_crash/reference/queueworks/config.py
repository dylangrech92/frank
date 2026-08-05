"""Runtime configuration for queueworks, loaded from the environment.

Applications that don't need custom wiring can build a :class:`QueueConfig`
from the process environment and derive both a state-file path and a
:class:`~queueworks.retry.RetryPolicy` from it, instead of hand-assembling
those pieces at every call site.
"""

from __future__ import annotations

import dataclasses
import os

from .retry import RetryPolicy

_ENV_PREFIX = "QUEUEWORKS_"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX}{name}={raw!r} is not a valid float") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX}{name}={raw!r} is not a valid int") from exc


@dataclasses.dataclass(frozen=True)
class QueueConfig:
    """Everything a process needs to point at a queue and its retry policy.

    Every field has a sane default so ``QueueConfig()`` works out of the box
    for local development; production deployments typically override
    ``state_path`` and leave the retry tuning alone unless they have
    measured a reason to change it.
    """

    state_path: str = "queueworks_state.json"
    max_retries: int = 3
    base_delay: float = 0.05
    backoff_factor: float = 4.0
    max_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "QueueConfig":
        """Build a :class:`QueueConfig` from ``QUEUEWORKS_*`` environment
        variables, falling back to the dataclass defaults for anything left
        unset."""

        defaults = cls()
        return cls(
            state_path=os.environ.get(_ENV_PREFIX + "STATE_PATH", defaults.state_path),
            max_retries=_env_int("MAX_RETRIES", defaults.max_retries),
            base_delay=_env_float("BASE_DELAY", defaults.base_delay),
            backoff_factor=_env_float("BACKOFF_FACTOR", defaults.backoff_factor),
            max_delay=_env_float("MAX_DELAY", defaults.max_delay),
        )

    def retry_policy(self) -> RetryPolicy:
        """Build the :class:`~queueworks.retry.RetryPolicy` this config describes."""

        return RetryPolicy(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            backoff_factor=self.backoff_factor,
            max_delay=self.max_delay,
        )
