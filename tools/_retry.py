"""Shared one-shot retry helper for browser actions.

Playwright auto-waits internally (visibility, actionability, network idle where
relevant); this adds exactly one deterministic retry with a fixed backoff on top
of that, for the rare case where auto-waiting still times out on a slow render.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

T = TypeVar("T")

_BACKOFF_SECONDS = 0.5


def retry_once_on_timeout(fn: Callable[[], T]) -> T:
    """Call ``fn()``; on a single ``TimeoutError``, sleep and retry exactly once.

    Any other exception, and a second ``TimeoutError``, propagate to the caller.
    """
    try:
        return fn()
    except PlaywrightTimeoutError:
        time.sleep(_BACKOFF_SECONDS)
        return fn()
