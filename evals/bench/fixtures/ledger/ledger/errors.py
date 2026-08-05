"""Shared error helpers.

`safe_call` centralizes the "log with context before it propagates"
behavior used by the CLI commands in `main.py`: an exception from deep
inside the import/aggregate/report pipeline should be logged with which
step was running before it reaches the top level, where a bare traceback
alone doesn't say which command or stage failed.
"""
from __future__ import annotations

import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LedgerError(Exception):
    """Base class for ledger-specific errors."""


class BatchImportError(LedgerError):
    """Raised when a batch cannot be imported at all (not just a bad row)."""


def safe_call(fn: Callable[..., T], *args, **kwargs) -> T:
    """Call `fn`, logging any exception with context before it propagates.

    This does not suppress the error: the caller still sees it, and
    `main.py`'s top-level handler still turns it into a non-zero exit
    code. It exists purely to attach which step of the pipeline was
    running when things broke to the log record.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("failed during %s", getattr(fn, "__name__", repr(fn)))
        raise
