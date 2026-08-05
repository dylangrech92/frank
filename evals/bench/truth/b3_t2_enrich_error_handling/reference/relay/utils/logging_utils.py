"""Central logging configuration for the demo process."""
from __future__ import annotations

import logging

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO, fmt: str = _DEFAULT_FORMAT) -> None:
    """Configure the root logger once per process. Safe to call more
    than once - later calls after the first are a no-op, so importing
    modules don't need to know whether logging has already been set up
    by whoever started the process."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(level=level, format=fmt)
    _CONFIGURED = True


def is_configured() -> bool:
    return _CONFIGURED
