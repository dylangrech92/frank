"""Structural validation for parsed events.

Each field has its own small grammar; malformed records are dropped before
they reach normalization so a bad upstream batch can't poison the report.
"""

from __future__ import annotations

import re

from .records import RawEvent

_SESSION_ID_TEMPLATE = r"^sess-[0-9a-f]{8}$"
_RECORD_ID_TEMPLATE = r"^evt-[0-9]{7}$"
_REFERRER_TEMPLATE = r"^(https?://[a-z0-9.\-]+/[a-z0-9/_.\-]*|-)$"
_PATH_TEMPLATE = r"^/[a-z0-9%._\-]+/[a-z0-9%._\-]+(\?[a-zA-Z0-9=&%_\-]*)?$"


def valid_session_id(session_id: str) -> bool:
    pattern = re.compile(_SESSION_ID_TEMPLATE)
    return bool(pattern.match(session_id))


def valid_record_id(record_id: str) -> bool:
    pattern = re.compile(_RECORD_ID_TEMPLATE)
    return bool(pattern.match(record_id))


def valid_referrer(referrer: str) -> bool:
    pattern = re.compile(_REFERRER_TEMPLATE, re.IGNORECASE)
    return bool(pattern.match(referrer))


def valid_path(path: str) -> bool:
    pattern = re.compile(_PATH_TEMPLATE, re.IGNORECASE)
    return bool(pattern.match(path))


def validate_event(event: RawEvent) -> bool:
    """Return True if every field on the event passes its grammar check."""
    return (
        valid_record_id(event.record_id)
        and valid_session_id(event.session_id)
        and valid_referrer(event.referrer)
        and valid_path(event.path)
        and event.status in (200, 301, 302, 303, 304, 404, 429, 500, 502, 503)
        and event.bytes_sent >= 0
    )
