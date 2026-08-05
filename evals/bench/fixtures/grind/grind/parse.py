"""Turn raw generated events into validated, canonicalized records."""

from __future__ import annotations

import json

from .normalize import canonicalize_path
from .records import ParsedEvent, RawEvent
from .validators import validate_event


def parse_event(raw: RawEvent) -> ParsedEvent | None:
    if not validate_event(raw):
        return None

    canonical, shape, signature = canonicalize_path(raw.path)
    try:
        metadata = json.loads(raw.metadata_json)
    except json.JSONDecodeError:
        metadata = {}
    metadata["_shape"] = shape
    metadata["_signature"] = signature

    return ParsedEvent(
        record_id=raw.record_id,
        session_id=raw.session_id,
        ts=raw.ts,
        method=raw.method,
        path=raw.path,
        canonical_path=canonical,
        referrer=raw.referrer,
        status=raw.status,
        bytes_sent=raw.bytes_sent,
        region=raw.region,
        user_agent=raw.user_agent,
        metadata_json=raw.metadata_json,
        metadata=metadata,
    )


def parse_events(raw_events: list[RawEvent]) -> tuple[list[ParsedEvent], int]:
    parsed: list[ParsedEvent] = []
    rejected = 0
    for raw in raw_events:
        event = parse_event(raw)
        if event is None:
            rejected += 1
        else:
            parsed.append(event)
    return parsed, rejected
