"""Retry/duplicate detection.

A client retry lands as a new event id with a fresh timestamp but otherwise
identical request details. Events are compared on everything except the id
and timestamp; anything matching an event already seen earlier in the batch
is flagged rather than dropped, so the aggregate report can still account
for retried bytes without double-counting unique requests.
"""

from __future__ import annotations

from .records import ParsedEvent


def _dedup_key(event: ParsedEvent) -> tuple:
    return (event.session_id, event.canonical_path, event.status, event.bytes_sent)


def mark_duplicates(events: list[ParsedEvent]) -> int:
    """Flag retried events in place. Returns the number flagged."""
    seen: list[tuple] = []
    duplicate_count = 0
    for event in events:
        key = _dedup_key(event)
        if key in seen:
            event.is_duplicate = True
            duplicate_count += 1
        else:
            seen.append(key)
    return duplicate_count
