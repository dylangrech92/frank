"""Record shapes passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawEvent:
    """A single synthetic access-log line, before parsing."""

    record_id: str
    session_id: str
    ts: str
    method: str
    path: str
    referrer: str
    status: int
    bytes_sent: int
    region: str
    user_agent: str
    metadata_json: str


@dataclass
class ParsedEvent:
    """A validated, normalized event ready for enrichment/aggregation."""

    record_id: str
    session_id: str
    ts: str
    method: str
    path: str
    canonical_path: str
    referrer: str
    status: int
    bytes_sent: int
    region: str
    user_agent: str
    metadata_json: str = ""
    metadata: dict = field(default_factory=dict)
    is_duplicate: bool = False
