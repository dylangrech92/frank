"""Event types and the canonical event-name constants used across the
pipeline, the bus, and the compat layer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

EVENT_ORDER_CREATED = "order.created"
EVENT_ORDER_VALIDATED = "order.validated"
EVENT_ORDER_ENRICHED = "order.enriched"
EVENT_ORDER_DELIVERED = "order.delivered"

ALL_EVENT_NAMES = (
    EVENT_ORDER_CREATED,
    EVENT_ORDER_VALIDATED,
    EVENT_ORDER_ENRICHED,
    EVENT_ORDER_DELIVERED,
)


@dataclass
class Event:
    """A single order event moving through the pipeline."""

    event_id: str
    name: str
    payload: dict
    created_at: float = field(default_factory=time.time)
    attempts: int = 0

    def bump_attempt(self) -> None:
        self.attempts += 1

    def age_s(self, now: float | None = None) -> float:
        """Seconds since this event was created."""
        now = time.time() if now is None else now
        return max(now - self.created_at, 0.0)

    def to_dict(self) -> dict:
        """A JSON-serializable view of this event, used by the sinks."""
        return {
            "event_id": self.event_id,
            "name": self.name,
            "payload": self.payload,
            "created_at": self.created_at,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """Reconstruct an `Event` from `to_dict`'s output - used when
        replaying dead-lettered events back through the pipeline."""
        return cls(
            event_id=data["event_id"],
            name=data.get("name", EVENT_ORDER_CREATED),
            payload=dict(data["payload"]),
            created_at=data.get("created_at", time.time()),
            attempts=data.get("attempts", 0),
        )
