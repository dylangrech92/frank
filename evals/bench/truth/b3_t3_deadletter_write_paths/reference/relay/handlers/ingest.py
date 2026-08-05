"""Turns raw external payloads into `Event` objects.

This is the only module that constructs an `Event` with a freshly
generated id from raw input; every other stage receives an event that
already exists.
"""
from __future__ import annotations

from relay.events import Event, EVENT_ORDER_CREATED
from relay.utils.ids import generate_event_id


def build_order_event(raw: dict) -> Event:
    """Wrap a raw dict, as received from an external producer, in an
    `Event` with a freshly generated id."""
    return Event(event_id=generate_event_id(), name=EVENT_ORDER_CREATED, payload=dict(raw))
