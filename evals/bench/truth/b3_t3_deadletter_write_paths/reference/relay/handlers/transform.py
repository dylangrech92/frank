"""Payload normalization for order events."""
from __future__ import annotations

from relay.bus import bus
from relay.events import EVENT_ORDER_CREATED


@bus.on(EVENT_ORDER_CREATED)
def normalize_event(event) -> None:
    """Normalize field casing and whitespace so downstream stages can
    rely on a consistent payload shape. Mutates `event.payload` in
    place."""
    payload = event.payload
    if "sku" in payload and isinstance(payload["sku"], str):
        payload["sku"] = payload["sku"].strip().upper()
    if "customer_id" in payload and isinstance(payload["customer_id"], str):
        payload["customer_id"] = payload["customer_id"].strip()
    if "qty" in payload:
        try:
            payload["qty"] = int(payload["qty"])
        except (TypeError, ValueError):
            payload["qty"] = 0
