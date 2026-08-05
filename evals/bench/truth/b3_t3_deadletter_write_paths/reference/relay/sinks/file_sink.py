"""Writes successfully delivered events to the output file as JSON
lines."""
from __future__ import annotations

import json

from relay import config
from relay.errors import DeliveryError
from relay.events import EVENT_ORDER_DELIVERED


def send_to_file_sink(event) -> None:
    """Append `event` to `config.OUTPUT_PATH`. Raises `DeliveryError` if
    the write fails."""
    record = {
        "event_id": event.event_id,
        "name": EVENT_ORDER_DELIVERED,
        "payload": event.payload,
        "attempts": event.attempts,
    }
    try:
        with open(config.OUTPUT_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        raise DeliveryError(f"could not write {event.event_id} to output: {exc}") from exc
