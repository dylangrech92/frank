"""A stand-in for an HTTP webhook sink.

Real network calls are out of scope for this project, so this writes
each delivery to a local outbox file in the same shape a webhook POST
body would use, instead of making a request.
"""
from __future__ import annotations

import json

from relay.compat.legacy import sanitize_order
from relay.errors import DeliveryError
from relay.events import EVENT_ORDER_DELIVERED

OUTBOX_PATH = "webhook_outbox.jsonl"


def send_to_webhook_sink(event) -> None:
    """Append `event` to the local webhook outbox. Raises
    `DeliveryError` if the write fails.

    External consumers of the webhook feed expect the same trimmed,
    coerced fields as the file sink, so the payload is put through the
    same normalization pass first.
    """
    sanitize_order(event)
    body = {
        "event": EVENT_ORDER_DELIVERED,
        "event_id": event.event_id,
        "payload": event.payload,
    }
    try:
        with open(OUTBOX_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(body) + "\n")
    except OSError as exc:
        raise DeliveryError(f"could not write {event.event_id} to webhook outbox: {exc}") from exc
