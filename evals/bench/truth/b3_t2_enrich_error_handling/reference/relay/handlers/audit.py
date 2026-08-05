"""Audit-log handlers for the order lifecycle.

`record_validation` and `record_enrichment` don't change the event; they
exist purely to leave a structured trail of what happened, at the two
stage boundaries that aren't already covered by
`transform.normalize_event`. `flag_large_order` is the exception: it can
raise, on purpose, for orders large enough to need a compliance
sign-off before they're treated as routine.
"""
from __future__ import annotations

import logging

from relay.bus import bus
from relay.events import EVENT_ORDER_ENRICHED, EVENT_ORDER_VALIDATED

_audit_log = logging.getLogger("relay.audit")

_LARGE_ORDER_QTY_ALERT_THRESHOLD = 250


@bus.on(EVENT_ORDER_VALIDATED)
def record_validation(event) -> None:
    """Audit-log every order that passes validation."""
    _audit_log.info("order %s validated (sku=%s)", event.event_id, event.payload.get("sku"))


@bus.on(EVENT_ORDER_ENRICHED)
def record_enrichment(event) -> None:
    """Audit-log the region an order was enriched with."""
    _audit_log.info(
        "order %s enriched (region=%s)", event.event_id, event.payload.get("region", "unknown")
    )


@bus.on(EVENT_ORDER_ENRICHED)
def flag_large_order(event) -> None:
    """Raise a plain (non-`RelayError`) alert for orders large enough to
    need manual compliance sign-off, registered on the bus the same way
    as the rest of this module's handlers.

    Because this runs as a bus-published handler rather than as a stage
    function called directly by the orchestrator, a raised exception
    here is caught by `EventBus.publish` itself and never reaches
    `run_pipeline`'s `except RelayError` block at all -- on top of not
    being a `RelayError` in the first place.
    """
    qty = event.payload.get("qty", 0)
    if qty >= _LARGE_ORDER_QTY_ALERT_THRESHOLD:
        raise ValueError(f"order {event.event_id} needs manual compliance sign-off (qty={qty})")
