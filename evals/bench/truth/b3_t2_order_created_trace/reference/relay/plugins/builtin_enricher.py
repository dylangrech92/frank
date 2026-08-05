"""The default enrichment plugin: adds a shipping region derived from a
postal code, and a flat handling fee based on order size."""
from __future__ import annotations

from relay.errors import EnrichmentError
from relay.handlers.transform import normalize_event

_POSTAL_REGIONS = {
    "9": "west",
    "8": "west",
    "7": "central",
    "6": "central",
    "5": "central",
    "4": "east",
    "3": "east",
    "2": "east",
    "1": "east",
    "0": "east",
}

_SMALL_ORDER_FEE_CENTS = 150
_LARGE_ORDER_QTY_THRESHOLD = 20


def region_for_postal_code(postal_code: str) -> str:
    if not postal_code:
        return "unknown"
    return _POSTAL_REGIONS.get(postal_code[0], "unknown")


def handling_fee_cents(qty: int) -> int:
    """Small orders carry a flat handling fee; large orders are assumed
    to already justify their own freight arrangement."""
    if qty >= _LARGE_ORDER_QTY_THRESHOLD:
        return 0
    return _SMALL_ORDER_FEE_CENTS


def enrich(event) -> None:
    normalize_event(event)
    postal = event.payload.get("postal_code")
    if not postal:
        raise EnrichmentError(f"order {event.event_id} has no postal_code to enrich from")
    event.payload["region"] = region_for_postal_code(postal)
    event.payload["handling_fee_cents"] = handling_fee_cents(event.payload.get("qty", 0))


class _BuiltinEnricherPlugin:
    enrich = staticmethod(enrich)


PLUGIN = _BuiltinEnricherPlugin()
