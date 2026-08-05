"""An enrichment plugin for bulk orders: everything `builtin_enricher`
does, plus a volume discount applied to the handling fee once an order's
quantity clears a threshold.

Not selected by `config.ENRICHMENT_PLUGIN` by default; a deployment opts
in by pointing that setting at `"discount_enricher"` instead of
`"builtin_enricher"`.
"""
from __future__ import annotations

from relay.errors import EnrichmentError
from relay.plugins.builtin_enricher import handling_fee_cents, region_for_postal_code

BULK_QTY_THRESHOLD = 10
BULK_DISCOUNT_PERCENT = 25


def discounted_fee_cents(qty: int) -> int:
    """The handling fee `builtin_enricher` would charge, reduced by
    `BULK_DISCOUNT_PERCENT` once `qty` clears `BULK_QTY_THRESHOLD`."""
    base_fee = handling_fee_cents(qty)
    if qty < BULK_QTY_THRESHOLD:
        return base_fee
    return base_fee - (base_fee * BULK_DISCOUNT_PERCENT // 100)


def enrich(event) -> None:
    from relay.handlers.transform import normalize_event

    normalize_event(event)
    postal = event.payload.get("postal_code")
    if not postal:
        raise EnrichmentError(f"order {event.event_id} has no postal_code to enrich from")
    qty = event.payload.get("qty", 0)
    event.payload["region"] = region_for_postal_code(postal)
    event.payload["handling_fee_cents"] = discounted_fee_cents(qty)
    event.payload["bulk_discount_applied"] = qty >= BULK_QTY_THRESHOLD


class _DiscountEnricherPlugin:
    enrich = staticmethod(enrich)


PLUGIN = _DiscountEnricherPlugin()
