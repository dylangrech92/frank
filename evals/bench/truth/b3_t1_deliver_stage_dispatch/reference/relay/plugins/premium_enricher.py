"""An enrichment plugin for customers on the premium tier: everything
`builtin_enricher` does, plus a waived handling fee and an expedited
shipping flag.

Not selected by `config.ENRICHMENT_PLUGIN` by default; a deployment
opts in by pointing that setting at `"premium_enricher"` instead of
`"builtin_enricher"`.
"""
from __future__ import annotations

from relay.errors import EnrichmentError
from relay.plugins.builtin_enricher import region_for_postal_code

PREMIUM_CUSTOMER_PREFIX = "cust-9"


def is_premium_customer(customer_id: str) -> bool:
    return bool(customer_id) and customer_id.startswith(PREMIUM_CUSTOMER_PREFIX)


def enrich(event) -> None:
    from relay.handlers.transform import normalize_event

    normalize_event(event)
    postal = event.payload.get("postal_code")
    if not postal:
        raise EnrichmentError(f"order {event.event_id} has no postal_code to enrich from")
    event.payload["region"] = region_for_postal_code(postal)
    event.payload["handling_fee_cents"] = 0
    event.payload["expedited"] = is_premium_customer(event.payload.get("customer_id", ""))


class _PremiumEnricherPlugin:
    enrich = staticmethod(enrich)


PLUGIN = _PremiumEnricherPlugin()
