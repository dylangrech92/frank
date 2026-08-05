"""Schema validation for order events.

Validation is split into small field-level checks so a bad order can be
rejected with every problem it has, not just the first one found.
"""
from __future__ import annotations

import re

from relay import config
from relay.errors import ValidationError

_SKU_RE = re.compile(config.SKU_PATTERN)


def _check_required_fields(payload: dict) -> list[str]:
    return [f for f in config.REQUIRED_ORDER_FIELDS if not payload.get(f)]


def _check_qty(payload: dict) -> str | None:
    qty = payload.get("qty")
    if qty is None:
        return None
    if qty <= 0:
        return "qty must be positive"
    if qty > config.MAX_QTY_PER_ORDER:
        return f"qty exceeds MAX_QTY_PER_ORDER ({config.MAX_QTY_PER_ORDER})"
    return None


def _check_sku_shape(payload: dict) -> str | None:
    sku = payload.get("sku")
    if not sku:
        return None
    if not _SKU_RE.match(sku):
        return f"sku {sku!r} does not match the expected shape"
    return None


def validate_order(event) -> None:
    """Raise `ValidationError` describing every problem found with
    `event.payload`, or return normally if the order is well-formed."""
    payload = event.payload
    problems: list[str] = []

    missing = _check_required_fields(payload)
    if missing:
        problems.append(f"missing fields: {', '.join(missing)}")

    qty_problem = _check_qty(payload)
    if qty_problem:
        problems.append(qty_problem)

    sku_problem = _check_sku_shape(payload)
    if sku_problem:
        problems.append(sku_problem)

    if problems:
        raise ValidationError(f"order {event.event_id}: {'; '.join(problems)}")
