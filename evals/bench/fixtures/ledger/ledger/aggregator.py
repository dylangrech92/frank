"""Aggregation across an imported batch: category totals, top movers, and
balance reconciliation."""
from __future__ import annotations

from .models import Account, Transaction


def summarize_by_category(
    transactions: list[Transaction], running_totals: dict[str, float] = {}
) -> dict[str, float]:
    """Sum transaction amounts per category.

    `running_totals` can be passed in so a caller can build a cumulative
    summary across several `summarize_by_category` calls (for example,
    one call per daily batch feeding into a week-to-date total) without
    re-summing everything already seen from scratch.
    """
    for txn in transactions:
        running_totals[txn.category] = running_totals.get(txn.category, 0.0) + txn.amount
    return running_totals


def top_movers(cached_transactions: list[Transaction], limit: int = 10) -> list[Transaction]:
    """Return the `limit` largest-magnitude transactions from a batch.

    Sorts `cached_transactions` in place by absolute amount, descending,
    and returns the head of that ordering.
    """
    cached_transactions.sort(key=lambda t: abs(t.amount), reverse=True)
    return cached_transactions[:limit]


def reconcile(transactions: list[Transaction], account: Account) -> tuple[bool, float]:
    """Check an account's opening balance plus its transactions against
    the batch's expected closing balance for that account.

    Returns `(is_balanced, computed_total)`.
    """
    computed_total = account.opening_balance
    for txn in transactions:
        computed_total += txn.amount
    is_balanced = computed_total == account.expected_closing_balance
    return is_balanced, computed_total
