"""Per-account running-balance statements.

A statement is the classic bank-statement view: every transaction for one
account, in chronological order, with a running balance next to each row
so a reader can see the balance evolve line by line rather than only at
the end (which is all `aggregator.reconcile` reports).
"""
from __future__ import annotations

from .models import Account, Transaction


def account_statement_rows(transactions: list[Transaction], account: Account) -> list[dict]:
    """Build statement rows for one account's transactions.

    `transactions` should already be filtered to the one account; this
    function only orders and running-totals them. `sorted()` is used
    rather than `list.sort()` so the caller's own list (and whatever else
    holds a reference to it) is never mutated as a side effect of
    building a statement.

    Args:
        transactions: this account's transactions, any order, mixed
            categories.
        account: the owning account, used for its opening balance.

    Returns:
        One dict per transaction, oldest first, each carrying the running
        balance immediately after that transaction is applied, plus a
        synthetic leading row for the opening balance itself.
    """
    ordered = sorted(transactions, key=lambda t: t.timestamp_utc)
    rows = [{
        "date": None,
        "txn_id": None,
        "description": "opening balance",
        "amount": None,
        "running_balance": round(account.opening_balance, 2),
    }]
    running = account.opening_balance
    for txn in ordered:
        running += txn.amount
        rows.append({
            "date": txn.timestamp_utc.date().isoformat(),
            "txn_id": txn.txn_id,
            "description": txn.description or txn.category,
            "amount": round(txn.amount, 2),
            "running_balance": round(running, 2),
        })
    return rows
