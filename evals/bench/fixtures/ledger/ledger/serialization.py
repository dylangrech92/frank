"""JSON export for an imported batch.

Downstream systems (a data warehouse load, a support ticket attachment, an
archival copy) want a batch as a single JSON document rather than the two
source CSVs plus whatever the CLI printed to stdout. This module is the
one place that knows how to turn the in-memory `Batch` model into that
document.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Account, Batch, RowError, Transaction


def account_to_dict(account: Account) -> dict[str, Any]:
    return {
        "account_id": account.account_id,
        "name": account.name,
        "currency": account.currency,
        "tz_name": account.tz_name,
        "opening_balance": account.opening_balance,
        "expected_closing_balance": account.expected_closing_balance,
        "utc_offset_minutes": account.utc_offset_minutes,
    }


def transaction_to_dict(txn: Transaction) -> dict[str, Any]:
    return {
        "txn_id": txn.txn_id,
        "account_id": txn.account_id,
        "timestamp_utc": txn.timestamp_utc.isoformat(),
        "amount": txn.amount,
        "category": txn.category,
        "description": txn.description,
    }


def row_error_to_dict(error: RowError) -> dict[str, Any]:
    return {
        "source_file": error.source_file,
        "line_number": error.line_number,
        "reason": error.reason,
    }


def batch_to_dict(batch: Batch) -> dict[str, Any]:
    """Serialize a whole batch to a plain-dict document.

    The three sections mirror `Batch`'s own fields exactly: every account,
    every transaction, and every rejected row, each through its own
    `*_to_dict` helper above.
    """
    return {
        "accounts": [account_to_dict(a) for a in batch.accounts],
        "transactions": [transaction_to_dict(t) for t in batch.transactions],
        "errors": [row_error_to_dict(e) for e in batch.errors],
    }


def write_batch_json(batch: Batch, path: str | Path) -> None:
    """Write `batch` to `path` as a single formatted JSON document."""
    Path(path).write_text(json.dumps(batch_to_dict(batch), indent=2), encoding="utf-8")
