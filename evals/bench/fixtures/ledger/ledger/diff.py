"""Compare two imported batches.

Batches for the same underlying period sometimes get re-delivered (a
corrected drop after a source-system fix, a re-run against yesterday's
directory to see what changed since a partial import). This module
answers "what's different between these two batches" without either
batch needing to know the other exists -- both are just whatever
`import_directory` already produced.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Batch


@dataclass
class BatchDiff:
    accounts_added: list[str]
    accounts_removed: list[str]
    transactions_added: list[str]
    transactions_removed: list[str]
    error_count_before: int
    error_count_after: int


def diff_batches(before: Batch, after: Batch) -> BatchDiff:
    """Compare `before` against `after` by id set membership.

    Only additions and removals are reported -- a transaction whose id
    is present in both batches is treated as unchanged even if some
    other field on it differs, since txn_id is the batch's own primary
    key for a transaction (see `models.Transaction`) and two rows
    sharing an id are never expected to disagree on anything else.
    """
    before_account_ids = {a.account_id for a in before.accounts}
    after_account_ids = {a.account_id for a in after.accounts}
    before_txn_ids = {t.txn_id for t in before.transactions}
    after_txn_ids = {t.txn_id for t in after.transactions}
    return BatchDiff(
        accounts_added=sorted(after_account_ids - before_account_ids),
        accounts_removed=sorted(before_account_ids - after_account_ids),
        transactions_added=sorted(after_txn_ids - before_txn_ids),
        transactions_removed=sorted(before_txn_ids - after_txn_ids),
        error_count_before=len(before.errors),
        error_count_after=len(after.errors),
    )


def diff_rows(diff: BatchDiff) -> list[dict]:
    """Flatten a `BatchDiff` into one row per changed entity, in a
    stable order (accounts before transactions, additions before
    removals) so report output doesn't jump around between runs."""
    rows: list[dict] = []
    for account_id in diff.accounts_added:
        rows.append({"change": "account added", "id": account_id})
    for account_id in diff.accounts_removed:
        rows.append({"change": "account removed", "id": account_id})
    for txn_id in diff.transactions_added:
        rows.append({"change": "transaction added", "id": txn_id})
    for txn_id in diff.transactions_removed:
        rows.append({"change": "transaction removed", "id": txn_id})
    return rows


def has_changes(diff: BatchDiff) -> bool:
    """True if anything at all differs between the two batches,
    including a change in how many rows were rejected on import (the
    id-set fields alone wouldn't catch a batch that re-rejects a
    previously-accepted row without changing any accepted id)."""
    return bool(
        diff.accounts_added
        or diff.accounts_removed
        or diff.transactions_added
        or diff.transactions_removed
        or diff.error_count_before != diff.error_count_after
    )
