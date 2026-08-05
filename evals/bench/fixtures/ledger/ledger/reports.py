"""Report building: combines the imported batch with aggregation, caching,
and timezone conversion to produce the CLI's `report` output sections."""
from __future__ import annotations

import logging
from collections import defaultdict

from . import aggregator, handlers
from .cache import AccountCache
from .currency import convert
from .models import Account, Batch
from .timezones import apply_cached_offset

logger = logging.getLogger(__name__)


def build_account_cache(batch: Batch, ttl_seconds: int) -> AccountCache:
    cache = AccountCache(ttl_seconds=ttl_seconds)
    cache.warm(batch.accounts)
    logger.debug("account cache warmed: %d entries", cache.size())
    return cache


def account_summary_rows(batch: Batch, cache: AccountCache) -> list[dict]:
    """One row per account that has at least one transaction in this
    batch, with currency resolved through the account cache (the cache
    already holds every account warmed for this batch, so a second scan
    of `batch.accounts` isn't needed here)."""
    totals_by_id: dict[str, float] = defaultdict(float)
    for txn in batch.transactions:
        totals_by_id[txn.account_id] += txn.amount

    accounts_by_id = {a.account_id: a for a in batch.accounts}
    rows = []
    for account_id, total in sorted(totals_by_id.items()):
        account = accounts_by_id.get(account_id)
        name = account.name if account else "unknown"
        cached = cache.lookup(name)
        currency = cached.currency if cached else "?"
        rows.append({
            "account_id": account_id,
            "account_name": name,
            "currency": currency,
            "total": round(total, 2),
        })
    return rows


def top_movers_rows(batch: Batch, limit: int = 10) -> list[dict]:
    movers = aggregator.top_movers(batch.transactions, limit=limit)
    return [
        {"txn_id": t.txn_id, "account_id": t.account_id, "amount": t.amount, "category": t.category}
        for t in movers
    ]


def recent_activity_rows(batch: Batch, limit: int = 10) -> list[dict]:
    """The most recent `limit` transactions, in import (chronological)
    order."""
    recent = batch.transactions[-limit:]
    return [
        {
            "txn_id": t.txn_id,
            "account_id": t.account_id,
            "amount": t.amount,
            "timestamp": t.timestamp_utc.isoformat(),
        }
        for t in recent
    ]


def daily_totals_rows(batch: Batch, account: Account) -> list[dict]:
    """Per-calendar-day totals for one account, in the account's local
    timezone."""
    totals: dict = defaultdict(float)
    for txn in batch.transactions:
        if txn.account_id != account.account_id:
            continue
        local_dt = apply_cached_offset(txn.timestamp_utc, account.utc_offset_minutes)
        totals[local_dt.date()] += txn.amount
    return [{"date": d.isoformat(), "total": round(total, 2)} for d, total in sorted(totals.items())]


def currency_exposure_rows(account_rows: list[dict], target_currency: str) -> list[dict]:
    """Convert each account-summary row's total into one reporting currency.

    Takes the output of `account_summary_rows` (each row already carries
    the account's native currency and total) rather than the raw batch,
    so this stays a pure presentation step downstream of the totals
    everything else already agrees on -- it can't itself introduce a
    totals discrepancy.

    Args:
        account_rows: rows as produced by `account_summary_rows`.
        target_currency: the currency every row's total is converted into.

    Returns:
        One row per input row, with the native total preserved alongside
        the converted total.
    """
    rows = []
    for row in account_rows:
        native_currency = row["currency"]
        if native_currency == "?":
            converted = None
        else:
            converted = convert(row["total"], native_currency, target_currency)
        rows.append({
            "account_id": row["account_id"],
            "account_name": row["account_name"],
            "native_currency": native_currency,
            "native_total": row["total"],
            "target_currency": target_currency,
            "converted_total": converted,
        })
    return rows


def render_report(fmt: str, rows: list[dict], fields: list[str]) -> str:
    return handlers.render(fmt, rows, fields)
