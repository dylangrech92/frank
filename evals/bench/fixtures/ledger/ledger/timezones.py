"""Timezone conversion helpers.

Transactions are stored with a UTC timestamp; reports show times in the
owning account's local timezone. To avoid a `zoneinfo` lookup for every
row in a large batch, an account's UTC offset is resolved once, at import
time, and cached on the `Account` record (see `importer.import_accounts`).
Every later conversion for that account -- both the per-row time shown in
a listing and the calendar day a transaction is bucketed into for the
daily-totals report -- reuses that one cached offset.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


def cached_utc_offset_minutes(tz_name: str, as_of: datetime | None = None) -> int:
    """Return the UTC offset (in minutes) in effect for `tz_name`, evaluated once.

    `as_of` defaults to the current time, i.e. import time -- the offset
    an account is tagged with reflects whichever side of a daylight
    saving change is active when the batch is imported.
    """
    reference = as_of or datetime.now()
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    localized = reference.astimezone(ZoneInfo(tz_name))
    offset = localized.utcoffset()
    return int(offset.total_seconds() // 60) if offset else 0


def apply_cached_offset(utc_dt: datetime, offset_minutes: int) -> datetime:
    """Shift a UTC timestamp by a previously-cached offset.

    Returns a naive datetime carrying the account's local wall-clock
    fields. Used for both per-row display and calendar-day bucketing, so
    the two always agree with each other.
    """
    naive_utc = utc_dt.replace(tzinfo=None) if utc_dt.tzinfo else utc_dt
    return naive_utc + timedelta(minutes=offset_minutes)
