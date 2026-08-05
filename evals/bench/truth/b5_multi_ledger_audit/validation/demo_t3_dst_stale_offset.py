#!/usr/bin/env python3
"""Demonstrates ledger/reports.py:daily_totals_rows using an account's
UTC offset cached once at import time (ledger/timezones.py) for every
transaction's calendar-day bucketing -- correct for transactions on the
same side of a DST change as the import date, wrong (off by a day right
around local midnight) for transactions on the other side.

America/New_York's real 2026 DST calendar is used: clocks spring forward
on 2026-03-08 (EST -05:00 -> EDT -04:00). An account imported in January
(EST, cached offset -300 minutes) is then fed a July transaction (truly
EDT, -240 minutes) that falls at 00:20 local time -- just after local
midnight, so the correct calendar day is 2026-07-15, but the stale
January offset pushes it to 23:20 the previous evening, bucketing it
into 2026-07-14 instead.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "ledger"
sys.path.insert(0, str(FIXTURE))

from ledger.timezones import cached_utc_offset_minutes, apply_cached_offset  # noqa: E402
from ledger.models import Account, Batch, Transaction  # noqa: E402
from ledger.reports import daily_totals_rows  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402


def correct_local_date(utc_dt: datetime, tz_name: str):
    """A correct, live zoneinfo-based conversion -- used only to compute
    the expected right answer for this script, not part of the app."""
    return utc_dt.astimezone(ZoneInfo(tz_name)).date()


def main() -> int:
    tz_name = "America/New_York"

    # Simulate importing this account's batch in January 2026 (EST).
    import_time = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    cached_offset = cached_utc_offset_minutes(tz_name, as_of=import_time)
    print(f"account imported {import_time.date()} -> cached_offset_minutes={cached_offset} "
          f"({cached_offset / 60:.0f}h, EST)")

    account = Account(
        account_id="ACC1001", name="Operating Account", currency="USD",
        tz_name=tz_name, opening_balance=0.0, expected_closing_balance=0.0,
        utc_offset_minutes=cached_offset,
    )

    # A transaction on 2026-07-15 at 00:20 local (EDT, -04:00) -> UTC 04:20.
    txn_utc = datetime(2026, 7, 15, 4, 20, tzinfo=timezone.utc)
    expected_date = correct_local_date(txn_utc, tz_name)
    print(f"transaction UTC timestamp: {txn_utc.isoformat()}")
    print(f"correct local calendar date (live zoneinfo, EDT -04:00): {expected_date}")

    batch = Batch()
    batch.transactions = [Transaction("TX1", "ACC1001", txn_utc, 42.00, "sales")]
    rows = daily_totals_rows(batch, account)
    print(f"daily_totals_rows() bucketed it under: {rows}")

    buggy_local = apply_cached_offset(txn_utc, cached_offset)
    print(f"apply_cached_offset() with the stale January offset gives: "
          f"{buggy_local.isoformat()} -> date {buggy_local.date()}")

    got_date = rows[0]["date"] if rows else None
    if got_date != str(expected_date):
        print(f"CONFIRMED: the transaction was bucketed into {got_date} instead of "
              f"the correct {expected_date} -- a full day off -- because the "
              f"account's UTC offset was cached in January (EST) and never "
              f"re-derived for this July (EDT) transaction's own date.")

        # Control case: a January transaction, same DST regime as the
        # import date, should NOT be affected.
        control_utc = datetime(2026, 1, 20, 4, 20, tzinfo=timezone.utc)
        control_expected = correct_local_date(control_utc, tz_name)
        batch.transactions = [Transaction("TX2", "ACC1001", control_utc, 10.00, "sales")]
        control_rows = daily_totals_rows(batch, account)
        control_got = control_rows[0]["date"] if control_rows else None
        print(f"control (January transaction, same regime as import): "
              f"got={control_got} expected={control_expected} "
              f"{'MATCH' if control_got == str(control_expected) else 'MISMATCH'}")
        if control_got == str(control_expected):
            print("CONFIRMED: the bug is specific to the DST boundary -- a "
                  "same-regime transaction buckets correctly.")
            return 0
    print("NOT REPRODUCED: the day-bucketing was correct, or the control case also "
          "failed; the DST defect may have been fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
