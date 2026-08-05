#!/usr/bin/env python3
"""Demonstrates ledger/cache.py:AccountCache being keyed by account NAME
instead of account_id: two distinct accounts sharing a display name
collide in the cache, and a lookup for one silently returns the other's
(stale, wrong) record.
"""
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "ledger"
sys.path.insert(0, str(FIXTURE))

from ledger.cache import AccountCache  # noqa: E402
from ledger.models import Account  # noqa: E402


def main() -> int:
    acc_ny = Account(
        account_id="ACC1001", name="Operating Account", currency="USD",
        tz_name="America/New_York", opening_balance=1000.00,
        expected_closing_balance=1000.00,
    )
    acc_eu = Account(
        account_id="ACC1003", name="Operating Account", currency="EUR",
        tz_name="America/Los_Angeles", opening_balance=200.00,
        expected_closing_balance=200.00,
    )
    print(f"ACC1001: name={acc_ny.name!r} currency={acc_ny.currency!r}")
    print(f"ACC1003: name={acc_eu.name!r} currency={acc_eu.currency!r}  "
          f"(same name as ACC1001, different account_id)")

    cache = AccountCache(ttl_seconds=300)
    cache.warm([acc_ny, acc_eu])  # ACC1003 warmed after ACC1001, same key

    resolved = cache.lookup("Operating Account")
    print(f"cache.lookup('Operating Account') -> account_id={resolved.account_id!r} "
          f"currency={resolved.currency!r}")

    if resolved.account_id != acc_ny.account_id:
        print(f"CONFIRMED: a lookup meant to resolve ACC1001's own record instead "
              f"returned {resolved.account_id}'s record (currency {resolved.currency} "
              f"instead of {acc_ny.currency}) -- the cache is keyed by a field "
              f"(display name) that is not unique across accounts.")
        return 0
    print("NOT REPRODUCED: lookup returned the correct account; the cache-key "
          "defect may have been fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
