#!/usr/bin/env python3
"""Demonstrates ledger/aggregator.py:reconcile comparing two currency
totals with `==`, so an account that is correct to the cent is flagged
MISMATCH purely from binary float representation error.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "ledger"
sys.path.insert(0, str(FIXTURE))

from ledger.aggregator import reconcile  # noqa: E402
from ledger.models import Account, Transaction  # noqa: E402


def txn(txn_id, amount):
    return Transaction(txn_id, "ACC1001", datetime.now(timezone.utc), amount, "fee")


def main() -> int:
    account = Account(
        account_id="ACC1001", name="Operating Account", currency="USD",
        tz_name="America/New_York", opening_balance=1000.00,
        expected_closing_balance=999.70,
    )
    transactions = [txn("T1", -0.10), txn("T2", -0.10), txn("T3", -0.10)]

    is_balanced, computed = reconcile(transactions, account)
    print(f"computed total:  {computed!r}")
    print(f"expected total:  {account.expected_closing_balance!r}")
    print(f"computed == expected: {computed == account.expected_closing_balance}")
    print(f"round(computed, 2) == expected: {round(computed, 2) == account.expected_closing_balance}")
    print(f"reconcile() -> is_balanced={is_balanced}")

    if not is_balanced and round(computed, 2) == account.expected_closing_balance:
        print("CONFIRMED: the account is correct to the cent (round(computed, 2) "
              "== expected), but reconcile() reports MISMATCH because `==` on the "
              "raw floats fails due to binary floating-point drift from summing "
              "0.10 three times.")
        return 0
    print("NOT REPRODUCED: reconcile() returned balanced, or the drift disappeared; "
          "the float-equality defect may have been fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
