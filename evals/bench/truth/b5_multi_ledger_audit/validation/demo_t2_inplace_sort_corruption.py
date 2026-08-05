#!/usr/bin/env python3
"""Demonstrates ledger/aggregator.py:top_movers sorting its argument list
IN PLACE, corrupting a list that's shared (and expected to still be in
chronological order) elsewhere -- exactly what ledger/reports.py does by
passing `batch.transactions` itself into `top_movers`, then reading
`batch.transactions[-limit:]` afterwards expecting recency order.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "ledger"
sys.path.insert(0, str(FIXTURE))

from ledger.aggregator import top_movers  # noqa: E402
from ledger.models import Batch, Transaction  # noqa: E402
from ledger import reports  # noqa: E402


def txn(txn_id, amount, minute_offset):
    ts = datetime(2026, 1, 15, 9, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minute_offset)
    return Transaction(txn_id, "ACC1", ts, amount, "misc")


def main() -> int:
    batch = Batch()
    # Chronological order (insertion order): small amounts first, one big
    # amount last -- a "most recent = biggest" batch would mask the bug,
    # so the big mover is placed in the middle on purpose.
    batch.transactions = [
        txn("T1", -1.00, 0),
        txn("T2", -2.00, 5),
        txn("T3", 500.00, 10),   # the big mover
        txn("T4", -3.00, 15),
        txn("T5", -4.00, 20),
    ]
    chronological_ids_before = [t.txn_id for t in batch.transactions]
    print(f"chronological order before top_movers(): {chronological_ids_before}")

    movers = top_movers(batch.transactions, limit=1)
    print(f"top mover: {[t.txn_id for t in movers]}")

    order_after = [t.txn_id for t in batch.transactions]
    print(f"batch.transactions order AFTER top_movers(): {order_after}")

    recent = reports.recent_activity_rows(batch, limit=2)
    recent_ids = [r["txn_id"] for r in recent]
    print(f"recent_activity_rows(limit=2) returns: {recent_ids} "
          f"(chronologically the 2 most recent are T4, T5)")

    if order_after != chronological_ids_before:
        print("CONFIRMED: top_movers() mutated batch.transactions in place -- it "
              "is no longer in chronological order.")
        if recent_ids != ["T4", "T5"]:
            print(f"CONFIRMED (downstream): recent_activity_rows now returns "
                  f"{recent_ids} instead of the true two most recent (['T4', 'T5']) "
                  f"because it reads the tail of the now-amount-sorted list.")
            return 0
    print("NOT REPRODUCED: order was preserved; the in-place-sort defect may have "
          "been fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
