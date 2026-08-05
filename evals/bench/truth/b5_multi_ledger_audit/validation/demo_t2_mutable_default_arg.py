#!/usr/bin/env python3
"""Demonstrates ledger/aggregator.py:summarize_by_category's mutable
default argument (`running_totals: dict[str, float] = {}`) accumulating
state across separate calls -- exactly the shape main.py's `aggregate`
command hits when given more than one batch directory in one process.
"""
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "ledger"
sys.path.insert(0, str(FIXTURE))

from ledger.aggregator import summarize_by_category  # noqa: E402
from ledger.models import Transaction  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


def txn(txn_id, account_id, amount, category):
    return Transaction(txn_id, account_id, datetime.now(timezone.utc), amount, category)


def main() -> int:
    day1 = [txn("A1", "ACC1", -0.10, "fee")]
    day2 = [txn("B1", "ACC1", 200.00, "sales")]

    totals_day1 = summarize_by_category(day1)
    print(f"day 1 totals (no explicit running_totals passed): {totals_day1}")

    totals_day2 = summarize_by_category(day2)
    print(f"day 2 totals (no explicit running_totals passed): {totals_day2}")

    if "fee" in totals_day2:
        print(f"CONFIRMED: day 2's result still contains day 1's 'fee' entry "
              f"({totals_day2['fee']!r}) even though day 2's own transactions "
              f"never had a 'fee' category -- both calls shared the same default "
              f"dict object.")
        print(f"is totals_day1 is totals_day2: {totals_day1 is totals_day2}")
        return 0
    else:
        print("NOT REPRODUCED: day 2's totals were isolated from day 1's; the "
              "mutable-default-argument defect may have been fixed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
