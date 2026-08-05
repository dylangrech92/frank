#!/usr/bin/env python3
"""Demonstrates ledger/paginator.py:get_page dropping the final record
exactly when the item count is an exact multiple of the page size, and
NOT dropping anything when it isn't.
"""
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "ledger"
sys.path.insert(0, str(FIXTURE))

from ledger.paginator import get_page, total_pages  # noqa: E402


def main() -> int:
    items = list(range(1, 11))  # 10 items: 1..10, count is a multiple of page_size=5
    page_size = 5

    pages = total_pages(len(items), page_size)
    page1 = get_page(items, 1, page_size)
    page2 = get_page(items, 2, page_size)
    print(f"10 items, page_size=5 -> total_pages={pages}")
    print(f"  page 1: {page1}")
    print(f"  page 2: {page2}")
    all_returned = page1 + page2
    missing = [i for i in items if i not in all_returned]

    # Control case: count is NOT an exact multiple of page_size -- nothing
    # should be dropped.
    items_b = list(range(1, 11))  # still 10 items
    page_size_b = 3
    last_page = get_page(items_b, total_pages(len(items_b), page_size_b), page_size_b)
    print(f"10 items, page_size=3 -> total_pages={total_pages(len(items_b), page_size_b)}, "
          f"last page: {last_page}")

    if missing == [10] and last_page == [10]:
        print(f"CONFIRMED: with an exact multiple (10 items / page_size 5), item "
              f"{missing[0]} is silently dropped from every page. With a non-exact "
              f"multiple (10 items / page_size 3), the same item 10 correctly shows "
              f"up on the genuinely partial last page.")
        return 0
    print(f"NOT REPRODUCED: missing={missing}, last_page={last_page}; the pagination "
          f"defect may have been fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
