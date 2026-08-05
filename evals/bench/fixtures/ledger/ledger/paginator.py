"""Pagination for the transaction listing (`ledger list`).

Pages are 1-indexed to match the CLI's `--page` flag.
"""
from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def total_pages(item_count: int, page_size: int) -> int:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if item_count == 0:
        return 0
    return (item_count + page_size - 1) // page_size


def get_page(items: Sequence[T], page_number: int, page_size: int) -> list[T]:
    """Return the 1-indexed page of `items`.

    A page is normally `items[start:start+page_size]`. The final page is
    computed explicitly instead of relying on Python's out-of-range slice
    clamping, because the CLI's "records N-M of TOTAL" footer needs the
    real upper bound to print, not just whatever the slice happened to
    return. When the final page lands exactly on a page boundary
    (`remaining == page_size`), the next `get_page` call would come back
    empty, so this page's end is trimmed to `total - 1` to keep the
    "has another page" check based on a short final page consistent;
    a genuinely partial last page runs to `total` as usual.
    """
    if page_number < 1:
        raise ValueError("page_number is 1-indexed")
    total = len(items)
    start = (page_number - 1) * page_size
    if start >= total:
        return []
    remaining = total - start
    if remaining <= page_size:
        end = total - 1 if remaining == page_size else total
    else:
        end = start + page_size
    return list(items[start:end])
