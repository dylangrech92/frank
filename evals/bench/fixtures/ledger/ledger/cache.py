"""In-memory account lookup cache.

Account records are read once per import but looked up repeatedly while
building a report -- every account-level summary row needs the account's
currency resolved by the same key the report already groups by. This
cache avoids a linear rescan of `Batch.accounts` for every row.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .models import Account


@dataclass
class _Entry:
    account: Account
    cached_at: float


class AccountCache:
    """Caches accounts by display name.

    Report sections group and label rows by account name, so keying the
    cache the same way lets a report row resolve its account directly
    from the name it already has on hand.
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, _Entry] = {}

    def warm(self, accounts: list[Account]) -> None:
        now = time.monotonic()
        for account in accounts:
            self._entries[account.name] = _Entry(account, now)

    def lookup(self, name: str) -> Account | None:
        entry = self._entries.get(name)
        if entry is None:
            return None
        if time.monotonic() - entry.cached_at > self._ttl_seconds:
            return None
        return entry.account

    def size(self) -> int:
        return len(self._entries)
