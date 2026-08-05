"""Configuration loading for the ledger CLI.

Reads a JSON config file, if one is given, and falls back to built-in
defaults for anything the file doesn't set. A single `Config` object is
threaded through the import, aggregation, and reporting commands.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PAGE_SIZE = 5
DEFAULT_CURRENCY = "USD"
DEFAULT_CACHE_TTL_SECONDS = 300


@dataclass
class Config:
    """Runtime configuration for a ledger run.

    Attributes:
        page_size: number of rows per page when listing transactions.
        base_currency: currency code used as the default when a report
            needs a single reporting currency; accounts still carry their
            own currency for account-level totals.
        strict_duplicate_check: when true, importing a batch that
            contains two rows with the same transaction id should reject
            the batch outright instead of keeping the later row. Some
            upstream feeds are known to resend the last row of a batch on
            retry, so most deployments run with this off; the stricter
            check exists for feeds that have been audited to never repeat
            a transaction id.
        report_dir: directory generated reports are written to.
        cache_ttl_seconds: how long an account cache entry stays warm
            before a lookup treats it as expired.
    """

    page_size: int = DEFAULT_PAGE_SIZE
    base_currency: str = DEFAULT_CURRENCY
    strict_duplicate_check: bool = False
    report_dir: str = "reports"
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    known_categories: list[str] = field(default_factory=lambda: [
        "sales", "refund", "fee", "payroll", "transfer", "adjustment",
    ])

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        """Load config from a JSON file, or return defaults if `path` is None."""
        if path is None:
            return cls()
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = data.get("known_categories")
        return cls(
            page_size=int(data.get("page_size", DEFAULT_PAGE_SIZE)),
            base_currency=data.get("base_currency", DEFAULT_CURRENCY),
            strict_duplicate_check=bool(data.get("strict_duplicate_check", False)),
            report_dir=data.get("report_dir", "reports"),
            cache_ttl_seconds=int(data.get("cache_ttl_seconds", DEFAULT_CACHE_TTL_SECONDS)),
            known_categories=list(known) if known else cls().known_categories,
        )
