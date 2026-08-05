"""Core data model: accounts, transactions, and the batch container the
importer produces."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Account:
    account_id: str
    name: str
    currency: str
    tz_name: str
    opening_balance: float
    expected_closing_balance: float = 0.0
    # Populated by the importer at import time (see importer.import_accounts);
    # zero until then.
    utc_offset_minutes: int = 0


@dataclass
class Transaction:
    txn_id: str
    account_id: str
    timestamp_utc: datetime
    amount: float
    category: str
    description: str = ""


@dataclass
class RowError:
    """A single rejected input row, kept for the import report."""

    source_file: str
    line_number: int
    reason: str


@dataclass
class Batch:
    """The result of importing one directory of CSV files."""

    accounts: list[Account] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.transactions)
