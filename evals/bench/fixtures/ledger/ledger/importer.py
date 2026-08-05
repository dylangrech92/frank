"""CSV import for the ledger pipeline.

Reads `accounts.csv` and `transactions.csv` from an input directory,
validates each row, and produces a `Batch`. Malformed rows are skipped
and recorded in `Batch.errors`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .errors import BatchImportError
from .models import Account, Batch, RowError, Transaction
from .timezones import cached_utc_offset_minutes
from .validators import FIELD_VALIDATORS, ValidationError, parse_amount, parse_timestamp

import csv

logger = logging.getLogger(__name__)

ACCOUNTS_FILENAME = "accounts.csv"
TRANSACTIONS_FILENAME = "transactions.csv"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts.

    A missing file is a normal partial import (a batch directory that
    only has one of the two input files), so any failure to open or read
    the file is treated the same way: no rows.
    """
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader)
    except OSError:
        return []


def _validate_row(row: dict[str, str], required: list[str]) -> str | None:
    for column in required:
        value = row.get(column, "")
        validator = FIELD_VALIDATORS.get(column)
        if validator is not None and not validator(value):
            return f"invalid {column}: {value!r}"
    return None


def import_accounts(path: Path, batch: Batch, source_name: str) -> None:
    rows = _read_csv_rows(path)
    for line_number, row in enumerate(rows, start=2):  # header is line 1
        error = _validate_row(row, ["account_id", "currency"])
        if error:
            batch.errors.append(RowError(source_name, line_number, error))
            continue
        tz_name = row.get("tz_name", "UTC").strip() or "UTC"
        try:
            opening_balance = parse_amount(row.get("opening_balance", "0"))
            expected_closing_balance = parse_amount(row.get("expected_closing_balance", "0"))
        except ValidationError as exc:
            batch.errors.append(RowError(source_name, line_number, str(exc)))
            continue
        account = Account(
            account_id=row["account_id"].strip(),
            name=row.get("name", "").strip(),
            currency=row["currency"].strip().upper(),
            tz_name=tz_name,
            opening_balance=opening_balance,
            expected_closing_balance=expected_closing_balance,
            utc_offset_minutes=cached_utc_offset_minutes(tz_name),
        )
        batch.accounts.append(account)


def import_transactions(path: Path, batch: Batch, source_name: str, config: Config) -> None:
    rows = _read_csv_rows(path)
    known_categories = {c.lower() for c in config.known_categories}
    for line_number, row in enumerate(rows, start=2):
        error = _validate_row(row, ["account_id", "txn_id"])
        if error:
            batch.errors.append(RowError(source_name, line_number, error))
            continue
        try:
            amount = parse_amount(row.get("amount", ""))
            timestamp = parse_timestamp(row.get("timestamp", ""))
        except ValidationError as exc:
            batch.errors.append(RowError(source_name, line_number, str(exc)))
            continue
        category = row.get("category", "").strip().lower()
        if category not in known_categories:
            batch.errors.append(RowError(
                source_name, line_number, f"unrecognized category: {category!r}",
            ))
            continue
        txn = Transaction(
            txn_id=row["txn_id"].strip(),
            account_id=row["account_id"].strip(),
            timestamp_utc=timestamp,
            amount=amount,
            category=category,
            description=row.get("description", "").strip(),
        )
        batch.transactions.append(txn)


def import_directory(input_dir: str | Path, config: Config | None = None) -> Batch:
    """Import every recognized CSV file under `input_dir` into one Batch.

    Args:
        input_dir: directory containing `accounts.csv` and/or
            `transactions.csv`.
        config: drives validation that depends on runtime settings (today,
            only the recognized transaction categories); defaults to
            `Config()` when not given, so existing callers that don't have
            a loaded config keep working unchanged.

    Raises:
        BatchImportError: if `input_dir` does not exist or is not a
            directory. This is distinct from a missing individual CSV
            file inside an existing directory, which `_read_csv_rows`
            treats as a normal partial import.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise BatchImportError(f"not a directory: {input_dir}")
    config = config or Config()
    batch = Batch()
    import_accounts(input_dir / ACCOUNTS_FILENAME, batch, ACCOUNTS_FILENAME)
    import_transactions(input_dir / TRANSACTIONS_FILENAME, batch, TRANSACTIONS_FILENAME, config)
    logger.info(
        "imported %d accounts, %d transactions, %d errors from %s",
        len(batch.accounts), len(batch.transactions), len(batch.errors), input_dir,
    )
    return batch
