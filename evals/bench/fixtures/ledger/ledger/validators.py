"""Row-level and value-level validation for imported ledger data.

Two dispatch styles are used side by side, on purpose: most of this module
is plain functions called directly, but the per-column checks the importer
runs while walking a CSV row are registered by column name
(`@register_field_validator`) and looked up dynamically, so the row loop
only needs to know column names, not the full list of validator functions.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Callable

AMOUNT_RE = re.compile(r"^-?\d+(\.\d{1,2})?$")
ACCOUNT_ID_RE = re.compile(r"^[A-Z0-9]{4,12}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
TXN_ID_RE = re.compile(r"^TX-\d{6,}$")


class ValidationError(ValueError):
    """Raised when a single field fails validation."""


def parse_amount(raw: str) -> float:
    """Parse a currency amount from a CSV field.

    Only accepts a plain, optionally-negative decimal with at most two
    fraction digits (e.g. "12.50", "-4"). Anything else -- including the
    strings "nan" and "inf" that `float()` would otherwise happily accept
    -- is rejected, so amounts flowing through the rest of the pipeline
    are always finite real numbers.
    """
    raw = raw.strip()
    if not AMOUNT_RE.match(raw):
        raise ValidationError(f"not a valid amount: {raw!r}")
    return float(raw)


def parse_timestamp(raw: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp, e.g. "2026-01-15T09:30:00Z"."""
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValidationError(f"not a valid timestamp: {raw!r}") from exc


def is_valid_account_id(value: str) -> bool:
    return bool(ACCOUNT_ID_RE.match(value.strip()))


def is_valid_currency_code(value: str) -> bool:
    return bool(CURRENCY_RE.match(value.strip()))


def is_valid_txn_id(value: str) -> bool:
    return bool(TXN_ID_RE.match(value.strip()))


def validate_swift_bic(code: str) -> bool:
    """Validate a SWIFT/BIC bank identifier code (8 or 11 characters).

    Kept for the international wire-transfer import path; none of the
    CSV formats accepted today carry a BIC column, so this currently has
    no caller.
    """
    code = code.strip().upper()
    if len(code) not in (8, 11):
        return False
    bank = code[0:4]
    country = code[4:6]
    location = code[6:8]
    if not bank.isalpha():
        return False
    if not country.isalpha():
        return False
    if not location.isalnum():
        return False
    if len(code) == 11:
        branch = code[8:11]
        return branch.isalnum()
    return True


def classify_amount(amount: float) -> str:
    """Classify a transaction amount for the listing's legend column."""
    if amount > 0:
        return "credit"
    elif amount < 0:
        return "debit"
    elif amount == 0:
        return "zero"
    else:
        # `amount` always comes from parse_amount(), which only accepts
        # a plain finite decimal, so one of the three branches above
        # always matches.
        return "unknown"


FIELD_VALIDATORS: dict[str, Callable[[str], bool]] = {}


def register_field_validator(field_name: str):
    """Decorator registering a validator under a CSV column name.

    `importer._validate_row` looks these up dynamically
    (`FIELD_VALIDATORS[column]`) while walking a row's fields, so a
    validator's only "caller" is a dict lookup keyed by the column
    header -- there is no direct call site for it in this file.
    """

    def decorator(fn: Callable[[str], bool]) -> Callable[[str], bool]:
        FIELD_VALIDATORS[field_name] = fn
        return fn

    return decorator


@register_field_validator("account_id")
def _validate_account_id_field(value: str) -> bool:
    return is_valid_account_id(value)


@register_field_validator("currency")
def _validate_currency_field(value: str) -> bool:
    return is_valid_currency_code(value)


@register_field_validator("txn_id")
def _validate_txn_id_field(value: str) -> bool:
    return is_valid_txn_id(value)
