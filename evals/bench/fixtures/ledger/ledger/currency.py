"""Currency conversion for cross-account reporting.

Individual accounts keep their own currency (an account's opening balance
and reconciliation always happen in that account's native currency -- see
`aggregator.reconcile`). Reports that need to compare or total *across*
accounts denominated in different currencies convert each account's total
into a single reporting currency first; this module holds that conversion
and the static exchange-rate table it uses.

The rate table is a fixed snapshot rather than a live feed on purpose: the
whole ledger tool is stdlib-only and runs with no network access, so "live"
rates aren't available, and a benchmark fixture needs deterministic output
across runs regardless. A real deployment would swap this module's table
for a rate provider without changing any caller.
"""
from __future__ import annotations

# Units of each currency per 1 USD, i.e. USD is the table's base currency.
# A rate of 0.92 for EUR means 1 USD buys 0.92 EUR.
USD_PER_UNIT: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.50,
}


class UnknownCurrencyError(ValueError):
    """Raised when a currency code isn't in the fixed rate table."""


def convert(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert `amount` from `from_currency` to `to_currency`.

    Both currency codes must be keys in `USD_PER_UNIT`. Conversion always
    routes through USD internally (amount -> USD -> target currency),
    which also means converting a currency to itself is a no-op modulo
    floating-point rounding, not a hardcoded identity shortcut.

    Args:
        amount: the amount, denominated in `from_currency`.
        from_currency: three-letter source currency code.
        to_currency: three-letter target currency code.

    Returns:
        The equivalent amount denominated in `to_currency`, rounded to two
        decimal places.

    Raises:
        UnknownCurrencyError: if either code isn't in the rate table.
    """
    from_currency = from_currency.strip().upper()
    to_currency = to_currency.strip().upper()
    if from_currency not in USD_PER_UNIT:
        raise UnknownCurrencyError(f"no rate for currency: {from_currency!r}")
    if to_currency not in USD_PER_UNIT:
        raise UnknownCurrencyError(f"no rate for currency: {to_currency!r}")
    amount_in_usd = amount / USD_PER_UNIT[from_currency]
    converted = amount_in_usd * USD_PER_UNIT[to_currency]
    return round(converted, 2)
