"""Small display-formatting helpers shared by the CLI's `stats`,
`metrics`, and `report` commands. Nothing here touches pipeline
behavior - nothing formatted here is read back by any other module."""
from __future__ import annotations


def format_cents(cents: int) -> str:
    """Render an integer cent amount as a dollar string, e.g. `150` ->
    `"$1.50"`."""
    sign = "-" if cents < 0 else ""
    whole, remainder = divmod(abs(cents), 100)
    return f"{sign}${whole}.{remainder:02d}"


def format_duration_s(seconds: float) -> str:
    """Render a duration in seconds as milliseconds if it is under one
    second, otherwise as seconds with two decimal places."""
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def format_count(n: int, noun: str) -> str:
    """Pluralize `noun` for `n`, e.g. `format_count(1, "event")` ->
    `"1 event"`, `format_count(3, "event")` -> `"3 events"`."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def format_percent(part: int, whole: int) -> str:
    """Render `part` as a percentage of `whole`, e.g. `(1, 4)` ->
    `"25.0%"`. Returns `"n/a"` if `whole` is zero."""
    if whole <= 0:
        return "n/a"
    return f"{(part / whole) * 100:.1f}%"


def truncate(text: str, max_length: int = 60) -> str:
    """Shorten `text` to `max_length` characters, appending an ellipsis
    if anything was cut."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"
