"""Report output handlers.

Each handler renders a list of report rows into one output format.
`render(fmt, ...)` looks the handler up by name (`handle_<fmt>`) in this
module via `getattr` rather than a dict, so adding a new format is just
adding a new `handle_<name>` function here -- there is no separate
registry to keep in sync with the CLI's `--format` choices.
"""
from __future__ import annotations

import csv
import io
import json
import sys


def handle_csv(rows: list[dict], fields: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def handle_json(rows: list[dict], fields: list[str]) -> str:
    return json.dumps(rows, indent=2, default=str)


def handle_text(rows: list[dict], fields: list[str]) -> str:
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(f, "")) for f in fields))
    return "\n".join(lines)


def render(fmt: str, rows: list[dict], fields: list[str]) -> str:
    """Render `rows` in the requested output format.

    Dispatches to `handle_<fmt>` in this module by name; an unknown
    format raises the same kind of error a missing dict key would.
    """
    handler_name = f"handle_{fmt}"
    handler = getattr(sys.modules[__name__], handler_name, None)
    if handler is None:
        raise ValueError(f"unknown report format: {fmt!r}")
    return handler(rows, fields)
