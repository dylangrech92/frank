"""Tiny formatting/rendering helpers for profiling output."""

from __future__ import annotations


def fmt_bytes(n: int) -> str:
    """Format *n* bytes as B / KB / MB / GB with one decimal place."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.1f} GB"


def fmt_seconds(s: float) -> str:
    """Format *s* seconds as ``"1.234s"`` or milliseconds for sub-second values."""
    if s < 1.0:
        return f"{s * 1000:.1f}ms"
    return f"{s:.3f}s"


def fmt_count(n: int) -> str:
    """Format *n* with thousands separators (e.g. ``1,234,567``)."""
    return f"{n:,}"


def render_top_table(
    headers: list[str],
    rows: list[tuple],
    top: int,
    total_label: str,
) -> str:
    """Render a fixed-width table truncated to *top* rows.

    Args:
        headers: Column headers.
        rows: Row tuples (one per table row).
        top: Maximum number of rows to display.
        total_label: Label for the total count (e.g. ``"files"``).

    Returns:
        A string with fixed-width columns, truncated to *top* rows, followed by
        a summary line like ``"(top 10 of 42 files)"``.
    """
    if not headers:
        return ""

    # Compute column widths from headers and rows.
    num_cols = len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            cell_str = str(cell)
            if i < num_cols and len(cell_str) > col_widths[i]:
                col_widths[i] = len(cell_str)

    # Format header row.
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))

    # Truncate rows.
    display_rows = rows[:top]
    total_rows = len(rows)

    lines = [header_line]
    for row in display_rows:
        line = "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)

    # Summary line.
    if total_rows > top:
        summary = f"(top {top} of {total_rows} {total_label})"
    else:
        summary = f"({total_rows} {total_label})"
    lines.append(summary)

    return "\n".join(lines)


def format_streams(stdout_text: str, stderr_text: str) -> str:
    """Render captured stdout/stderr with section labels.

    The ``--- stderr ---`` section is only emitted when *stderr_text* is
    non-empty, so a run that produced no error output does not carry an
    empty labelled block.
    """
    if stderr_text:
        return f'--- stdout ---\n{stdout_text}\n--- stderr ---\n{stderr_text}'
    return f'--- stdout ---\n{stdout_text}'


def tail_lines(text: str, n: int = 10) -> str:
    """Return the last *n* lines of *text*, joined with newlines."""
    return '\n'.join(str(text).splitlines()[-n:])
