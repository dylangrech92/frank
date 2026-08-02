"""Shared boilerplate for the single-file edit tools.

The leading underscore keeps this module out of tool discovery by the registry.
``update_file``, ``replace_one``, ``replace_many`` and ``edit_lines`` all have
to resolve a path under the project root, reject non-files, refuse to write
against a stale or never-read view of a file, and — after a successful write —
re-stamp the read registry and emit exactly one mutation event.  ``delete_file``
shares the freshness gate alone: a delete has nothing to re-stamp, and it also
accepts directories, which carry no read stamp.  Centralizing that surface here
keeps the tools byte-identical on their error codes, hints, and the
record_read/emit_mutation ordering, instead of each maintaining its own copy.

Exports
-------
resolve_existing_file : resolve a path under root and confirm it is a regular file
freshness_gate        : refuse a write against a stale / never-read file view
finalize_write        : re-stamp the read registry + emit one 'changed' event
looks_line_numbered   : True when a search string carries read_file's number prefix
"""

from __future__ import annotations

import re
from pathlib import Path

from tools._read_registry import check_fresh, record_read
from tools._sandbox import emit_mutation, resolve_in_root
from tools.result import ToolResult

# read_file renders content cat -n style ("     1\t<line>"); this matches a
# leading, optionally space-padded line number followed by the tab separator.
_LINE_NUMBER_PREFIX = re.compile(r'^\s*\d+\t')


def resolve_existing_file(raw_path: str) -> tuple[Path | None, ToolResult | None]:
    """Resolve *raw_path* under the project root and confirm it is a regular file.

    Returns ``(resolved, None)`` when the path stays under the root and points at
    an existing regular file, or ``(None, error)`` carrying the shared error
    contract otherwise: ``path-escapes-root``, ``not-found`` (with the create_file
    hint), or ``not-a-file``.
    """
    try:
        resolved = resolve_in_root(Path.cwd(), raw_path)
    except ValueError as exc:
        return None, ToolResult.err(str(exc), code='path-escapes-root')

    if not resolved.exists():
        return None, ToolResult.err(
            f'{raw_path} does not exist.',
            code='not-found',
            hint="Use create_file to create a new file.",
        )

    if not resolved.is_file():
        return None, ToolResult.err(
            f'{raw_path} is not a regular file.',
            code='not-a-file',
        )

    return resolved, None


def freshness_gate(resolved: Path, raw_path: str) -> ToolResult | None:
    """Refuse to write against a stale or never-read view of *resolved*.

    Returns a ``file-changed-on-disk`` error when the file changed on disk since
    this session last read it, a ``not-read-yet`` error when this session never
    read it, or ``None`` when the session's view is fresh and the write may
    proceed.
    """
    freshness = check_fresh(resolved)
    if freshness == 'stale':
        return ToolResult.err(
            f'{raw_path} changed on disk after you last read it — by your own last edit or another process.',
            code='file-changed-on-disk',
            hint='Re-read the file with read_file, then re-apply your edit against the current content.',
        )
    if freshness == 'unread':
        return ToolResult.err(
            f'{raw_path} has not been read yet in this session.',
            code='not-read-yet',
            hint='Read the file with read_file before editing it.',
        )
    return None


def finalize_write(resolved: Path) -> None:
    """Re-stamp the read registry and emit exactly one 'changed' mutation event.

    Call after a successful write so this session's own write does not make the
    file look stale for its next edit, and downstream mutation subscribers see
    the change exactly once.  Ordering (record_read then emit_mutation) is fixed
    here so every edit tool matches.
    """
    record_read(resolved)
    emit_mutation('changed', resolved)


def looks_line_numbered(text: str) -> bool:
    """Return True when every non-empty line of *text* carries a line-number prefix.

    read_file renders content cat -n style (``     1\\t<line>``); a model
    sometimes pastes those numbered lines straight into a ``search`` string,
    which then matches nothing.  This cheap check runs only on the no-match path
    so the replace tools can point at the real cause instead of a generic
    "not found".
    """
    non_empty = [line for line in text.split('\n') if line != '']
    if not non_empty:
        return False
    return all(_LINE_NUMBER_PREFIX.match(line) for line in non_empty)
