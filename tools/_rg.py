"""Shared ripgrep plumbing for the content- and filename-search tools.

The leading underscore keeps this module out of tool discovery by the registry.
Both ``find`` (file contents) and ``find_files`` (filenames) shell out to the
same ``rg`` binary and must agree on how an absent binary is reported and which
harness-private directory is never searched — that shared surface lives here so
the two tools cannot drift apart.  Their actual ``rg`` invocations differ
(``--line-number`` match search vs ``--files`` name listing) and stay local to
each tool.
"""

from __future__ import annotations

import shutil

from tools.result import ToolResult

# The harness writes its own session state under .coding_agent; it is never a
# meaningful search target, so both find tools exclude it explicitly (on top of
# whatever .gitignore already excludes).
IGNORE_GLOB = '!.coding_agent'


def locate_rg() -> tuple[str | None, ToolResult | None]:
    """Locate the ripgrep binary, returning ``(path, None)`` or ``(None, error)``.

    Returns:
        A ``(rg_path, None)`` pair when ``rg`` is on ``PATH``; otherwise
        ``(None, error_result)`` carrying the shared ``missing-engine`` error so
        both find tools report an absent binary identically.
    """
    rg = shutil.which('rg')
    if rg is None:
        return None, ToolResult.err(
            'The ripgrep binary rg was not found on PATH.',
            code='missing-engine',
            hint='Install it with: brew install ripgrep',
        )
    return rg, None


def strip_dot_prefix(lines: list[str]) -> list[str]:
    """Strip the leading ``./`` ripgrep prepends when the search path is ``.``.

    Searching the literal ``.`` makes ``rg`` print every path with a ``./``
    prefix; removing it keeps reported paths clean project-root-relative.
    """
    return [line[2:] if line.startswith('./') else line for line in lines]
