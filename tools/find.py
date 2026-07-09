"""Find tool: search file contents across the project using ripgrep."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._sandbox import resolve_in_root
from tools._rg import IGNORE_GLOB, locate_rg, strip_dot_prefix


class Find(Tool):
    """Searches file contents across the project using ripgrep.

    Returns matching lines with file path and line number. The *path* parameter
    limits the search to a subdirectory relative to the project root. When
    *fuzzy* is false (the default), *query* is matched as a literal fixed string.
    When *fuzzy* is true, *query* is split on whitespace and each part is matched
    as a case-insensitive regular expression with anything between the parts.

    For symbol names use find_symbol; for usages of a known symbol use find_references.
    """

    name = 'find'
    summary = 'Search file contents across the project (ripgrep).'
    description = (
        'Searches file CONTENTS across the project using ripgrep, returning matching '
        'lines with file and line number. Use find_files to search by filename/glob; '
        'find_symbol for symbol names; find_references for usages of a known symbol.'
    )
    action = 'search'
    oversize_hint = 'narrow the query or pass path to limit the scope'
    alternative = 'find_symbol (symbol names) or find_references (usages)'
    parallel_safe = True  # spawns its own ripgrep subprocess, reads only
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': (
                    'The text to search for. When fuzzy is false, matched as a literal '
                    'fixed string. When fuzzy is true, split on whitespace and each part '
                    'matched case-insensitively with anything allowed between parts.'
                ),
            },
            'path': {
                'type': 'string',
                'description': (
                    'Subdirectory relative to the project root to limit the search to. '
                    'Must be a relative path.'
                ),
            },
            'fuzzy': {
                'type': 'boolean',
                'default': False,
                'description': (
                    'When false, query is matched as a literal fixed string. '
                    'When true, query is split on whitespace and matched case-insensitively '
                    'with anything allowed between the whitespace-separated parts.'
                ),
            },
        },
        'required': ['query'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the find tool, searching file contents with ripgrep.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects *query*
                (required), optional *path* (subdirectory to limit search), and
                optional *fuzzy* (boolean toggle for pattern matching mode).

        Returns:
            On success, a ``ToolResult.ok`` with the ripgrep output as body and
            metadata including match count. Match paths in the body are relative to
            the project root.  On failure, a ``ToolResult.err`` with an
            appropriate kebab-case error code.
        """
        term: str = kwargs.get('query', '') if isinstance(kwargs.get('query'), str) else ''
        raw_path: str | None = kwargs.get('path')
        fuzzy: bool = kwargs.get('fuzzy', False)

        root = Path.cwd()

        # Resolve optional path argument under project root
        search_dir = root
        relative_path: str | None = None
        if raw_path is not None:
            try:
                search_dir = resolve_in_root(root, raw_path)
            except ValueError:
                return ToolResult.err(
                    f'{raw_path} escapes the project root.',
                    code='path-escapes-root',
                )

            if not search_dir.is_dir():
                return ToolResult.err(
                    f'{raw_path} is not an existing directory.',
                    code='not-a-directory',
                )

            relative_path = str(search_dir.relative_to(root))

        # Locate the ripgrep binary
        rg, rg_error = locate_rg()
        if rg is None:
            return rg_error  # type: ignore[return-value]

        # Build the search pattern and rg arguments
        rg_args: list[str] = [rg, '--line-number', '--no-heading']

        if fuzzy:
            parts = term.split()
            escaped_parts = [re.escape(part) for part in parts]
            pattern = '.*'.join(escaped_parts)
            rg_args.extend(['--ignore-case', pattern])
        else:
            rg_args.extend(['--fixed-strings', term])

        # Respect .gitignore (do NOT pass --no-ignore); exclude .coding_agent
        rg_args.extend(['--glob', IGNORE_GLOB])

        # Always append exactly one path argument: the existing relative_path when
        # the user supplied a path, otherwise the literal "." so ripgrep searches the
        # working directory (not stdin) when no path is given.
        if relative_path is not None:
            rg_args.append(relative_path)
        else:
            rg_args.append(".")

        result = subprocess.run(
            rg_args,
            capture_output=True,
            text=True,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
        )

        # Exit code 0: matches found
        if result.returncode == 0:
            output = result.stdout

            # Searching "." makes rg print each match path with a leading "./" prefix;
            # strip that prefix so reported paths stay clean project-root-relative paths.
            output = '\n'.join(strip_dot_prefix(output.splitlines()))
            match_count = len(output.splitlines())
            return ToolResult.ok(output, match_count=match_count)

        # Exit code 1: no matches
        if result.returncode == 1:
            return ToolResult.ok(
                'No matches found.',
                match_count=0,
            )

        # Any other exit code: rg error
        return ToolResult.err(
            result.stderr.strip() or f'rg exited with code {result.returncode}.',
            code='search-failed',
        )
