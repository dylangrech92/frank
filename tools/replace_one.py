"""Replace one tool: replaces exactly one occurrence of a literal string in a single file."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._edit import (
    finalize_write,
    freshness_gate,
    looks_line_numbered,
    resolve_existing_file,
)


class ReplaceOne(Tool):
    """Replaces exactly one occurrence of a literal string in a single file.

    Refuses to perform the replacement when the search string matches zero times or
    more than once -- every match must be unique for safety.  The path must be
    relative to the project root.  Only regular files can be modified; directories
    and other special paths are rejected.
    """

    name = 'replace_one'
    summary = 'Replace one unique occurrence of a literal string in a file.'
    description = (
        'Replaces exactly one occurrence of a literal string in a single file. '
        'Refuses when the match is not unique. The search string must be the raw file text — '
        "do not include read_file's display-only line-number prefixes (the '     1\\t' column). "
        'The path must be relative to the project root.'
    )
    alternative = 'replace_many or update_file'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root of the file to edit.',
            },
            'search': {
                'type': 'string',
                'description': 'The exact literal text to find in the file.',
            },
            'replace': {
                'type': 'string',
                'description': 'The replacement text.',
            },
        },
        'required': ['path', 'search', 'replace'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, replacing exactly one occurrence of a literal string.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path``
                (required, relative to project root), ``search`` (required, the exact
                literal text to find), and ``replace`` (required, the replacement text).

        Returns:
            A ``ToolResult`` describing the single replacement on success, or an error
            when the path escapes root, the file does not exist/is not a file, or the
            match count is zero or greater than one.
        """
        path_arg = kwargs.get('path')
        raw_path = path_arg if isinstance(path_arg, str) else ''
        search = kwargs.get('search', '')
        replace = kwargs.get('replace', '')

        # Resolve under root and confirm the target is an existing regular file.
        resolved, error = resolve_existing_file(raw_path)
        if error is not None:
            return error
        assert resolved is not None

        # Refuse to edit against a stale or never-read view of the file --
        # another process may have changed it since this session last saw it.
        stale_error = freshness_gate(resolved, raw_path)
        if stale_error is not None:
            return stale_error

        # Read the file as UTF-8
        content = resolved.read_text(encoding='utf-8')

        # Count occurrences of the search string
        count = content.count(search)

        if count == 0:
            hint = (
                "Check the exact text with read_file. The search text must not include "
                "read_file's display-only line-number prefixes (the '     1\\t' column)."
            )
            if looks_line_numbered(search):
                hint = (
                    "Your search text includes read_file's line-number prefixes "
                    "(the '     1\\t' column) — those are display-only. Strip them so "
                    "the search matches the real file content."
                )
            return ToolResult.err(
                f'The search string was not found in {raw_path}.',
                code='no-match',
                hint=hint,
            )

        if count > 1:
            return ToolResult.err(
                (
                    f'Found {count} occurrences of the search string in {raw_path}; '
                    f'replacement was refused because the match must be unique.'
                ),
                code='ambiguous-match',
                hint='Include more surrounding context in the search string.',
            )

        # Exactly one occurrence: perform the replacement
        new_content = content.replace(search, replace, 1)
        resolved.write_text(new_content, encoding='utf-8')

        # Re-stamp the read registry and emit exactly one mutation event.
        finalize_write(resolved)

        return ToolResult.ok(
            f'Replaced 1 occurrence in {raw_path}.',
            occurrences=1,
        )
