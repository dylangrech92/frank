"""Replace one tool: replaces exactly one occurrence of a literal string in a single file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._sandbox import emit_mutation, resolve_in_root


class ReplaceOne(Tool):
    """Replaces exactly one occurrence of a literal string in a single file.

    Refuses to perform the replacement when the search string matches zero times or
    more than once -- every match must be unique for safety.  The path must be
    relative to the project root.  Only regular files can be modified; directories
    and other special paths are rejected.
    """

    name = 'replace_one'
    description = (
        'Replaces exactly one occurrence of a literal string in a single file. '
        'Refuses when the match is not unique. The path must be relative to the project root.'
    )
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
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''
        search = kwargs.get('search', '')
        replace = kwargs.get('replace', '')

        # Resolve to absolute path under project root
        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        # Reject if the target does not exist
        if not resolved.exists():
            return ToolResult.err(
                f'{raw_path} does not exist.',
                code='not-found',
                hint="Use create_file to create a new file.",
            )

        # Reject if the target is not a regular file
        if not resolved.is_file():
            return ToolResult.err(
                f'{raw_path} is not a regular file.',
                code='not-a-file',
            )

        # Read the file as UTF-8
        content = resolved.read_text(encoding='utf-8')

        # Count occurrences of the search string
        count = content.count(search)

        if count == 0:
            return ToolResult.err(
                f'The search string was not found in {raw_path}.',
                code='no-match',
                hint='Check the exact text with read_file.',
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

        # Emit the mutation event (exactly once on success)
        emit_mutation('changed', resolved)

        return ToolResult.ok(
            f'Replaced 1 occurrence in {raw_path}.',
            occurrences=1,
        )
