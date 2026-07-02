"""Update file tool: overwrites an existing file with entirely new content inside the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._sandbox import emit_mutation, resolve_in_root


class UpdateFile(Tool):
    """Overwrites an existing file with entirely new content (full overwrite, not a patch).

    The path must be relative to the project root.  Only regular files can be
    updated; directories and other special paths are rejected.
    """

    name = 'update_file'
    description = (
        'Overwrites an existing file with entirely new content (full overwrite, not a patch). '
        'The path must be relative to the project root.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root of the file to overwrite.',
            },
            'content': {
                'type': 'string',
                'description': 'New content to write to the file, replacing all existing content.',
            },
        },
        'required': ['path', 'content'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, overwriting an existing file with entirely new content.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path``
                (required, relative to project root) and ``content`` (required).

        Returns:
            A ``ToolResult`` naming the updated path on success, or an error
            when the path escapes root, the file does not exist, or the target
            is not a regular file.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''
        content = kwargs.get('content', '')

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

        # Write content as UTF-8
        resolved.write_text(content, encoding='utf-8')

        # Emit the mutation event (exactly once on success)
        emit_mutation('changed', resolved)

        return ToolResult.ok(
            f'File updated: {raw_path}',
            path=raw_path,
            bytes_written=len(content.encode('utf-8')),
        )
