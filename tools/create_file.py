"""Create file tool: writes a new file with the given content inside the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._sandbox import emit_mutation, resolve_in_root


class CreateFile(Tool):
    """Creates a new file with the given content inside the project.

    The path must be relative to the project root.  Parent directories are
    created as needed.  If the target path already exists an error is returned;
    use update_file to overwrite.
    """

    name = 'create_file'
    description = (
        'Creates a new file with the given content inside the project. '
        'The path must be relative to the project root.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root where the file should be created.',
            },
            'content': {
                'type': 'string',
                'description': 'Content to write to the file. Defaults to an empty string.',
            },
        },
        'required': ['path'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, creating a new file with the given content.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path``
                (required, relative to project root) and optional ``content``
                (defaults to empty string).

        Returns:
            A ``ToolResult`` naming the created path on success, or an error
            when the path escapes root or already exists.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''
        content = kwargs.get('content', '')

        # Resolve to absolute path under project root
        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        # Reject if the target already exists
        if resolved.exists():
            return ToolResult.err(
                f'{raw_path} already exists.',
                code='already-exists',
                hint="Use update_file to overwrite an existing file.",
            )

        # Create parent directories as needed
        resolved.parent.mkdir(parents=True, exist_ok=True)

        # Write content as UTF-8
        resolved.write_text(content, encoding='utf-8')

        # Emit the mutation event (exactly once on success)
        emit_mutation('created', resolved)

        return ToolResult.ok(
            f'File created: {raw_path}',
            path=raw_path,
            bytes_written=len(content.encode('utf-8')),
        )

