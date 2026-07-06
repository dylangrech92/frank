"""Create folder tool: creates a new directory inside the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._sandbox import emit_mutation, resolve_in_root


class CreateFolder(Tool):
    """Creates a directory inside the project.

    The path must be relative to the project root.  Parent directories are
    created as needed.
    """

    name = 'create_folder'
    summary = 'Create a new directory (and parents).'
    description = (
        'Creates a directory inside the project. '
        'The path must be relative to the project root.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root where the directory should be created.',
            },
        },
        'required': ['path'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, creating a new directory.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path``
                (required, relative to project root).

        Returns:
            A ``ToolResult`` indicating success or an error when the target
            is not a directory path.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''

        # Resolve to absolute path under project root
        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        # If the target exists and is a directory, report success (idempotent)
        if resolved.is_dir():
            return ToolResult.ok(
                f'Directory already exists: {raw_path}',
                path=raw_path,
                created=False,
            )

        # If it exists but is not a directory, error out
        if resolved.exists():
            return ToolResult.err(
                f'{raw_path} exists but is not a directory.',
                code='not-a-directory',
            )

        # Create the directory with all parent directories
        resolved.mkdir(parents=True)

        # Emit the mutation event (exactly once on success)
        emit_mutation('created', resolved)

        return ToolResult.ok(
            f'Directory created: {raw_path}',
            path=raw_path,
            created=True,
        )

