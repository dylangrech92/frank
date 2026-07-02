"""Delete file tool: deletes a file or an empty directory inside the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._sandbox import emit_mutation, resolve_in_root


class DeleteFile(Tool):
    """Deletes a file or an empty directory inside the project.

    The path must be relative to the project root.  The project root itself
    is never deleted regardless of the path provided.  Non-empty directories are
    rejected to prevent accidental data loss.  Paths that resolve to neither
    a regular file nor a directory (for example, FIFOs) are also rejected.
    """

    name = 'delete_file'
    description = (
        'Deletes a file or an empty directory inside the project. '
        'The path must be relative to the project root.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root of the file or empty directory to delete.',
            },
        },
        'required': ['path'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, deleting a file or an empty directory inside the project.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path``
                (required, relative to project root).

        Returns:
            A ``ToolResult`` naming the deleted path on success, or an error
            when the path escapes root, is the project root itself, does not exist,
            is a non-empty directory, or resolves to neither a regular file nor a
            directory.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''
        root = Path.cwd().resolve()

        # Reject absolute paths and symlink escapes before checking the target
        try:
            resolved = resolve_in_root(root, raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        # Refuse to delete the project root itself
        if resolved == root:
            return ToolResult.err(
                'Cannot delete the project root.',
                code='cannot-delete-root',
            )

        # Reject if the target does not exist
        if not resolved.exists():
            return ToolResult.err(
                f'{raw_path} does not exist.',
                code='not-found',
            )

        # Reject non-empty directories; only empty directories can be removed
        if resolved.is_dir():
            if any(resolved.iterdir()):
                return ToolResult.err(
                    f'{raw_path} is a non-empty directory.',
                    code='directory-not-empty',
                    hint="Only empty directories can be removed.",
                )

        # Reject paths that are neither regular files nor directories (e.g. FIFOs)
        if resolved.exists() and not resolved.is_file() and not resolved.is_dir():
            return ToolResult.err(
                f'{raw_path} is neither a regular file nor a directory and cannot be deleted.',
                code='unsupported-file-type',
            )

        # Remove the file or empty directory
        if resolved.is_file():
            resolved.unlink()
        elif resolved.is_dir():
            resolved.rmdir()

        # Emit the mutation event (exactly once on success)
        emit_mutation('deleted', resolved)

        return ToolResult.ok(
            f'Deleted: {raw_path}',
            path=raw_path,
        )
