"""Move file tool: renames or moves a file or directory inside the project via plain filesystem operation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from tools._sandbox import emit_mutation, resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


def pre_move_hook(source: Path, destination: Path) -> None:
    """No-op seam for a later phase.

    A later phase wires the LSP workspace/prepareSupportDefaultBehavior =
    true, workspace/preRenameFiles (or workspace/willRenameFiles) request
    here so language servers can prepare import updates before the move takes
    place.

    Args:
        source: The resolved absolute source path.
        destination: The resolved absolute destination path.
    """


class MoveFile(Tool):
    """Moves or renames a file or directory inside the project.

    Uses a plain filesystem move with no import fixing.  Both the source and
    destination must be given as paths relative to the project root.  The
    source must exist and the destination must not already exist.
    """

    name = 'move_file'
    description = (
        'Moves or renames a file or directory inside the project (plain filesystem move, '
        'no import fixing). Paths must be relative to the project root.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Current location of the file or directory relative to the project root.',
            },
            'destination': {
                'type': 'string',
                'description': 'New location for the file or directory relative to the project root.',
            },
        },
        'required': ['path', 'destination'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, moving or renaming a file or directory.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path``
                (required, source relative to project root) and ``destination``
                (required, target relative to project root).

        Returns:
            A ``ToolResult`` describing the successful move on success, or an
            error when either path escapes root, the source does not exist, or
            the destination already exists.
        """
        raw_source = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''
        raw_destination = kwargs.get('destination') if isinstance(kwargs.get('destination'), str) else ''

        try:
            resolved_source = resolve_in_root(Path.cwd(), raw_source)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        try:
            resolved_destination = resolve_in_root(Path.cwd(), raw_destination)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        if not resolved_source.exists():
            return ToolResult.err(
                f'{raw_source} does not exist.',
                code='not-found',
            )

        if resolved_destination.exists():
            return ToolResult.err(
                f'{raw_destination} already exists.',
                code='already-exists',
                hint='Delete it first or choose another name.',
            )

        resolved_destination.parent.mkdir(parents=True, exist_ok=True)

        pre_move_hook(resolved_source, resolved_destination)

        shutil.move(str(resolved_source), str(resolved_destination))

        emit_mutation('renamed', resolved_destination, extra={'old_path': str(resolved_source)})

        return ToolResult.ok(
            f'Moved {raw_source} -> {raw_destination}',
            path=raw_destination,
        )
