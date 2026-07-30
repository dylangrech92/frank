"""Move file tool: renames or moves a file or directory inside the project via plain filesystem operation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from tools._sandbox import emit_mutation, resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


def pre_move_hook(source: Path, destination: Path) -> list[str]:
    """Fire workspace/willRenameFiles on the source file's language server and apply any returned import-fixing WorkspaceEdit before the filesystem move; returns human-readable note lines (edited files or a degrade warning), empty when there is nothing to report."""
    try:
        import main as main_module

        if main_module.MANAGER is None:
            return []

        MANAGER = main_module.MANAGER
        language = MANAGER.language_for_path(str(source))
        if language is None:
            return []

        from lsp.manager import LSPUnavailableError, path_to_uri

        try:
            client = MANAGER.get_client(language)
        except LSPUnavailableError:
            return []

        capabilities = client.server_capabilities.get('capabilities', {}).get('workspace', {}).get('fileOperations', {}).get('willRename') if isinstance(client.server_capabilities, dict) else None
        if not capabilities:
            return [f'note: the {language} language server does not support willRenameFiles; imports were NOT updated']

        # Inferred-project servers (ts-ls) only consider open documents, so
        # surface every same-language workspace file before asking for edits.
        _skip_dirs = {'.git', 'node_modules', '__pycache__', '.coding_agent'}
        opened = 0
        for dirpath, dirnames, filenames in os.walk(str(Path.cwd())):
            dirnames[:] = [d for d in dirnames if d not in _skip_dirs and not d.startswith('.')]
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if MANAGER.language_for_path(fpath) != language:
                    continue
                if path_to_uri(fpath) in MANAGER._open_docs:
                    continue
                MANAGER._did_open(fpath)
                opened += 1
                if opened >= 30:
                    break
            if opened >= 30:
                break

        params = {'files': [{'oldUri': path_to_uri(str(source)), 'newUri': path_to_uri(str(destination))}]}
        try:
            result = client.request('workspace/willRenameFiles', params, timeout=10.0)
        except Exception as exc:
            return [f'note: willRenameFiles failed ({exc}); imports were NOT updated']

        if result is None or (isinstance(result, dict) and not result.get('changes') and not result.get('documentChanges')):
            return []

        from lsp.edits import WorkspaceEditError, apply_workspace_edit

        try:
            files = apply_workspace_edit(result, str(Path.cwd()))
        except WorkspaceEditError as exc:
            return [f'note: could not apply import updates ({exc}); moving anyway']

        return [f'updated imports in {rel}' for rel in files]
    except Exception as exc:
        return [f'note: import update skipped ({exc})']


class MoveFile(Tool):
    """Moves or renames a file or directory inside the project.

    Uses a plain filesystem move that updates imports via the language server where supported.  Both the source and
    destination must be given as paths relative to the project root.  The
    source must exist and the destination must not already exist.
    """

    name = 'move_file'
    description = (
        'Moves or renames a file or directory inside the project (plain filesystem move, '
        'updates imports via the language server where supported). Paths must be relative to the project root.'
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

        hook_notes = pre_move_hook(resolved_source, resolved_destination)

        shutil.move(str(resolved_source), str(resolved_destination))

        emit_mutation('renamed', resolved_destination, extra={'old_path': str(resolved_source)})

        body = f'Moved {raw_source} -> {raw_destination}'
        for note in hook_notes:
            body += f'\n{note}'
        return ToolResult.ok(
            body,
            path=raw_destination,
        )
