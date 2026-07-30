"""Rename-symbol tool: rename a symbol at a position across the whole workspace."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_IDENTIFIER_RE = re.compile(r'^\$?[a-zA-Z_\x80-\xff][a-zA-Z0-9_]*$')

from lsp.manager import LSPUnavailableError, path_to_uri
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


class RenameSymbol(Tool):
    """Requests ``textDocument/rename`` from the language server for a given position.

    The *path* parameter must be relative to the project root and point to an
    existing file inside the sandbox.  *line* and *column* are 1-based.

    Applies the multi-file edit immediately.
    """

    name = 'rename_symbol'
    description = (
        'Rename the symbol at a position across the whole workspace '
        '(like an IDE\'s F2); applies the multi-file edit immediately. '
        'Requires path, line (1-based), column (1-based), and new_name.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root.',
            },
            'line': {
                'type': 'integer',
                'description': '1-based line number.',
            },
            'column': {
                'type': 'integer',
                'description': '1-based column number.',
            },
            'new_name': {
                'type': 'string',
                'description': 'The new symbol name.',
            },
        },
        'required': ['path', 'line', 'column', 'new_name'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the rename-symbol tool for the given position.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Requires ``path``
                (str), ``line`` (int), ``column`` (int), and ``new_name`` (str).

        Returns:
            A ``ToolResult`` with confirmation of the rename on success,
            or an error when validation fails or the language server is unavailable.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''
        line_raw = kwargs.get('line')
        column_raw = kwargs.get('column')
        new_name_raw = kwargs.get('new_name')

        # --- validate new_name -------------------------------------------------
        if (
            not isinstance(new_name_raw, str)
            or not new_name_raw.strip()
            or not _IDENTIFIER_RE.match(new_name_raw)
        ):
            return ToolResult.err(
                f'new_name must be a non-empty plain identifier (letters/digits/underscore, not starting with a digit), got {new_name_raw!r}.',
                code='bad-arguments',
            )

        # --- validate path ---------------------------------------------------
        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.err(
                f'{resolved} does not exist or is not a regular file.',
                code='not-a-file',
            )

        # --- validate line and column ----------------------------------------
        for label, value in [('line', line_raw), ('column', column_raw)]:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                return ToolResult.err(
                    f'{label} must be a positive integer, got {value!r}.',
                    code='bad-arguments',
                )

        line: int = line_raw  # type: ignore[assignment]
        column: int = column_raw  # type: ignore[assignment]

        # --- language server availability ------------------------------------
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.MANAGER is None:
            return ToolResult.err(
                'no language servers are running',
                code='lsp-unavailable',
            )

        MANAGER = main_module.MANAGER
        language = MANAGER.language_for_path(str(resolved))
        if language is None:
            return ToolResult.err(
                f'no language server configured for this file type '
                f'({Path(raw_path).suffix})',
                code='lsp-unavailable',
                hint=f'Try get_diagnostics for a supported file type.',
            )

        try:
            client = MANAGER.get_client(language)
        except LSPUnavailableError as exc:
            return ToolResult.err(
                str(exc),
                code='lsp-unavailable',
            )

        # --- ensure document is open/synced ----------------------------------
        uri = path_to_uri(str(resolved))
        if uri not in MANAGER._open_docs:
            MANAGER._did_open(str(resolved))

        # --- try textDocument/prepareRename (ignore failures — many servers don\'t implement it) ---
        _prepare_failed = False
        _prepare_result: Any = None
        try:
            _prepare_result = client.request(
                'textDocument/prepareRename',
                {'textDocument': {'uri': uri}, 'position': {'line': line - 1, 'character': column - 1}},
                timeout=10.0,
            )
        except Exception:
            _prepare_failed = True

        if not _prepare_failed and _prepare_result is None:
            return ToolResult.err(
                f'cannot rename the symbol at {raw_path}:{line}:{column}',
                code='bad-arguments',
            )

        METHOD = 'textDocument/rename'
        try:
            result = client.request(
                METHOD,
                {
                    'textDocument': {'uri': uri},
                    'position': {'line': line - 1, 'character': column - 1},
                    'newName': new_name_raw,
                },
                timeout=15.0,
            )
        except TimeoutError:
            return ToolResult.err(
                'language server did not answer in time',
                code='lsp-timeout',
            )
        except Exception as exc:
            return ToolResult.err(
                f'the {language} language server does not '
                f'support {METHOD} (or the request failed: {exc})',
                code='lsp-capability',
            )

        if result is None or (isinstance(result, dict) and not result.get('changes') and not result.get('documentChanges')):
            return ToolResult.ok(f'nothing to rename at {raw_path}:{line}:{column}')

        # --- apply workspace edit --------------------------------------------
        from lsp.edits import WorkspaceEditError, apply_workspace_edit  # pylint: disable=import-outside-toplevel

        try:
            files = apply_workspace_edit(result, str(Path.cwd()))
        except WorkspaceEditError as exc:
            return ToolResult.err(str(exc), code='edit-failed')

        body = f'renamed to {new_name_raw} in {len(files)} file(s):\n'
        for rel in files:
            body += f'{rel}\n'

        return ToolResult.ok(body, count=len(files), new_name=new_name_raw)
