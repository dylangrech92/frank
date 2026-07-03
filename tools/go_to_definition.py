"""Go-to-definition tool: navigate to the definition of a symbol at a position."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lsp.manager import LSPUnavailableError, path_to_uri
from lsp.locations import render_locations
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


class GoToDefinition(Tool):
    """Requests ``textDocument/definition`` from the language server for a given position.

    The *path* parameter must be relative to the project root and point to an
    existing file inside the sandbox.  *line* and *column* are 1-based.

    Returns the location(s) of the definition.
    """

    name = 'go_to_definition'
    description = (
        'Navigate to the definition of the symbol at a position in a file. '
        'Requires path, line (1-based), and column (1-based).'
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
        },
        'required': ['path', 'line', 'column'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the go-to-definition tool for the given position.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Requires ``path``
                (str), ``line`` (int), and ``column`` (int).

        Returns:
            A ``ToolResult`` with definition location(s) as its body on success,
            or an error when validation fails or the language server is unavailable.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''
        line_raw = kwargs.get('line')
        column_raw = kwargs.get('column')

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

        METHOD = 'textDocument/definition'
        try:
            result = client.request(
                METHOD,
                {
                    'textDocument': {'uri': uri},
                    'position': {'line': line - 1, 'character': column - 1},
                },
                timeout=10.0,
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

        if result is None or isinstance(result, list) and len(result) == 0:
            return ToolResult.ok(
                f'no definition found at {raw_path}:{line}:{column}'
            )

        lines = render_locations(result, str(MANAGER._root_path))
        if not lines:
            return ToolResult.ok(
                f'no definition found at {raw_path}:{line}:{column}'
            )

        body = '\n'.join(lines)

        rel = os.path.relpath(str(resolved), Path.cwd())

        return ToolResult.ok(body, path=rel, line=line, column=column, count=len(lines))
