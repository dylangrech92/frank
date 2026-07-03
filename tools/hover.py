"""Hover tool: get type/signature information for a symbol at a position in a file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lsp.manager import LSPUnavailableError, path_to_uri
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


class Hover(Tool):
    """Requests ``textDocument/hover`` from the language server for a given position.

    The *path* parameter must be relative to the project root and point to an
    existing file inside the sandbox.  *line* and *column* are 1-based.

    Returns the hover body text (markdown or plaintext) along with the position
    in metadata.
    """

    name = 'hover'
    description = (
        'Get type/signature information for the symbol at a position in a file. '
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
        """Execute the hover tool for the given position.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Requires ``path``
                (str), ``line`` (int), and ``column`` (int).

        Returns:
            A ``ToolResult`` with hover text as its body on success, or an error
            when validation fails or the language server is unavailable.
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

        # --- send hover request ----------------------------------------------
        try:
            result = client.request(
                'textDocument/hover',
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

        # --- extract result --------------------------------------------------
        if not result:
            return ToolResult.ok(
                f'no hover information at {raw_path}:{line}:{column}'
            )

        contents = result.get('contents')
        body = _extract_contents(contents) or 'no hover information at {0}:{1}:{2}'.format(raw_path, line, column)

        rel = os.path.relpath(str(resolved), Path.cwd())

        return ToolResult.ok(body, path=rel, line=line, column=column)


def _extract_contents(contents: Any) -> str | None:
    """Return plain text from an LSP *contents* value.

    The contents may be a string (plaintext), a dict with ``"value"``
    (markdown/plaintext object), or a list of either.

    Args:
        contents: Raw ``textDocument/hover`` contents field.

    Returns:
        Concatenated text joined by blank lines, or ``None`` when *contents*
        is None or empty.
    """
    if isinstance(contents, str):
        return contents

    if isinstance(contents, dict):
        value = contents.get('value', '')
        if value:
            return str(value)
        return None

    if isinstance(contents, (list, tuple)):
        parts: list[str] = []
        for item in contents:
            text: str | None = None
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = str(item.get('value', ''))
            if text:
                parts.append(text)
        return '\n\n'.join(parts) if parts else None

    return None
