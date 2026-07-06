"""Get diagnostics tool: reports language-server diagnostics for the project or a file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from diagnostics import STORE
from lsp.manager import uri_to_path
from tools.base import Tool
from tools.result import ToolResult


class GetDiagnostics(Tool):
    """Reports LSP diagnostics for every open file in the sandbox.

    When *path* is given, only diagnostics whose resolved path ends with that
    string are included.

    The body is one line per diagnostic::

        <relative-path>:<line>:<col> <severity-word> <message>

    followed by a summary line ``-- N diagnostics``.  Meta carries the total
    count as ``{'count': N}``.
    """

    name = 'get_diagnostics'
    summary = 'Report current LSP diagnostics for a file or project.'
    description = (
        'Report current language-server diagnostics for the project or a single file. '
        'When path is given, restrict to diagnostics whose path ends with that string.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Optional file path suffix to filter diagnostics (e.g. "main.py"). '
                               'When absent, report all diagnostics.',
            },
        },
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, returning current LSP diagnostics.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Expects optional
                ``path`` (str) to filter diagnostics by path suffix.

        Returns:
            A ``ToolResult`` with formatted diagnostic lines as its body on success,
            or an error when the language server is unavailable (code ``lsp-unavailable``).
        """
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.MANAGER is None:
            return ToolResult.err(
                'no language servers are running',
                code='lsp-unavailable',
            )

        file_filter = kwargs.get('path')
        pairs = STORE.full(file_filter=file_filter)

        if not pairs:
            if file_filter:
                return ToolResult.ok(f'no diagnostics for {file_filter!r}')
            return ToolResult.ok('no diagnostics')

        root = Path(main_module.MANAGER._root_path)
        severity_map = {1: 'error', 2: 'warning', 3: 'info', 4: 'hint'}

        lines: list[str] = []
        for uri, diagnostic in pairs:
            try:
                path = uri_to_path(uri)
            except Exception:
                path = uri
            try:
                rel = os.path.relpath(path, root)
            except ValueError:  # pragma: no cover -- Windows edge case
                rel = path

            rng = diagnostic.get('range', {})
            start = rng.get('start', {})
            line = start.get('line', 0) or 0
            col = start.get('character', 0) or 0
            severity_code = diagnostic.get('severity') or 1
            severity_word = severity_map.get(severity_code, 'error')
            message = diagnostic.get('message', '')

            lines.append(f'{rel}:{line + 1}:{col + 1} {severity_word} {message}')

        n = len(lines)
        lines.append(f'-- {n} diagnostics')

        return ToolResult.ok('\n'.join(lines), count=n)
