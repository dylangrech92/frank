"""Document-symbols tool: return the outline of a file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lsp.manager import LSPUnavailableError, path_to_uri
from lsp.locations import flatten_symbols, render_symbol_line
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


class DocumentSymbols(Tool):
    """Return the outline (classes, functions, methods) of a file.

    The *path* parameter must be relative to the project root and point to an
    existing file inside the sandbox.  Emits ``textDocument/documentSymbol`` to
    the language server for that file.
    """

    name = 'document_symbols'
    summary = 'Get the outline (classes/functions) of a file.'
    description = (
        'Returns the outline (classes, functions, methods) of a file. '
        'Requires path relative to the project root.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root.',
            },
        },
        'required': ['path'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the document-symbols tool for the given file.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Requires ``path`` (str).

        Returns:
            A ``ToolResult`` with symbol lines as its body on success, or an
            error when validation fails or the language server is unavailable.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''

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

        METHOD = 'textDocument/documentSymbol'
        try:
            result = client.request(
                METHOD,
                {'textDocument': {'uri': uri}},
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

        syms = flatten_symbols(result)
        if not syms:
            return ToolResult.ok(f'no symbols in {raw_path}')

        lines = [render_symbol_line(s, str(MANAGER._root_path)) for s in syms]
        rel = os.path.relpath(str(resolved), Path.cwd())

        return ToolResult.ok(
            '\n'.join(lines),
            path=rel,
            count=len(syms),
        )
