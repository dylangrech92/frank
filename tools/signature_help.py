"""Signature help: get parameter hints for a function call at a position."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lsp.manager import LSPUnavailableError, path_to_uri
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


class SignatureHelp(Tool):
    """Requests ``textDocument/signatureHelp`` from the language server.

    The *path* parameter must be relative to the project root and point to an
    existing file inside the sandbox.  *line* and *column* are 1-based.

    Returns parameter/hover hints for a function call at the given position,
    typically placed inside the call parentheses.
    """

    name = 'signature_help'
    summary = 'Get parameter hints for a function call at a position.'
    description = (
        'Get parameter hints for a function call at a position '
        '(line/column 1-based, typically placed inside the call parentheses).'
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
        """Execute the signature help tool for the given position.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Requires ``path``
                (str), ``line`` (int), and ``column`` (int).

        Returns:
            A ``ToolResult`` with signature help output on success,
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
        column: int = column_raw  #type: ignore[assignment]

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

        METHOD = 'textDocument/signatureHelp'
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

        # --- handle result ---------------------------------------------------
        if result is None or not isinstance(result, dict) or not result.get('signatures'):
            return ToolResult.ok(
                f'no signature help at {raw_path}:{line}:{column}'
            )

        signatures: list[dict[str, Any]] = result['signatures']
        active_signature_idx: int = result.get('activeSignature', 0) or 0
        if not 0 <= active_signature_idx < len(signatures):
            active_signature_idx = 0

        active_parameter_idx: int | None = result.get('activeParameter')

        lines: list[str] = []
        for idx, sig in enumerate(signatures):
            label = str(sig.get('label', ''))
            if idx == active_signature_idx:
                lines.append(f'> {label}')
            else:
                lines.append(f'  {label}')

        # ACTIVE signature details
        active_sig = signatures[active_signature_idx]
        params = active_sig.get('parameters', [])

        if (
            active_parameter_idx is not None
            and 0 <= active_parameter_idx < len(params)
        ):
            param = params[active_parameter_idx]
            param_label = param.get('label', '')
            if isinstance(param_label, list) and len(param_label) == 2:
                start, end = param_label
                param_text = str(active_sig['label'])[start:end]
            else:
                param_text = str(param_label)
            lines.append(f'  active parameter: {param_text}')

        if 'documentation' in active_sig:
            doc = active_sig['documentation']
            if isinstance(doc, dict):
                doc = doc.get('value', '')
            lines.append(f'  {doc}')

        body = '\n'.join(lines)
        rel = os.path.relpath(str(resolved), Path.cwd())

        return ToolResult.ok(body, path=rel, line=line, column=column, count=len(signatures))
