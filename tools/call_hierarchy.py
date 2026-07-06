"""Call-hierarchy tool: inspect callers or callees of a symbol at a position."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lsp.manager import LSPUnavailableError, path_to_uri
from lsp.locations import format_location, symbol_kind_name
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


class CallHierarchy(Tool):
    """Requests call-hierarchy information from the language server for a given position.

    The *path* parameter must be relative to the project root and point to an
    existing file inside the sandbox.  *line* and *column* are 1-based.
    *direction* specifies whether to find callers (``'incoming'``) or callees
    (``'outgoing'``).

    Returns the call-hierarchy items for the symbol at the given position.
    """

    name = 'call_hierarchy'
    summary = 'Inspect incoming/outgoing callers of a symbol.'
    description = (
        'Inspect the call hierarchy of a symbol at a position in a file. '
        'Requires path, line (1-based), column (1-based), and direction. '
        'Direction ``incoming`` finds callers (who calls this symbol). '
        'Direction ``outgoing`` finds callees (symbols that this one calls).'
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
            'direction': {
                'type': 'string',
                'enum': ['incoming', 'outgoing'],
                'description': (
                    'Call hierarchy direction: ``incoming`` finds callers '
                    '(who calls this symbol), ``outgoing`` finds callees '
                    '(symbols that this one calls).'
                ),
            },
        },
        'required': ['path', 'line', 'column', 'direction'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the call-hierarchy tool for the given position and direction.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Requires ``path``
                (str), ``line`` (int), ``column`` (int), and ``direction`` (str).

        Returns:
            A ``ToolResult`` with call-hierarchy items as its body on success,
            or an error when validation fails or the language server is unavailable.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''
        direction_raw = kwargs.get('direction')
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

        # --- validate direction ----------------------------------------------
        if direction_raw not in ('incoming', 'outgoing'):
            return ToolResult.err(
                f'direction must be "incoming" or "outgoing", got {direction_raw!r}.',
                code='bad-arguments',
            )
        direction: str = direction_raw  # type: ignore[assignment]

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

        # ---- Step 1: prepare call hierarchy ---------------------------------
        prepare_method = 'textDocument/prepareCallHierarchy'
        try:
            prepare_result = client.request(
                prepare_method,
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
                f'support {prepare_method} (or the request failed: {exc})',
                code='lsp-capability',
            )

        if prepare_result is None or (isinstance(prepare_result, list) and len(prepare_result) == 0):
            return ToolResult.ok(
                f'no callable symbol at {raw_path}:{line}:{column}'
            )

        first_item = prepare_result[0]
        item_name = first_item.get('name', 'unknown')

        # ---- Step 2: request incoming or outgoing calls ---------------------
        if direction == 'incoming':
            call_method = 'callHierarchy/incomingCalls'
            header_prefix = 'incoming calls to'
        else:
            call_method = 'callHierarchy/outgoingCalls'
            header_prefix = 'outgoing calls from'

        try:
            call_result = client.request(
                call_method,
                {'item': first_item},
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
                f'support {call_method} (or the request failed: {exc})',
                code='lsp-capability',
            )

        if call_result is None or (isinstance(call_result, list) and len(call_result) == 0):
            if direction == 'incoming':
                return ToolResult.ok(f'no incoming calls for {item_name}')
            else:
                return ToolResult.ok(f'no outgoing calls from {item_name}')

        # ---- Render ---------------------------------------------------------
        lines = [f'{header_prefix} {item_name}:']
        for entry in call_result:
            if not isinstance(entry, dict):
                continue
            key = 'from' if direction == 'incoming' else 'to'
            item = entry.get(key)
            if not isinstance(item, dict):
                continue

            name = item.get('name', 'unknown')
            kind_int = item.get('kind', 0)
            kind_str = symbol_kind_name(kind_int)

            sel_range = item.get('selectionRange', {})
            start = sel_range.get('start', {})
            sel_line = int(start.get('line', 0)) if isinstance(start, dict) else 0
            sel_char = int(start.get('character', 0)) if isinstance(start, dict) else 0

            item_uri = item.get('uri', '')
            location_str = format_location(item_uri, sel_line, sel_char, str(MANAGER._root_path))

            lines.append(f'{name} [{kind_str}] — {location_str}')

        rel = os.path.relpath(str(resolved), Path.cwd())

        return ToolResult.ok(
            '\n'.join(lines),
            path=rel,
            line=line,
            column=column,
            count=len(call_result),
            direction=direction,
        )
