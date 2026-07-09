"""Find-symbol tool: fuzzy-search symbols by name across the whole workspace."""

from __future__ import annotations

from typing import Any

from lsp.locations import flatten_symbols, render_symbol_line
from tools.base import Tool
from tools.result import ToolResult


class FindSymbol(Tool):
    """Fuzzy-search symbols by name across the whole workspace (like IDE ctrl+T).

    Sends ``workspace/symbol`` to every registered language server and merges the
    results.  No path parameter — searches everywhere.
    """

    name = 'find_symbol'
    parallel_safe = True  # read-only LSP query; client-map read is lock-guarded (F3)
    summary = 'Fuzzy-search symbols by name across the workspace.'
    description = (
        'Fuzzy-search symbols by name across the whole workspace (like an IDE\'s '
        'ctrl+T). Searches all open documents in all language servers. Prefer this '
        'over paging through large files with read_file when you know the name you '
        'are looking for.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Symbol name or fragment to search for.',
            },
        },
        'required': ['query'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the find-symbol tool.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Requires ``query`` (str).

        Returns:
            A ``ToolResult`` with matching symbol lines as its body on success, or an
            error when validation fails or the language server is unavailable.
        """
        raw_query = kwargs.get('query')
        if not isinstance(raw_query, str) or not raw_query.strip():
            return ToolResult.err(
                'query must be a non-empty string.',
                code='bad-arguments',
            )

        # --- language server availability ------------------------------------
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.MANAGER is None:
            return ToolResult.err(
                'no language servers are running',
                code='lsp-unavailable',
            )

        MANAGER = main_module.MANAGER
        # Thread-safe snapshot: plain dict iteration would race a concurrent
        # spawn mutating _clients under _spawn_lock (F3).
        clients = MANAGER.snapshot_clients()
        if not clients:
            return ToolResult.err(
                'no language servers are running',
                code='lsp-unavailable',
            )

        # Deduplicate by client identity, sorted by language name.
        seen: set[int] = set()
        deduped: list[tuple[str, Any]] = []
        for lang, client in clients:
            cid = id(client)
            if cid not in seen:
                seen.add(cid)
                deduped.append((lang, client))
        deduped.sort(key=lambda pair: pair[0])

        # --- query each client -----------------------------------------------
        merged: list[Any] = []
        for lang, client in deduped:
            try:
                result = client.request(
                    'workspace/symbol',
                    {'query': raw_query},
                    timeout=10.0,
                )
                if isinstance(result, list):
                    merged.extend(result)
            except Exception:
                pass

        syms = flatten_symbols(merged)
        if not syms:
            return ToolResult.ok(f'no symbols matching {raw_query!r}')

        lines = [render_symbol_line(s, str(MANAGER._root_path)) for s in syms]
        n = len(syms)
        singular = 'symbol' if n == 1 else 'symbols'
        footer = f'-- {n} {singular}'
        body = '\n'.join(lines + [footer])

        return ToolResult.ok(body, count=n)
