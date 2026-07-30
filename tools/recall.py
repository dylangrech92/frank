"""Recall tool: search project memory for facts relevant to a query."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class Recall(Tool):
    """Search the project's persistent memory for relevant facts.

    Performs a hybrid semantic + keyword search over remembered atoms and
    returns the most relevant ones.
    """

    name = 'recall'
    parallel_safe = True  # opens its own fresh store/connection per call (F4)
    description = (
        'Search the project persistent memory for facts relevant to a '
        'natural-language query. Returns the most relevant remembered atoms '
        '(hybrid semantic + keyword search).'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'A natural-language description of what you are looking for.',
            },
            'limit': {
                'type': 'integer',
                'description': 'Maximum number of memories to return (default 10).',
            },
        },
        'required': ['query'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the recall tool."""
        query_raw = kwargs.get('query') if isinstance(kwargs.get('query'), str) else ''
        if not query_raw:
            return ToolResult.err(
                'The "query" argument is required and must be a non-empty string.',
                code='missing-argument',
            )

        limit = kwargs.get('limit', 10)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10
        limit = max(1, min(50, limit))

        results = _recall_fresh(query_raw, limit=limit)
        if not results:
            return ToolResult.ok('No relevant memories found.', count=0)

        body = '\n'.join(str(r['text']) for r in results)
        return ToolResult.ok(body, count=len(results))


def _recall_fresh(query: str, limit: int) -> list[dict]:
    """Hybrid recall across all registered layers using a store opened fresh for this call.

    ``memory.recall.recall()`` rides ``get_memory()``'s cached main-thread SQLite
    connection, which is unsafe to touch from a worker thread. This mirrors that
    function's logic but opens (and closes) its own :class:`MemoryStore` and
    :class:`EmbeddingService` on whichever thread calls it -- the same
    fresh-store-per-thread pattern used by ``main._memory_maintenance`` and the
    episodic writer (``memory/episodic.py``) -- so the tool can join a parallel
    batch.

    Args:
        query: Natural-language search string.
        limit: Maximum number of results to return.

    Returns:
        Up to *limit* result dicts (each with at least ``text`` and ``score``),
        sorted by score descending.
    """
    import os

    from memory.embedding import EmbeddingService
    from memory.recall import MemoryContext, _LAYERS, _ensure_layers_registered
    from memory.store import open_store

    project_root = os.path.abspath(os.getcwd())
    store = open_store(project_root)
    try:
        model_path = os.environ.get('CODING_AGENT_EMBED_MODEL') or None
        embedder = EmbeddingService(model_path=model_path)
        ctx = MemoryContext(store=store, embedder=embedder, project_root=project_root)
        _ensure_layers_registered()

        results: list[dict] = []
        for layer in _LAYERS:
            try:
                hits = layer['recall'](ctx, query, limit)
            except Exception:
                continue
            if hits:
                results.extend(hits)
        if not results:
            return []
        results.sort(key=lambda d: d['score'], reverse=True)
        return results[:limit]
    finally:
        store.close()
