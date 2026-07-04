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

        from memory.recall import recall as _recall  # pylint: disable=import-outside-toplevel

        results = _recall(query_raw, limit=limit)
        if not results:
            return ToolResult.ok('No relevant memories found.', count=0)

        body = '\n'.join(str(r['text']) for r in results)
        return ToolResult.ok(body, count=len(results))
