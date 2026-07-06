"""record_spec tool: persist a spec node into the data graph."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class RecordSpec(Tool):
    """Record a specification: a title, its body, and optional acceptance/status."""

    name = 'record_spec'
    summary = 'Record a specification for a feature/component.'
    description = (
        'Record a specification for a feature or component: a short title, the spec '
        'body, and optionally acceptance criteria and a status (e.g. draft, '
        'accepted, done).'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'title': {'type': 'string', 'description': 'A short name for the spec.'},
            'body': {'type': 'string', 'description': 'The specification text.'},
            'acceptance': {'type': 'string', 'description': 'Optional acceptance criteria.'},
            'status': {'type': 'string', 'description': 'Optional status (e.g. draft, accepted, done).'},
        },
        'required': ['title', 'body'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        title = kwargs.get('title') if isinstance(kwargs.get('title'), str) else ''
        body = kwargs.get('body') if isinstance(kwargs.get('body'), str) else ''
        if not title.strip():
            return ToolResult.err('The "title" argument is required.', code='missing-argument')
        if not body.strip():
            return ToolResult.err('The "body" argument is required.', code='missing-argument')

        extra: dict[str, Any] = {}
        acceptance = kwargs.get('acceptance')
        status = kwargs.get('status')
        if isinstance(acceptance, str) and acceptance.strip():
            extra['acceptance'] = acceptance
        if isinstance(status, str) and status.strip():
            extra['status'] = status

        from memory.recall import get_memory
        from memory import graph

        try:
            ctx = get_memory()
            node_id = graph.create_node(ctx, 'spec', title, body, extra=extra or None)
        except Exception as exc:
            return ToolResult.err(f'Failed to record spec: {exc}', code='graph-error')

        return ToolResult.ok(f'Recorded spec [{node_id}] {title}.', node_id=node_id, node_type='spec')
