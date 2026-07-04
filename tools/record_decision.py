"""record_decision tool: persist a decision (with alternatives) into the data graph."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class RecordDecision(Tool):
    """Record a design decision, its rationale, and the alternatives considered."""

    name = 'record_decision'
    description = (
        'Record a design/architecture decision: a short title, the rationale, and '
        'optionally the alternatives considered and the id of a node this decision '
        'implements (adds an implements edge).'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'title': {'type': 'string', 'description': 'A short name for the decision.'},
            'rationale': {'type': 'string', 'description': 'Why this decision was made.'},
            'alternatives': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Other options that were considered.',
            },
            'implements': {
                'type': 'integer',
                'description': 'Optional id of a node this decision implements (adds an implements edge).',
            },
        },
        'required': ['title', 'rationale'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        title = kwargs.get('title') if isinstance(kwargs.get('title'), str) else ''
        rationale = kwargs.get('rationale') if isinstance(kwargs.get('rationale'), str) else ''
        if not title.strip():
            return ToolResult.err('The "title" argument is required.', code='missing-argument')
        if not rationale.strip():
            return ToolResult.err('The "rationale" argument is required.', code='missing-argument')

        alternatives = kwargs.get('alternatives')
        if isinstance(alternatives, str):
            alternatives = [a.strip() for a in alternatives.split(',') if a.strip()]
        elif isinstance(alternatives, list):
            alternatives = [str(a) for a in alternatives]
        else:
            alternatives = []
        extra = {'alternatives': alternatives} if alternatives else None

        implements = kwargs.get('implements')

        from memory.recall import get_memory
        from memory import graph

        try:
            ctx = get_memory()
            node_id = graph.create_node(ctx, 'decision', title, rationale, extra=extra)
        except Exception as exc:
            return ToolResult.err(f'Failed to record decision: {exc}', code='graph-error')

        edge_note = ''
        if isinstance(implements, int) and not isinstance(implements, bool):
            try:
                graph.link_nodes(ctx, node_id, implements, 'implements')
            except Exception as exc:
                edge_note = f' (implements edge to {implements} skipped: {exc})'

        return ToolResult.ok(
            f'Recorded decision [{node_id}] {title}.{edge_note}', node_id=node_id, node_type='decision'
        )
