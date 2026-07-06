"""link_nodes tool: create a typed edge between two existing data-graph nodes."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult

_EDGE_TYPES = ('supersedes', 'implements', 'constrains', 'refines', 'relates_to')


class LinkNodes(Tool):
    """Create a typed edge between two existing graph nodes."""

    name = 'link_nodes'
    summary = 'Create a typed edge between two data-graph nodes.'
    description = (
        'Create a typed edge between two existing data-graph nodes. edge_type is one '
        'of: supersedes, implements, constrains, refines, relates_to.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'from_id': {'type': 'integer', 'description': 'The source node id.'},
            'to_id': {'type': 'integer', 'description': 'The target node id.'},
            'edge_type': {
                'type': 'string',
                'enum': list(_EDGE_TYPES),
                'description': 'The kind of edge.',
            },
        },
        'required': ['from_id', 'to_id', 'edge_type'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        from_id = kwargs.get('from_id')
        to_id = kwargs.get('to_id')
        edge_type = kwargs.get('edge_type') if isinstance(kwargs.get('edge_type'), str) else ''

        if not isinstance(from_id, int) or isinstance(from_id, bool):
            return ToolResult.err('The "from_id" argument must be an integer node id.', code='bad-arguments')
        if not isinstance(to_id, int) or isinstance(to_id, bool):
            return ToolResult.err('The "to_id" argument must be an integer node id.', code='bad-arguments')
        if edge_type not in _EDGE_TYPES:
            return ToolResult.err(
                f"'edge_type' must be one of {_EDGE_TYPES}, got {edge_type!r}.", code='bad-arguments'
            )

        from memory.recall import get_memory
        from memory import graph

        try:
            ctx = get_memory()
            edge_id = graph.link_nodes(ctx, from_id, to_id, edge_type)
        except Exception as exc:
            return ToolResult.err(f'Failed to link nodes: {exc}', code='graph-error')

        return ToolResult.ok(
            f'Linked {from_id} -[{edge_type}]-> {to_id}.', edge_id=edge_id, edge_type=edge_type
        )
