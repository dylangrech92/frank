"""record_rule tool: persist an always-enforced project rule into the data graph."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class RecordRule(Tool):
    """Record a project rule that is injected into every assembled context."""

    name = 'record_rule'
    summary = 'Record an always-enforced project rule.'
    description = (
        'Record an always-enforced project rule (e.g. a naming convention or '
        'invariant). Active rules are injected into every context so the agent '
        'follows them without being reminded. Provide a short title and the '
        'constraint text.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'title': {'type': 'string', 'description': 'A short name for the rule.'},
            'constraint': {'type': 'string', 'description': 'The rule text -- what must always hold.'},
        },
        'required': ['title', 'constraint'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        title = kwargs.get('title') if isinstance(kwargs.get('title'), str) else ''
        constraint = kwargs.get('constraint') if isinstance(kwargs.get('constraint'), str) else ''
        if not title.strip():
            return ToolResult.err('The "title" argument is required.', code='missing-argument')
        if not constraint.strip():
            return ToolResult.err('The "constraint" argument is required.', code='missing-argument')

        from memory.recall import get_memory
        from memory import graph

        try:
            ctx = get_memory()
            node_id = graph.create_node(ctx, 'rule', title, constraint)
        except Exception as exc:
            return ToolResult.err(f'Failed to record rule: {exc}', code='graph-error')

        return ToolResult.ok(f'Recorded rule [{node_id}] {title}.', node_id=node_id, node_type='rule')
