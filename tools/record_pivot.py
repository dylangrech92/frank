"""record_pivot tool: transactionally record a pivot that supersedes prior nodes."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


def _coerce_ids(raw: Any) -> list[int]:
    """Normalise the *supersedes* argument to a list of ints (best effort)."""
    if isinstance(raw, bool):
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        out = []
        for part in raw.replace(',', ' ').split():
            try:
                out.append(int(part))
            except ValueError:
                continue
        return out
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str):
                try:
                    out.append(int(item.strip()))
                except ValueError:
                    continue
        return out
    return []


class RecordPivot(Tool):
    """Record a pivot node that supersedes one or more existing nodes (transactional)."""

    name = 'record_pivot'
    summary = 'Record a pivot that supersedes prior decision/spec nodes.'
    description = (
        'Record a pivot: a change of direction that supersedes one or more existing '
        'decision/spec/pivot nodes. Provide a title, why the pivot happened, and the '
        'id(s) of the node(s) it supersedes. This is transactional: if any target id '
        'does not exist, nothing is written.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'title': {'type': 'string', 'description': 'A short name for the pivot.'},
            'why': {'type': 'string', 'description': 'Why the direction changed.'},
            'supersedes': {
                'type': 'array',
                'items': {'type': 'integer'},
                'description': 'The id(s) of the node(s) this pivot supersedes.',
            },
        },
        'required': ['title', 'why', 'supersedes'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        title = kwargs.get('title') if isinstance(kwargs.get('title'), str) else ''
        why = kwargs.get('why') if isinstance(kwargs.get('why'), str) else ''
        if not title.strip():
            return ToolResult.err('The "title" argument is required.', code='missing-argument')
        if not why.strip():
            return ToolResult.err('The "why" argument is required.', code='missing-argument')

        supersedes = _coerce_ids(kwargs.get('supersedes'))
        if not supersedes:
            return ToolResult.err(
                'The "supersedes" argument must list at least one existing node id.',
                code='missing-argument',
            )

        from memory.recall import get_memory
        from memory import graph

        try:
            ctx = get_memory()
            node_id = graph.record_pivot(ctx, title, why, supersedes)
        except Exception as exc:
            return ToolResult.err(
                f'Failed to record pivot (no changes were written): {exc}', code='graph-error'
            )

        return ToolResult.ok(
            f'Recorded pivot [{node_id}] {title}; superseded {supersedes}.',
            node_id=node_id,
            node_type='pivot',
        )
