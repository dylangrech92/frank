"""Forget tool: invalidate remembered fact(s) by key."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class Forget(Tool):
    """Invalidate remembered fact(s) by key, optionally narrowed to a kind.

    Soft-deletes matching live atoms so they no longer surface in recall.
    """

    name = 'forget'
    summary = 'Invalidate remembered fact(s) by key.'
    description = (
        'Invalidate/forget remembered fact(s) by key (optionally narrowed to a '
        'specific kind). Soft-deletes matching live atoms so they no longer '
        'surface in recall.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'key': {
                'type': 'string',
                'description': 'The key identifying the fact(s) to forget.',
            },
            'kind': {
                'type': 'string',
                'description': 'Optional category to narrow the forget to one kind.',
                'enum': ['project', 'convention', 'discovery', 'misc'],
            },
        },
        'required': ['key'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the forget tool."""
        key_raw = kwargs.get('key') if isinstance(kwargs.get('key'), str) else ''
        if not key_raw:
            return ToolResult.err(
                'The "key" argument is required and must be a non-empty string.',
                code='missing-argument',
            )

        kind_raw = kwargs.get('kind') if isinstance(kwargs.get('kind'), str) else None
        if kind_raw is not None:
            from memory.atomic import KINDS  # pylint: disable=import-outside-toplevel

            if kind_raw not in KINDS:
                return ToolResult.err(
                    f"'kind' must be one of {KINDS}, got {kind_raw!r}.",
                    code='invalid-kind',
                    hint=f"Supported kinds are {KINDS}.",
                )

        from memory.recall import forget as _forget  # pylint: disable=import-outside-toplevel

        n = _forget(key_raw, kind=kind_raw)
        return ToolResult.ok(
            f"Forgot {n} memory item(s) for key '{key_raw}'.",
            count=n,
        )
