"""load_tool tool: bring a catalog tool's full definition into context."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class LoadTool(Tool):
    """Load a tool from the catalog so its full definition is callable.

    The catalog in the system message lists every available tool with a short
    summary.  Pass the tool's ``name`` here; the full schema is injected into the
    tools array immediately, and the tool can be called normally right away —
    no need to wait for the next turn.
    """

    name = 'load_tool'
    summary = 'Load a tool from the catalog so its full definition is callable.'
    description = (
        'Load a tool from the catalog into context. The catalog (listed in the system '
        'message) shows every available tool with a one-line summary. Pass the tool\'s '
        '`name` and its full definition becomes callable immediately — no need to wait '
        'for the next turn. Loading an already-loaded tool is a no-op success.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'name': {
                'type': 'string',
                'description': 'Name of the tool to load, exactly as shown in the catalog.',
            },
        },
        'required': ['name'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the load_tool tool.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``name``
                (required) — a tool name from the catalog.

        Returns:
            A ``ToolResult`` confirming the load on success, or an error listing
            the valid catalog names when the requested tool is unknown.
        """
        name_raw = kwargs.get('name') if isinstance(kwargs.get('name'), str) else ''

        if not name_raw:
            return ToolResult.err(
                'The "name" argument is required and must be a non-empty string.',
                code='missing-argument',
            )

        from tools import registry  # pylint: disable=import-outside-toplevel

        if not registry._registry:  # auto-discover if somehow not yet done
            registry.discover()

        already = name_raw in registry.PINNED or name_raw in registry._active
        if name_raw in registry.PINNED:
            return ToolResult.ok(
                f"'{name_raw}' is always available — no need to load it.",
                name=name_raw,
            )
        if name_raw not in registry._registry:
            valid = sorted(n for n in registry._registry if n not in registry.PINNED)
            return ToolResult.err(
                f"unknown tool: {name_raw!r}",
                code='unknown-tool',
                hint=f"Valid tools: {', '.join(valid)}",
            )

        registry.activate(name_raw)

        if already:
            return ToolResult.ok(
                f"'{name_raw}' is already loaded.",
                name=name_raw,
            )
        return ToolResult.ok(
            f"Loaded '{name_raw}'. It is now callable.",
            name=name_raw,
        )
