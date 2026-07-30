"""Clear breakpoint tool: removes breakpoints at a file and optional line."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class ClearBreakpoint(Tool):
    """Remove a breakpoint at a file, optionally restricted to one line.

    When *line* is omitted, clears ALL breakpoints in that file.
    """

    name = 'clear_breakpoint'
    description = (
        'Remove a breakpoint at a file (a specific line, or all lines in that file when line is omitted).'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path to the source file (absolute or relative to project root).',
            },
            'line': {
                'type': 'integer',
                'description': 'Optional 1-based line number. When absent, clears all breakpoints in the file.',
            },
        },
        'required': ['path'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, clearing a breakpoint.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path`` (str)
                and optional ``line`` (int).

        Returns:
            A ``ToolResult`` with count of removed breakpoints on success, or an error
            when the debugger is unavailable (code ``debug-unavailable``).
        """
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.DEBUG_MANAGER is None:
            return ToolResult.err(
                'debugger not initialised',
                code='debug-unavailable',
            )

        file = kwargs['path']
        line = kwargs.get('line')

        try:
            removed = main_module.DEBUG_MANAGER.clear_breakpoint(file, line)
        except Exception as exc:
            return ToolResult.err(f'failed to clear breakpoint: {exc}', code='debug-error')

        if line is None:
            bp_word = 'breakpoint' if removed == 1 else 'breakpoints'
            body = f'cleared {removed} {bp_word} in {file}'
        elif removed == 0:
            body = f'no breakpoint at {file}:{line}'
        else:
            body = f'cleared breakpoint at {file}:{line}'

        return ToolResult.ok(body, removed=removed)
