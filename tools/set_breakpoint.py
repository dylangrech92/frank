"""Set break point tool: registers a breakpoint at a file and line."""

from __future__ import annotations

import os
from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class SetBreakpoint(Tool):
    """Register a breakpoint in the debug adapter.

    Works before or during a session; mid-session breakpoints take effect immediately.
    """

    name = 'set_breakpoint'
    description = (
        'Set a breakpoint at a file and line, optionally conditional. '
        'Works before or during a debug session; mid-session breakpoints take effect immediately.'
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
                'description': '1-based line number.',
            },
            'condition': {
                'type': 'string',
                'description': 'Optional Python expression; breakpoint only fires when it is true.',
            },
        },
        'required': ['path', 'line'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, registering a breakpoint.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path`` (str),
                ``line`` (int), and optional ``condition`` (str).

        Returns:
            A ``ToolResult`` with breakpoint status on success, or an error when
            the debugger is unavailable (code ``debug-unavailable``).
        """
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.DEBUG_MANAGER is None:
            return ToolResult.err(
                'debugger not initialised',
                code='debug-unavailable',
            )

        file = kwargs['path']
        line = kwargs['line']
        condition = kwargs.get('condition')

        try:
            bp = main_module.DEBUG_MANAGER.set_breakpoint(file, line, condition)
        except Exception as exc:
            return ToolResult.err(f'failed to set breakpoint: {exc}', code='debug-error')

        try:
            rel = os.path.relpath(bp['file'], main_module.DEBUG_MANAGER._root)
        except (ValueError, AttributeError):
            rel = bp['file']

        lines: list[str] = [f'breakpoint set at {rel}:{line}']

        if condition is not None:
            lines.append(f' (condition: {condition})')

        if bp['active'] is False:
            lines.append(' [pending - no active session]')
        elif bp['verified'] is True:
            lines.append(' [verified]')
        elif bp['verified'] is False:
            lines.append(' [unverified]')

        body = ''.join(lines)

        verified = bp['verified'] if bp['verified'] is not None else False

        return ToolResult.ok(body, line=line, active=bp['active'], verified=verified)
