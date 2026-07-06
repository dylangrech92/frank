"""Stop process tool: gracefully terminate a tracked background process."""

from __future__ import annotations

from typing import Any

from runtime.process import stop_background
from tools.base import Tool
from tools.result import ToolResult


class StopProcess(Tool):
    """Gracefully stops a tracked background process and drains its final output.

    Sends SIGTERM to the process group; escalates to SIGKILL if still alive after
    two seconds. Reads any remaining buffered lines before returning.

    No mutation event is emitted.
    """

    name = 'stop_process'
    summary = 'Stop a tracked background process.'
    description = (
        'Stops a tracked background process by sending SIGTERM (escalating to '
        'SIGKILL if needed) and drains its final output.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'handle': {
                'type': 'string',
                'description': (
                    'The background process handle returned by '
                    '"run_command" with background=true.'
                ),
            },
        },
        'required': ['handle'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the stop-process tool.

        Args:
            handle: Background process handle string (required).

        Returns:
            A ``ToolResult`` whose body describes that the process was stopped and
            includes any final drained output; metadata carries *exit_code* (int | None).
            Unknown handles produce an error result with code ``unknown-handle``.
        """
        handle_id = kwargs.get('handle') if isinstance(kwargs.get('handle'), str) else ''

        if not handle_id:
            return ToolResult.err(
                'The "handle" parameter is required.',
                code='missing-handle',
            )

        result = stop_background(handle_id)
        if result is None:
            return ToolResult.err(
                f'Unknown handle "{handle_id}". Start a process with run_command using background=true to create one.',
                code='unknown-handle',
                hint='Use run_command with background=true to spawn a tracked process.',
            )

        output = result['output']
        if output:
            body_lines = [f'{output.rstrip(chr(10))}', '', 'The process was stopped.']
        else:
            body_lines = ['No remaining output.', '', 'The process was stopped.']

        return ToolResult.ok(
            '\n'.join(body_lines),
            exit_code=result['exit_code'],
        )
