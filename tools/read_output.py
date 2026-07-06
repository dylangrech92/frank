"""Read output tool: poll the drained buffer of a background process."""

from __future__ import annotations

from typing import Any

from runtime.process import read_output_from
from tools.base import Tool
from tools.result import ToolResult


class ReadOutput(Tool):
    """Polls and drains the buffered output of a tracked background process.

    Each call returns only lines produced since the previous call, along with
    the subprocess's running state and exit code when available.

    No mutation event is emitted.
    """

    name = 'read_output'
    summary = 'Drain the buffered stdout of a background process.'
    description = (
        'Drains the buffered stdout of a background process and returns it '
        'along with its running state and exit code. Output is cleared from '
        'the buffer after each call.'
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
        """Execute the read-output tool.

        Args:
            handle: Background process handle string (required).

        Returns:
            A ``ToolResult`` with drained output text and metadata carrying
            *running* (bool) and *exit_code* (int | None).  Unknown handles
            produce an error result with code ``unknown-handle``.
        """
        handle_id = kwargs.get('handle') if isinstance(kwargs.get('handle'), str) else ''

        if not handle_id:
            return ToolResult.err(
                'The "handle" parameter is required.',
                code='missing-handle',
            )

        result = read_output_from(handle_id)
        if result is None:
            return ToolResult.err(
                f'Unknown handle "{handle_id}". Start a process with run_command using background=true to create one.',
                code='unknown-handle',
                hint='Use run_command with background=true to spawn a tracked process.',
            )

        output = result['output']
        if output:
            body = output.rstrip('\n')
        else:
            body = 'No new output.'

        meta: dict[str, Any] = {'running': result['running']}

        if result['exit_code'] is not None:
            meta['exit_code'] = result['exit_code']

        return ToolResult.ok(body, **meta)
