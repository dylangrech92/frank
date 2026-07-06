"""Run command tool: execute a shell command inside the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.process import is_denied, run_one_shot, start_background
from tools.base import Tool
from tools.result import ToolResult


class RunCommand(Tool):
    """Runs a shell command inside the project root.

    The *cmd* argument is passed to ``shell=True`` via ``subprocess.Popen``, so
    all standard shell features (pipes, redirects, globs, variable expansion)
    are available.  A deny-list check blocks hazardous commands before execution.

    Timeout is only meaningful for foreground runs; the background mode does not
    apply a per-process timeout because it delegates lifecycle to the model via
    ``read_output`` and ``stop_process``.
    """

    name = 'run_command'
    summary = 'Run a shell command (foreground or background).'
    description = (
        'Runs a shell command inside the project root. The default mode is '
        'foreground and waits for result; set background=true to run it in '
        'the background and poll later. A deny-list blocks certain hazardous '
        'commands before they are executed.'
    )
    action = 'run the command'
    oversize_hint = 'pipe the output through head/tail or redirect it to a file and read a slice'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'cmd': {
                'type': 'string',
                'description': (
                    'The shell command to execute inside the project root.'
                ),
            },
            'timeout': {
                'type': 'integer',
                'description': (
                    'Maximum seconds to wait for a foreground command to finish. '
                    'Default is 60. Only meaningful for foreground runs.'
                ),
            },
            'background': {
                'type': 'boolean',
                'description': (
                    'When true, spawn the command in the background and return '
                    'immediately with a process handle. Default is false.'
                ),
            },
        },
        'required': ['cmd'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute a shell command inside the project root.

        Args:
            cmd: The shell command to execute (required).
            timeout: Max seconds for foreground runs (optional, default 60).
            background: Spawn in background if true (optional, default false).

        Returns:
            A ``ToolResult`` describing success or failure.  Foreground returns
            stdout/stderr; background returns a process handle.  Deny-list blocks,
            timeouts, and subprocess errors are reported as error results.
        """
        cmd = kwargs.get('cmd') if isinstance(kwargs.get('cmd'), str) else ''
        timeout: int = kwargs.get('timeout') or 60
        background: bool = bool(kwargs.get('background', False))

        # --- Deny-list gate (no mutation events) ---
        deny_reason = is_denied(cmd)
        if deny_reason is not None:
            return ToolResult.err(
                f'Destructive command blocked ({deny_reason}).',
                code='destructive-command-blocked',
                hint='Review the command and remove any patterns that match the deny-list.',
            )

        # --- Background mode ---
        if background:
            handle_id = start_background(cmd, str(Path.cwd()))
            if handle_id is None:
                return ToolResult.err(
                    'Destructive command blocked (deny-list) — denied by runtime.process.is_denied on the command string.',
                    code='destructive-command-blocked',
                )
            return ToolResult.ok(
                f'Background process started (handle: {handle_id}). '
                f'Call read_output with handle "{handle_id}" to check progress, or call stop_process with that handle to end it.',
                handle=handle_id,
            )

        # --- Foreground mode ---
        result = run_one_shot(cmd, str(Path.cwd()), timeout_seconds=timeout)

        if result['timed_out']:
            partial = result['stdout']
            stderr_part = result.get('stderr', '')
            if stderr_part:
                body = (
                    f'--- stdout ---\n{partial}\n'
                    f'--- stderr ---\n{stderr_part}'
                )
            else:
                body = f'--- stdout ---\n{partial}'

            return ToolResult.err(
                f'Command was killed after {timeout} seconds.\n\n{body}',
                code='timeout',
                hint=f'The command ran longer than {timeout}s. Try a shorter timeout or run in background mode with background=true.',
            )

        # Normalize: stderr may be '' when not produced — label only if non-empty
        stdout_text = result['stdout']
        stderr_text = result.get('stderr', '')
        if stderr_text:
            output = (
                f'--- stdout ---\n{stdout_text}\n'
                f'--- stderr ---\n{stderr_text}'
            )
        else:
            output = f'--- stdout ---\n{stdout_text}'

        exit_code = result['exit_code']
        return ToolResult.ok(
            output,
            exit_code=exit_code,
            timed_out=False,
        )
