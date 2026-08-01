"""Run command tool: execute a shell command inside the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.process import is_denied, run_one_shot, start_background
from runtime.profiling import format_streams
from tools._snapshot import publish_snapshot_diff, render_mutation_line, snapshot_tree
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

    A foreground command that mutates files (e.g. a shell redirect or a script
    that writes to disk) publishes mutation events through the shared bus by
    snapshotting the project tree before and after the run and diffing it, so
    the per-turn files_changed accounting, the reactive lint delta, and the
    verify gate all arm even when the edit did not go through an edit tool. That
    same diff is also named in the foreground result body, so the model sees
    which files a command touched at the moment it happens. Background commands
    do not publish mutation events.
    """

    name = 'run_command'
    description = (
        'Runs a shell command. The default mode is foreground and waits for '
        'the result; set background=true to run it in the background and poll '
        'later. Commands always run with the project root as the working '
        'directory, so a cd into the project is never needed. A deny-list '
        'blocks certain hazardous commands before they are executed. When a '
        'command creates, changes, or deletes project files, the result names '
        'the affected files.'
    )
    action = 'run the command'
    oversize_hint = 'pipe the output through head/tail or redirect it to a file and read a slice'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'cmd': {
                'type': 'string',
                'description': 'The shell command to execute.',
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
        raw_cmd = kwargs.get('cmd')
        cmd = raw_cmd if isinstance(raw_cmd, str) else ''
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
        root = Path.cwd()
        before = snapshot_tree(root)
        result = run_one_shot(cmd, str(root), timeout_seconds=timeout)

        # Publish snapshot-diff mutation events once, regardless of how the
        # command ended (success, nonzero exit, or timeout) — files may be
        # mutated on any of those paths. A failed/aborted snapshot degrades to
        # no events (and an empty mutation line) rather than crashing the
        # command result.
        mutation_line = ''
        if before is not None:
            after = snapshot_tree(root)
            if after is not None:
                created, deleted, changed = publish_snapshot_diff(before, after)
                mutation_line = render_mutation_line(root, created, deleted, changed)

        if result['timed_out']:
            body = format_streams(str(result['stdout']), str(result.get('stderr', '')))
            if mutation_line:
                body += f'\n{mutation_line}'
            return ToolResult.err(
                f'Command was killed after {timeout} seconds.\n\n{body}',
                code='timeout',
                hint=f'The command ran longer than {timeout}s. Try a shorter timeout or run in background mode with background=true.',
            )

        output = format_streams(str(result['stdout']), str(result.get('stderr', '')))
        # Name the files this command touched (fact only) before the nonzero-exit
        # grounding line, so the render reads streams, then mutation, then
        # grounding.
        if mutation_line:
            output += f'\n{mutation_line}'

        exit_code = result['exit_code']
        # A nonzero exit is otherwise only visible as a trailing meta line while
        # the result header still reads "success". Append a grounding line that
        # names the exit code and the concrete absolute working directory at the
        # failure moment, so a command that failed after a hallucinated `cd`
        # learns where it actually ran instead of retrying the same wrong path.
        if exit_code != 0:
            output += (
                f'\n(exit code {exit_code} — this command ran in working directory '
                f'{root}; commands always run there, a cd into the project is '
                f'never needed)'
            )
        return ToolResult.ok(
            output,
            exit_code=exit_code,
            timed_out=False,
        )
