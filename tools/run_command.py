"""Run command tool: execute a shell command inside the project root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from runtime.process import is_denied, run_one_shot, start_background
from tools._sandbox import IGNORED_DIRS, emit_mutation
from tools.base import Tool
from tools.result import ToolResult

# Upper bound on files walked when snapshotting the tree around a foreground
# command. A very large tree must not pay a per-command walk tax, so detection
# is abandoned (no snapshot, no diff) once the walk crosses this many files.
_MAX_SNAPSHOT_FILES = 20000


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]] | None:
    """Record ``{abs_path: (st_mtime_ns, st_size)}`` for every file under *root*.

    Prunes any directory whose name starts with ``.`` plus the shared
    :data:`IGNORED_DIRS`, and skips any file whose name starts with ``.``. Walk
    or stat problems degrade to a partial/absent snapshot rather than raising —
    a detection failure must never break the command result.

    Returns:
        The snapshot mapping, or ``None`` when the file cap is exceeded or the
        walk cannot be completed (so the caller skips the diff entirely).
    """
    snapshot: dict[str, tuple[int, int]] = {}
    count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith('.') and d not in IGNORED_DIRS
            ]
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                count += 1
                if count > _MAX_SNAPSHOT_FILES:
                    return None
                full = os.path.join(dirpath, filename)
                try:
                    stat = os.stat(full)
                except OSError:
                    # A file that vanished mid-walk (a race) or is unreadable is
                    # simply absent from this snapshot, not a fatal error.
                    continue
                snapshot[full] = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    return snapshot


def _publish_snapshot_diff(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> None:
    """Emit a mutation event for every file that a command created/changed/deleted.

    A path present only afterwards is ``created``, present only before is
    ``deleted``, and present in both with a different ``(mtime_ns, size)`` is
    ``changed``. Paths are already absolute (from :func:`os.walk`), matching how
    the edit tools publish resolved paths. ``emit_mutation`` is called directly
    (not wrapped) so subscriber errors surface exactly as they do for the other
    publishers.
    """
    before_paths = set(before)
    after_paths = set(after)
    for path in after_paths - before_paths:
        emit_mutation('created', path)
    for path in before_paths - after_paths:
        emit_mutation('deleted', path)
    for path in before_paths & after_paths:
        if before[path] != after[path]:
            emit_mutation('changed', path)


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
    verify gate all arm even when the edit did not go through an edit tool.
    Background commands do not publish mutation events.
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
        root = Path.cwd()
        before = _snapshot_tree(root)
        result = run_one_shot(cmd, str(root), timeout_seconds=timeout)

        # Publish snapshot-diff mutation events once, regardless of how the
        # command ended (success, nonzero exit, or timeout) — files may be
        # mutated on any of those paths. A failed/aborted snapshot degrades to
        # no events rather than crashing the command result.
        if before is not None:
            after = _snapshot_tree(root)
            if after is not None:
                _publish_snapshot_diff(before, after)

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
