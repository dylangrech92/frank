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

# Upper bound on paths named per kind in the mutation note appended to a
# foreground result. A command that writes hundreds of files (e.g. a generator
# script) must not bloat the render, so extra paths collapse to a "+N more"
# tail.
_MAX_RENDERED_PATHS = 5


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


def _format_streams(stdout_text: str, stderr_text: str) -> str:
    """Render captured stdout/stderr with section labels.

    The ``--- stderr ---`` section is only emitted when *stderr_text* is
    non-empty, so a command that produced no error output does not carry an
    empty labelled block. Shared by the timeout and normal foreground paths so
    the two render captured streams identically.
    """
    if stderr_text:
        return f'--- stdout ---\n{stdout_text}\n--- stderr ---\n{stderr_text}'
    return f'--- stdout ---\n{stdout_text}'


def _diff_snapshots(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> tuple[list[str], list[str], list[str]]:
    """Classify the difference between two tree snapshots.

    A path present only in *after* is created, present only in *before* is
    deleted, and present in both with a different ``(mtime_ns, size)`` is
    changed. Each list is sorted so the classification is deterministic
    regardless of dict iteration order.

    Returns:
        ``(created, deleted, changed)`` — three sorted lists of absolute paths.
    """
    before_paths = set(before)
    after_paths = set(after)
    created = sorted(after_paths - before_paths)
    deleted = sorted(before_paths - after_paths)
    changed = sorted(p for p in before_paths & after_paths if before[p] != after[p])
    return created, deleted, changed


def _publish_snapshot_diff(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> tuple[list[str], list[str], list[str]]:
    """Emit a mutation event for every file that a command created/changed/deleted.

    Classification is delegated to :func:`_diff_snapshots`; this publishes one
    event per path in the order class created, deleted, changed. Paths are
    already absolute (from :func:`os.walk`), matching how the edit tools publish
    resolved paths. ``emit_mutation`` is called directly (not wrapped) so
    subscriber errors surface exactly as they do for the other publishers.

    Returns:
        The ``(created, deleted, changed)`` classification, so the caller can
        render the same fact it just published without recomputing it.
    """
    created, deleted, changed = _diff_snapshots(before, after)
    for path in created:
        emit_mutation('created', path)
    for path in deleted:
        emit_mutation('deleted', path)
    for path in changed:
        emit_mutation('changed', path)
    return created, deleted, changed


def _render_mutation_line(
    root: Path,
    created: list[str],
    deleted: list[str],
    changed: list[str],
) -> str:
    """Build the one-line world-fact note naming files a command touched.

    Paths arrive absolute (from the snapshot walk) and are rendered relative to
    *root* so the note reads in project terms. Each kind names at most
    :data:`_MAX_RENDERED_PATHS` paths, then a ``+N more`` tail, so a script that
    writes many files cannot bloat the result. Empty kinds are omitted, and an
    all-empty diff yields the empty string so the caller can test truthiness.

    The note states the world fact only — which files changed — with no
    directive or conditional advice attached.
    """
    segments = [
        seg for seg in (
            _format_kind('created', _relativize(created, root)),
            _format_kind('changed', _relativize(changed, root)),
            _format_kind('deleted', _relativize(deleted, root)),
        )
        if seg is not None
    ]
    if not segments:
        return ''
    return f"(this command modified project files — {'; '.join(segments)})"


def _relativize(paths: list[str], root: Path) -> list[str]:
    """Render each absolute path in *paths* relative to *root*.

    A path that cannot be expressed relative to *root* (e.g. a different drive)
    is kept as-is rather than dropped, so nothing silently vanishes from the note.
    """
    rel: list[str] = []
    for path in paths:
        try:
            rel.append(os.path.relpath(path, root))
        except ValueError:
            rel.append(path)
    return rel


def _format_kind(label: str, paths: list[str]) -> str | None:
    """Format one ``label: p1, p2, +N more`` segment, or ``None`` when empty."""
    if not paths:
        return None
    shown = paths[:_MAX_RENDERED_PATHS]
    joined = ', '.join(shown)
    extra = len(paths) - len(shown)
    if extra > 0:
        joined += f', +{extra} more'
    return f'{label}: {joined}'


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
    summary = 'Run a shell command (foreground or background).'
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
        before = _snapshot_tree(root)
        result = run_one_shot(cmd, str(root), timeout_seconds=timeout)

        # Publish snapshot-diff mutation events once, regardless of how the
        # command ended (success, nonzero exit, or timeout) — files may be
        # mutated on any of those paths. A failed/aborted snapshot degrades to
        # no events (and an empty mutation line) rather than crashing the
        # command result.
        mutation_line = ''
        if before is not None:
            after = _snapshot_tree(root)
            if after is not None:
                created, deleted, changed = _publish_snapshot_diff(before, after)
                mutation_line = _render_mutation_line(root, created, deleted, changed)

        if result['timed_out']:
            body = _format_streams(str(result['stdout']), str(result.get('stderr', '')))
            if mutation_line:
                body += f'\n{mutation_line}'
            return ToolResult.err(
                f'Command was killed after {timeout} seconds.\n\n{body}',
                code='timeout',
                hint=f'The command ran longer than {timeout}s. Try a shorter timeout or run in background mode with background=true.',
            )

        output = _format_streams(str(result['stdout']), str(result.get('stderr', '')))
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
