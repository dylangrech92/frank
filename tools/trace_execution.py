"""Trace-execution tool: exact call counts and stack depth for python.

This tool is the python-only instrument for iteration counts and nesting
depth.  It reports exact counts via ``sys.settrace`` — never times.  For php
the parity boundary is cachegrind call counts from ``profile_hotspots``; for
node the parity boundary is sampling-based data from ``profile_hotspots``.
``trace_execution`` is the exact-counts instrument for python only.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

from runtime.profiling import (
    fmt_seconds,
    format_streams,
    make_artifact_dir,
    render_top_table,
    run_measured,
    tail_lines,
    which_interpreter,
    write_python_bootstrap,
)
from tools._sandbox import resolve_existing_file, resolve_in_root
from tools._snapshot import publish_snapshot_diff, render_mutation_line, snapshot_tree
from tools.base import Tool
from tools.result import ToolResult

_SNIPPET_SUFFIX = '.py'


def _resolve_focus_file(
    focus_file: str, root: Path,
) -> str:
    """Resolve *focus_file* inside the project.

    Returns the absolute path on success.  Raises ValueError on escape or
    missing file.
    """
    resolved = resolve_in_root(root, focus_file)
    if not resolved.is_file():
        raise ValueError(
            f'focus_file not found: {focus_file!r} (resolved to {resolved})'
        )
    return str(resolved)


class TraceExecution(Tool):
    """Python execution tracing: exact call counts, per-line hit counts, max stack depth.

    Traces a target script (or snippet) with ``sys.settrace`` and reports
    exact call counts per function, per-line hit counts for an optional
    focus file, and maximum observed stack depth.  Reports counts only —
    never times.  For php the parity boundary is cachegrind call counts
    from ``profile_hotspots``; for node the parity boundary is sampling-based
    data from ``profile_hotspots``.
    """

    name = 'trace_execution'
    description = (
        'Traces a target script (or snippet) with sys.settrace and reports '
        'exact call counts per function, per-line hit counts for an optional '
        'focus file, and maximum observed stack depth.  Reports counts only — '
        'never times.  For php the parity boundary is cachegrind call counts '
        'from profile_hotspots; for node the parity boundary is sampling-based '
        'data from profile_hotspots.  A NONZERO exit with a result JSON present '
        'is still reported as success.'
    )
    action = 'trace the execution'
    oversize_hint = 'lower top or focus on a specific file'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'target': {
                'type': 'string',
                'description': (
                    'Project-relative path to the script to trace. Exactly '
                    'one of target or snippet must be provided.'
                ),
            },
            'snippet': {
                'type': 'string',
                'description': (
                    'Python source code to trace. Written to a temporary file '
                    'outside the project tree. Exactly one of target or '
                    'snippet must be provided.'
                ),
            },
            'args': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': (
                    'Arguments forwarded to the traced program.'
                ),
            },
            'is_module': {
                'type': 'boolean',
                'description': (
                    'Run target with -m as a module name.'
                ),
            },
            'focus_file': {
                'type': 'string',
                'description': (
                    'A project-relative file path — collect per-line hit '
                    'counts only for this file.'
                ),
            },
            'top': {
                'type': 'integer',
                'description': (
                    'Number of top entries to show. Range 1-50. Default 15.'
                ),
            },
            'timeout': {
                'type': 'integer',
                'description': (
                    'Maximum seconds to wait for the traced program. '
                    'Default 120.'
                ),
            },
        },
        'required': [],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the trace-execution tool.

        Args:
            target: Project-relative path to a script (optional).
            snippet: Python source code to trace (optional).
            args: Arguments forwarded to the traced program (optional).
            is_module: Run target with -m (optional).
            focus_file: Project-relative file path for per-line hit counts (optional).
            top: Number of top entries to show (optional, default 15).
            timeout: Max seconds to wait (optional, default 120).

        Returns:
            A ``ToolResult`` with the call-count table, optional line-hits
            table, max stack depth, stdout/stderr tails, and any file mutations
            the traced program produced.
        """
        target_raw = kwargs.get('target')
        target = target_raw if isinstance(target_raw, str) else ''
        snippet_raw = kwargs.get('snippet')
        snippet = snippet_raw if isinstance(snippet_raw, str) else ''
        args_raw = kwargs.get('args')
        args: list[str] = args_raw if isinstance(args_raw, list) else []
        is_module_raw = kwargs.get('is_module')
        is_module = is_module_raw if isinstance(is_module_raw, bool) else False
        focus_file_raw = kwargs.get('focus_file')
        focus_file = focus_file_raw if isinstance(focus_file_raw, str) else ''
        top_raw = kwargs.get('top')
        top: int = top_raw if isinstance(top_raw, int) else 15
        timeout_raw = kwargs.get('timeout')
        timeout: int = timeout_raw if isinstance(timeout_raw, int) else 120

        # --- Argument validation ---
        if not target and not snippet:
            return ToolResult.err(
                'Exactly one of "target" or "snippet" must be provided.',
                code='bad-arguments',
                hint='Provide either a target path or a snippet, but not both.',
            )

        if target and snippet:
            return ToolResult.err(
                'Exactly one of "target" or "snippet" must be provided.',
                code='bad-arguments',
                hint='Provide either a target path or a snippet, but not both.',
            )

        if not (1 <= top <= 50):
            return ToolResult.err(
                f'top must be an integer between 1 and 50. Got {top}.',
                code='bad-arguments',
                hint='Set top to an integer in the range 1..50.',
            )

        if timeout <= 0:
            return ToolResult.err(
                'timeout must be a positive integer.',
                code='bad-arguments',
                hint='Set timeout to a positive integer (seconds).',
            )

        if is_module and snippet:
            return ToolResult.err(
                'is_module is only valid with "target", not "snippet".',
                code='bad-arguments',
                hint='Provide a target path (not a snippet) when is_module is true.',
            )

        root = Path.cwd()

        # --- Locate interpreter ---
        interpreter = which_interpreter('python')
        if interpreter is None:
            env_key = 'CODING_AGENT_PY_BIN'
            return ToolResult.err(
                f'python interpreter not found. Set {env_key} to override.',
                code='interpreter-unavailable',
                hint=f'Set the {env_key} environment variable to point to the interpreter.',
            )

        # --- Resolve focus_file ---
        focus_abs = ''
        if focus_file:
            try:
                focus_abs = _resolve_focus_file(focus_file, root)
            except ValueError as exc:
                return ToolResult.err(
                    f'focus_file not found: {focus_file!r} — {exc}',
                    code='missing-target',
                    hint='Provide a focus_file path that exists in the project.',
                )

        # --- Resolve target or prepare snippet ---
        script_path: str | None = None
        tmp_file: str | None = None

        if target and is_module:
            script_path = target  # module name passed verbatim; the interpreter resolves it from cwd
        elif target:
            try:
                resolved = resolve_existing_file(root, target)
            except ValueError as exc:
                return ToolResult.err(
                    str(exc),
                    code='missing-target',
                    hint='Provide a target path that exists in the project.',
                )
            script_path = str(resolved)
        else:
            # Snippet path: write to a temp file outside the project tree.
            fd, tmp_path = tempfile.mkstemp(
                suffix=_SNIPPET_SUFFIX, prefix='trace_execution_',
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                    fh.write(snippet)
                script_path = tmp_path
                tmp_file = tmp_path
            except OSError as exc:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                return ToolResult.err(
                    f'Failed to write snippet to temp file: {exc}',
                    code='bad-arguments',
                    hint='The snippet content may be invalid.',
                )

        # --- Create artifacts directory ---
        artifact_dir = make_artifact_dir()

        try:
            # --- Snapshot before ---
            before = snapshot_tree(root)

            # --- Build result path and config ---
            result_path = os.path.join(artifact_dir, 'trace_result.json')

            config: dict[str, Any] = {
                'target': script_path,
                'is_module': is_module,
                'args': args,
                'result_path': result_path,
                'focus_file': focus_abs,
            }

            bootstrap_path = write_python_bootstrap('trace', config, artifact_dir)

            cmd = shlex.join([
                interpreter,
                bootstrap_path,
            ])

            # --- Execute measured ---
            result = run_measured(cmd, str(root), timeout_seconds=timeout)

            # --- Snapshot after ---
            after = snapshot_tree(root)
            mutation_line = ''
            if before is not None and after is not None:
                created, deleted, changed = publish_snapshot_diff(before, after)
                mutation_line = render_mutation_line(root, created, deleted, changed)

            # --- Timeout path ---
            if result.timed_out:
                stderr_tail = tail_lines(result.stderr)
                lines = [
                    f'Traced program timed out after {timeout} seconds.',
                    'No trace result survives a kill — the program must '
                    'terminate on its own for the tracer to write its file.',
                    f'--- stderr ---\n{stderr_tail}',
                ]
                if mutation_line:
                    lines.append(mutation_line)
                return ToolResult.err(
                    '\n'.join(lines),
                    code='timeout',
                    hint='Try a longer timeout or a smaller program.',
                )

            # --- Check for result JSON ---
            if not os.path.isfile(result_path):
                stderr_tail = tail_lines(result.stderr)
                lines = [
                    f'Trace result missing. Program exited with code '
                    f'{result.exit_code}.',
                    f'--- stderr ---\n{stderr_tail}',
                    'The program must terminate on its own for the tracer '
                    'to write its result JSON.',
                ]
                if mutation_line:
                    lines.append(mutation_line)
                return ToolResult.err(
                    '\n'.join(lines),
                    code='profile-result-missing',
                    hint='Ensure the traced program terminates normally so the tracer can write its result JSON.',
                )

            # --- Read result JSON ---
            try:
                with open(result_path, 'r', encoding='utf-8') as fh:
                    trace_result = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                stderr_tail = tail_lines(result.stderr)
                lines = [
                    f'Failed to read trace result JSON: {exc}',
                    f'--- stderr ---\n{stderr_tail}',
                ]
                if mutation_line:
                    lines.append(mutation_line)
                return ToolResult.err(
                    '\n'.join(lines),
                    code='profile-result-missing',
                    hint='The tracer produced a result file but it could not be parsed.',
                )

            # --- Render ---
            render_lines: list[str] = []
            render_lines.append(f'Interpreter: {interpreter}')
            render_lines.append(f'Wall time: {fmt_seconds(result.wall_s)}')
            render_lines.append('')

            max_depth = trace_result.get('max_depth', 0) or 0
            render_lines.append(f'max stack depth: {max_depth}')

            calls = trace_result.get('calls') or []
            if calls:
                call_rows: list[tuple] = []
                for entry in calls:
                    count = str(entry.get('count', 0))
                    function = entry.get('function', '(unknown)')
                    site = f'{entry.get("file", "")}:{entry.get("line", 0)}'
                    call_rows.append((count, function, site))
                call_rows = call_rows[:top]
                headers = ['count', 'function', 'site']
                table = render_top_table(headers, call_rows, top, 'calls')
                if table:
                    render_lines.append(table)

            if focus_file and focus_abs:
                line_hits = trace_result.get('line_hits') or []
                if line_hits:
                    hit_rows: list[tuple] = []
                    for entry in line_hits:
                        count = str(entry.get('count', 0))
                        line_num = str(entry.get('line', 0))
                        hit_rows.append((count, line_num))
                    hit_rows = hit_rows[:top]
                    line_headers = ['count', 'line']
                    line_table = render_top_table(
                        line_headers, hit_rows, top, 'line hits',
                    )
                    if line_table:
                        render_lines.append('')
                        render_lines.append(f'line hits in {focus_file}:')
                        render_lines.append(line_table)

            target_error = trace_result.get('target_error')
            if target_error:
                render_lines.append('')
                render_lines.append(f'target raised: {target_error}')

            if mutation_line:
                render_lines.append('')
                render_lines.append(mutation_line)

            # --- Stdout/stderr tails ---
            stdout_tail = tail_lines(result.stdout)
            stderr_tail = tail_lines(result.stderr)
            streams = format_streams(stdout_tail, stderr_tail)
            render_lines.append('')
            render_lines.append(streams)

            # --- Grounding line for nonzero exit ---
            if result.exit_code != 0:
                render_lines.append('')
                render_lines.append(
                    f'(exit code {result.exit_code} — this program ran in '
                    f'working directory {root}; programs always run there, a '
                    f'cd into the project is never needed)'
                )

            return ToolResult.ok(
                '\n'.join(render_lines),
                exit_code=result.exit_code,
                timed_out=False,
            )

        finally:
            # Clean up artifacts directory.
            try:
                shutil.rmtree(artifact_dir)
            except OSError:
                pass
            # Clean up temp file if we created one.
            if tmp_file is not None:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass
