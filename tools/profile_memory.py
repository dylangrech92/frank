"""Profile-memory tool: per-allocation-site memory profiling for python, node, and php."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

from runtime.profiling import (
    fmt_bytes,
    fmt_seconds,
    parse_cachegrind,
    parse_heapprofile,
    php_xdebug_status,
    render_top_table,
    run_measured,
    which_interpreter,
    write_python_bootstrap,
)
from tools._sandbox import resolve_in_root
from tools._snapshot import publish_snapshot_diff, render_mutation_line, snapshot_tree
from tools.base import Tool
from tools.result import ToolResult

# Extension chosen for the temp file written from a user-provided snippet so
# the interpreter picks the right parser. Mirrors the pattern in verify_scratch.
_SNIPPET_SUFFIXES: dict[str, str] = {
    'python': '.py',
    'node': '.js',
    'php': '.php',
}


def _format_streams(stdout_text: str, stderr_text: str) -> str:
    """Render captured stdout/stderr with section labels.

    The ``--- stderr ---`` section is only emitted when *stderr_text* is
    non-empty, so a program that produced no error output does not carry an
    empty labelled block.
    """
    if stderr_text:
        return f'--- stdout ---\n{stdout_text}\n--- stderr ---\n{stderr_text}'
    return f'--- stdout ---\n{stdout_text}'


def _stderr_tail(stderr: str, n: int = 10) -> str:
    """Return the last *n* lines of *stderr* joined with newlines."""
    return '\n'.join(str(stderr).splitlines()[-n:])


def _resolve_target(
    target: str, root: Path,
) -> tuple[Path, str]:
    """Resolve a project-relative *target* to a Path under *root* and return
    ``(resolved_path, relative_label)``.

    Raises ValueError on escape or missing file.
    """
    resolved = resolve_in_root(root, target)
    if not resolved.is_file():
        raise ValueError(
            f'target file not found: {target!r} (resolved to {resolved})'
        )
    return resolved, target


def _render_python_memory(
    result_path: str, top: int,
) -> str | None:
    """Render tracemalloc result JSON into a top-N table.

    Returns a string containing the header, current/peak summary, and the
    top-N table, or None on read/parse failure.
    """
    try:
        with open(result_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    current_bytes = data.get('current_bytes', 0)
    peak_bytes = data.get('peak_bytes', 0)
    top_entries = data.get('top', [])
    target_error = data.get('target_error')

    lines: list[str] = []
    lines.append(f'Current: {fmt_bytes(current_bytes)}, Peak: {fmt_bytes(peak_bytes)}')
    lines.append('')

    # Render top-N table.
    rows: list[tuple] = []
    for entry in top_entries:
        file_ = entry.get('file', '')
        line = entry.get('line', 0)
        size = entry.get('size_bytes', 0)
        count = entry.get('count', 0)
        site = f'{file_}:{line}'
        rows.append((
            fmt_bytes(size),
            str(count),
            site,
        ))

    headers = ['size', 'count', 'site']
    table = render_top_table(headers, rows, top, 'sites')
    if table:
        lines.append(table)

    if target_error:
        lines.append('')
        lines.append(f'target raised: {target_error}')

    return '\n'.join(lines)


def _render_node_memory(
    artifact_dir: str, top: int,
) -> str:
    """Parse all *.heapprofile files in *artifact_dir* and render a top-N table.

    Returns a string containing the top-N table and a note about sampling.
    """
    import glob  # noqa: PLC0415 — stdlib, allowed here.

    files = sorted(glob.glob(os.path.join(artifact_dir, '*.heapprofile')))
    if not files:
        return ''

    merged: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for path in files:
        try:
            items = parse_heapprofile(path)
        except ValueError:
            continue
        for item in items:
            key = (
                item.get('function', ''),
                item.get('file', ''),
                item.get('line', 0),
            )
            if key not in seen:
                merged.append(item)
                seen.add(key)
            else:
                # Sum self/total bytes for duplicate function names.
                for existing in merged:
                    if (
                        existing['function'] == item['function']
                        and existing['file'] == item['file']
                        and existing['line'] == item['line']
                    ):
                        existing['self_bytes'] += item['self_bytes']
                        existing['total_bytes'] += item['total_bytes']
                        break

    merged.sort(key=lambda r: r['self_bytes'], reverse=True)

    rows: list[tuple] = []
    for item in merged:
        rows.append((
            fmt_bytes(item.get('self_bytes', 0)),
            fmt_bytes(item.get('total_bytes', 0)),
            item.get('function', '(anonymous)'),
        ))

    headers = ['self_bytes', 'total_bytes', 'function']
    table = render_top_table(headers, rows, top, 'functions')

    lines: list[str] = []
    if table:
        lines.append(table)
    lines.append('')
    lines.append(
        'Note: V8 --heap-prof data is allocation-sampling-based and '
        'approximate, unlike python exact counts.'
    )

    return '\n'.join(lines)


def _render_php_memory(
    artifact_dir: str, top: int,
) -> str:
    """Parse the cachegrind output file in *artifact_dir* and render the
    Memory event column as a top-N table.

    Returns a string containing the top-N table, or empty string if no
    cachegrind artifact is found.
    """
    import glob  # noqa: PLC0415

    files = sorted(glob.glob(os.path.join(artifact_dir, 'cachegrind.out.*')))
    if not files:
        return ''

    data = parse_cachegrind(files[0])
    events = data.get('events', [])

    # Find the Memory event (case-insensitive prefix match).
    memory_event = None
    for event in events:
        if event.lower().startswith('memory'):
            memory_event = event
            break

    if memory_event is None:
        return ''

    rows: list[tuple] = []
    for func in data.get('functions', []):
        name = func.get('function', '(unknown)')
        file_ = func.get('file', '')
        self_cost = func.get('self', {}).get(memory_event, 0)
        inc_cost = func.get('inclusive', {}).get(memory_event, 0)
        calls = func.get('calls', 0)
        label = f'{file_}:{name}' if file_ else name
        rows.append((
            str(calls),
            fmt_bytes(self_cost),
            fmt_bytes(inc_cost),
            label,
        ))

    headers = ['calls', f'self {memory_event}', f'inclusive {memory_event}', 'function']
    return render_top_table(headers, rows, top, 'functions')


class ProfileMemory(Tool):
    """Per-allocation-site memory profiling for python, node, and php.

    Profiles a target script (or snippet) and renders a top-N table of memory
    allocations.  Supports python (tracemalloc), node (V8 --heap-prof), and
    php (Xdebug cachegrind with Memory event).
    """

    name = 'profile_memory'
    summary = 'Per-allocation-site memory profiling for python, node, and php.'
    description = (
        'Profiles a target script (or snippet) and renders a top-N table of '
        'memory allocations.  Supports python (tracemalloc), node (V8 '
        '--heap-prof), and php (Xdebug cachegrind with Memory event).  A '
        'NONZERO exit with an artifact present is still reported as success.'
    )
    action = 'profile the memory usage'
    oversize_hint = 'lower top or focus on a specific allocation site'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'language': {
                'type': 'string',
                'enum': ['python', 'node', 'php'],
                'description': 'The language/runtime to profile.',
            },
            'target': {
                'type': 'string',
                'description': (
                    'Project-relative path to the script to profile. Exactly '
                    'one of target or snippet must be provided.'
                ),
            },
            'snippet': {
                'type': 'string',
                'description': (
                    'Source code to profile. Written to a temporary file '
                    'outside the project tree. Exactly one of target or '
                    'snippet must be provided.'
                ),
            },
            'args': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': (
                    'Arguments forwarded to the profiled program.'
                ),
            },
            'is_module': {
                'type': 'boolean',
                'description': (
                    'Python only: run target with -m as a module name.'
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
                    'Maximum seconds to wait for the profiled program. '
                    'Default 120.'
                ),
            },
        },
        'required': ['language'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the profile-memory tool.

        Args:
            language: One of 'python', 'node', 'php' (required).
            target: Project-relative path to a script (optional).
            snippet: Source code to profile (optional).
            args: Arguments forwarded to the profiled program (optional).
            is_module: Python only: run target with -m (optional).
            top: Number of top entries to show (optional, default 15).
            timeout: Max seconds to wait (optional, default 120).

        Returns:
            A ``ToolResult`` with the top-N memory table, stdout/stderr tails,
            and any file mutations the profiled program produced.
        """
        language_raw = kwargs.get('language')
        language = language_raw if isinstance(language_raw, str) else ''
        target_raw = kwargs.get('target')
        target = target_raw if isinstance(target_raw, str) else ''
        snippet_raw = kwargs.get('snippet')
        snippet = snippet_raw if isinstance(snippet_raw, str) else ''
        args_raw = kwargs.get('args')
        args: list[str] = args_raw if isinstance(args_raw, list) else []
        is_module_raw = kwargs.get('is_module')
        is_module = is_module_raw if isinstance(is_module_raw, bool) else False
        top_raw = kwargs.get('top')
        top: int = top_raw if isinstance(top_raw, int) else 15
        timeout_raw = kwargs.get('timeout')
        timeout: int = timeout_raw if isinstance(timeout_raw, int) else 120

        # --- Argument validation ---
        if language not in ('python', 'node', 'php'):
            return ToolResult.err(
                f'language must be one of: python, node, php. Got {language!r}.',
                code='bad-arguments',
                hint='Set language to one of: python, node, php.',
            )

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

        if language != 'python' and is_module:
            return ToolResult.err(
                'is_module is only valid for python.',
                code='bad-arguments',
                hint='Set language to "python" to use is_module.',
            )

        if is_module and snippet:
            return ToolResult.err(
                'is_module is only valid with "target", not "snippet".',
                code='bad-arguments',
                hint='Provide a target path (not a snippet) when is_module is true.',
            )

        root = Path.cwd()

        # --- Locate interpreter ---
        interpreter = which_interpreter(language)
        if interpreter is None:
            env_key = {
                'python': 'CODING_AGENT_PY_BIN',
                'node': 'CODING_AGENT_NODE_BIN',
                'php': 'CODING_AGENT_PHP_BIN',
            }[language]
            return ToolResult.err(
                f'{language} interpreter not found. Set {env_key} to override.',
                code='interpreter-unavailable',
                hint=f'Set the {env_key} environment variable to point to the interpreter.',
            )

        # --- Resolve target or prepare snippet ---
        script_path: str | None = None
        tmp_file: str | None = None

        if target and is_module:
            script_path = target  # module name passed verbatim; the interpreter resolves it from cwd
        elif target:
            try:
                resolved, _ = _resolve_target(target, root)
            except ValueError as exc:
                return ToolResult.err(
                    str(exc),
                    code='missing-target',
                    hint='Provide a target path that exists in the project.',
                )
            script_path = str(resolved)
        else:
            # Snippet path: write to a temp file outside the project tree.
            suffix = _SNIPPET_SUFFIXES.get(language, '')
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix='profile_memory_')
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
        artifact_dir = tempfile.mkdtemp(prefix='perf_profile_')

        try:
            # --- Snapshot before ---
            before = snapshot_tree(root)

            # --- Build command ---
            if language == 'python':
                result_path = os.path.join(artifact_dir, 'memory_result.json')
                config = {
                    'target': script_path,
                    'is_module': is_module,
                    'args': args,
                    'result_path': result_path,
                }
                bootstrap_path = write_python_bootstrap('tracemalloc', config, artifact_dir)
                cmd = shlex.join([interpreter, bootstrap_path])
                artifact_pattern = result_path

            elif language == 'node':
                cmd = shlex.join([
                    interpreter, '--heap-prof', '--heap-prof-dir',
                    artifact_dir, script_path,
                ] + args)
                artifact_pattern = os.path.join(artifact_dir, '*.heapprofile')

            else:  # php
                xdebug_status = php_xdebug_status(interpreter)
                if xdebug_status == 'missing':
                    return ToolResult.err(
                        f'Xdebug is not available for {interpreter}.',
                        code='xdebug-unavailable',
                        hint='Install or enable Xdebug for this PHP binary.',
                    )
                php_args: list[str] = [
                    interpreter,
                    '-dxdebug.mode=profile',
                    f'-dxdebug.output_dir={artifact_dir}',
                    '-dxdebug.use_compression=0',
                    script_path,
                ]
                if xdebug_status == 'available':
                    php_args.insert(1, '-dzend_extension=xdebug')
                cmd = shlex.join(php_args + args)
                artifact_pattern = os.path.join(artifact_dir, 'cachegrind.out.*')

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
                stderr_tail = _stderr_tail(result.stderr)
                lines = [
                    f'Profiled program timed out after {timeout} seconds.',
                    'No profile artifact survives a kill — the program must '
                    'terminate on its own for the profiler to write its file.',
                    f'--- stderr ---\n{stderr_tail}',
                ]
                if mutation_line:
                    lines.append(mutation_line)
                return ToolResult.err(
                    '\n'.join(lines),
                    code='timeout',
                    hint='Try a longer timeout or a smaller program.',
                )

            # --- Check for artifact ---
            import glob as _glob  # noqa: PLC0415
            if language == 'python':
                # The python result file is handled by _render_python_memory,
                # which returns None (-> profile-result-missing) for both an
                # absent and an unparseable result. Skip the artifact-missing
                # pre-check so python never reports 'profile-artifact-missing'.
                has_artifact = True
            elif language == 'node':
                has_artifact = bool(_glob.glob(artifact_pattern))
            else:
                has_artifact = bool(_glob.glob(artifact_pattern))

            if not has_artifact:
                stderr_tail = _stderr_tail(result.stderr)
                lines = [
                    f'Profile artifact missing. Program exited with code '
                    f'{result.exit_code}.',
                    f'--- stderr ---\n{stderr_tail}',
                    'The program must terminate on its own for the profiler '
                    'to write its file.',
                ]
                if mutation_line:
                    lines.append(mutation_line)
                return ToolResult.err(
                    '\n'.join(lines),
                    code='profile-artifact-missing',
                    hint='Ensure the profiled program terminates normally so the profiler can write its output.',
                )

            # --- Render profile data ---
            lines: list[str] = []
            lines.append(f'Language: {language}')
            lines.append(f'Interpreter: {interpreter}')
            lines.append(f'Wall time: {fmt_seconds(result.wall_s)}')
            lines.append('')

            if language == 'python':
                table = _render_python_memory(artifact_pattern, top)
                if table is None:
                    stderr_tail = _stderr_tail(result.stderr)
                    lines = [
                        'Failed to read memory profile result.',
                        f'--- stderr ---\n{stderr_tail}',
                    ]
                    if mutation_line:
                        lines.append(mutation_line)
                    return ToolResult.err(
                        '\n'.join(lines),
                        code='profile-result-missing',
                        hint='The tracer produced a result file but it could not be parsed.',
                    )
                lines.append(table)

            elif language == 'node':
                table = _render_node_memory(artifact_dir, top)
                lines.append(table)

            else:  # php
                table = _render_php_memory(artifact_dir, top)
                if not table:
                    stderr_tail = _stderr_tail(result.stderr)
                    lines = [
                        'Xdebug cachegrind artifact present, but this build '
                        'records no Memory events — only Time/Cache events are '
                        'available.',
                        f'--- stderr ---\n{stderr_tail}',
                    ]
                    if mutation_line:
                        lines.append(mutation_line)
                    return ToolResult.err(
                        '\n'.join(lines),
                        code='profile-artifact-missing',
                        hint='This Xdebug build does not record memory allocation events.',
                    )
                lines.append(table)

            if mutation_line:
                lines.append('')
                lines.append(mutation_line)

            # --- Stdout/stderr tails ---
            stdout_tail = _stderr_tail(result.stdout)
            stderr_tail = _stderr_tail(result.stderr)
            streams = _format_streams(stdout_tail, stderr_tail)
            lines.append('')
            lines.append(streams)

            # --- Grounding line for nonzero exit ---
            if result.exit_code != 0:
                lines.append('')
                lines.append(
                    f'(exit code {result.exit_code} — this program ran in '
                    f'working directory {root}; programs always run there, a '
                    f'cd into the project is never needed)'
                )

            return ToolResult.ok(
                '\n'.join(lines),
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
