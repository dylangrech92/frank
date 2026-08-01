"""Profile-hotspot tool: per-function CPU profiling for python, node, and php."""

from __future__ import annotations

import os
import pstats
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

from runtime.profiling import (
    fmt_seconds,
    format_streams,
    php_xdebug_status,
    parse_cachegrind,
    parse_cpuprofile,
    render_top_table,
    run_measured,
    tail_lines,
    which_interpreter,
)
from tools._sandbox import resolve_existing_file
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


def _render_python_hotspots(
    profile_path: str, top: int, focus: str | None,
) -> str:
    """Render pstats output for the given profile file.

    Returns a string containing the top-N table, and (if *focus* is given) the
    callers/callees section for the named function.
    """
    # pstats.Stats builds .stats/.all_callees dynamically — typeshed doesn't
    # declare them, so keep the variable untyped for the checker.
    stats: Any = pstats.Stats(profile_path)
    stats.sort_stats('tottime')

    # Top-N rows of (ncalls, tottime, cumtime, file:line:name).
    # pstats tuple: (cc, nc, tottime, cumtime, callers_dict).
    rows: list[tuple] = []
    for func_key, (_cc, nc, tottime, cumtime, _callers) in stats.stats.items():
        # func_key is (file, line, name).
        file_, line, name = func_key
        label = f'{file_}:{line}:{name}'
        rows.append((
            str(nc),
            f'{tottime:.4f}',
            f'{cumtime:.4f}',
            label,
        ))

    headers = ['ncalls', 'tottime', 'cumtime', 'function']
    table = render_top_table(headers, rows, top, 'functions')

    lines: list[str] = []
    if table:
        lines.append(table)

    if focus is not None:
        matches = [k for k in stats.stats if k[2] == focus]
        if matches:
            lines.append('')
            for match_key in matches:
                m_file, m_line, m_name = match_key
                m_label = f'{m_file}:{m_line}:{m_name}'
                # Callers of this match.
                callers_of = stats.stats[match_key][4]  # type: ignore[index]
                lines.append(f'Callers of "{focus}" (from {m_label}):')
                caller_rows: list[tuple] = []
                for caller_key, call_data in callers_of.items():
                    c_file, c_line, c_name = caller_key
                    c_label = f'{c_file}:{c_line}:{c_name}'
                    # call_data is (cc, nc, tottime, cumtime).
                    caller_rows.append((
                        str(call_data[1]),
                        f'{call_data[2]:.4f}',
                        f'{call_data[3]:.4f}',
                        c_label,
                    ))
                caller_headers = ['ncalls', 'tottime', 'cumtime', 'caller']
                caller_table = render_top_table(
                    caller_headers, caller_rows, top, 'callers',
                )
                if caller_table:
                    lines.append(caller_table)

                # Callees of this match.
                stats.calc_callees()
                callees_of = stats.all_callees.get(match_key, {})
                lines.append(f'Callees of "{focus}" (from {m_label}):')
                callee_rows: list[tuple] = []
                for callee_key, call_data in callees_of.items():
                    c_file, c_line, c_name = callee_key
                    c_label = f'{c_file}:{c_line}:{c_name}'
                    # call_data is (cc, nc, tottime, cumtime).
                    callee_rows.append((
                        str(call_data[1]),
                        f'{call_data[2]:.4f}',
                        f'{call_data[3]:.4f}',
                        c_label,
                    ))
                callee_headers = ['ncalls', 'tottime', 'cumtime', 'callee']
                callee_table = render_top_table(
                    callee_headers, callee_rows, top, 'callees',
                )
                if callee_table:
                    lines.append(callee_table)
        else:
            lines.append('')
            lines.append(f'Function "{focus}" not found in profile.')

    return '\n'.join(lines)


def _render_node_hotspots(
    artifact_dir: str,
) -> list[dict]:
    """Parse all *.cpuprofile files in *artifact_dir* and merge into a single
    sorted list. Returns the merged list of dicts.
    """
    import glob  # noqa: PLC0415 — stdlib, allowed here.

    files = sorted(glob.glob(os.path.join(artifact_dir, '*.cpuprofile')))
    merged: list[dict] = []
    seen: set[str] = set()
    for path in files:
        try:
            items = parse_cpuprofile(path)
        except ValueError:
            continue
        for item in items:
            key = item.get('function', '')
            if key not in seen:
                merged.append(item)
                seen.add(key)
            else:
                # Sum self/total/hits for duplicate function names.
                for existing in merged:
                    if existing['function'] == key:
                        existing['self_s'] += item['self_s']
                        existing['total_s'] += item['total_s']
                        existing['hits'] += item['hits']
                        break
    merged.sort(key=lambda r: r['self_s'], reverse=True)
    return merged


def _render_node_table(
    merged: list[dict], top: int,
) -> str:
    """Render the merged V8 profile data into a fixed-width table."""
    rows: list[tuple] = []
    for item in merged:
        rows.append((
            str(item.get('hits', 0)),
            f'{item.get("self_s", 0.0):.4f}',
            f'{item.get("total_s", 0.0):.4f}',
            item.get('function', '(anonymous)'),
        ))
    headers = ['hits', 'self_s', 'total_s', 'function']
    return render_top_table(headers, rows, top, 'functions')


def _render_php_hotspots(
    artifact_dir: str, top: int,
) -> str:
    """Parse the cachegrind output file in *artifact_dir* and render the
    top-N table ranked by self Time.
    """
    import glob  # noqa: PLC0415

    files = sorted(glob.glob(os.path.join(artifact_dir, 'cachegrind.out.*')))
    if not files:
        return ''
    data = parse_cachegrind(files[0])
    events = data.get('events', ['Time'])
    first_event = events[0] if events else 'Time'

    rows: list[tuple] = []
    for func in data.get('functions', []):
        name = func.get('function', '(unknown)')
        file_ = func.get('file', '')
        self_cost = func.get('self', {}).get(first_event, 0)
        inc_cost = func.get('inclusive', {}).get(first_event, 0)
        calls = func.get('calls', 0)
        label = f'{file_}:{name}' if file_ else name
        rows.append((
            str(calls),
            str(self_cost),
            str(inc_cost),
            label,
        ))

    headers = ['calls', f'self {first_event}', f'inclusive {first_event}', 'function']
    return render_top_table(headers, rows, top, 'functions')


class ProfileHotspots(Tool):
    """Per-function CPU hotspot profiling for python, node, and php.

    Profiles a target script (or snippet) and renders a top-N table of hot
    functions with measured self/cumulative times.  Supports python (cProfile),
    node (V8 --cpu-prof), and php (Xdebug cachegrind).
    """

    name = 'profile_hotspots'
    description = (
        'Profiles a target script (or snippet) and renders a top-N table of '
        'hot functions with measured self/cumulative times.  Supports python '
        '(cProfile), node (V8 --cpu-prof), and php (Xdebug cachegrind).  A '
        'NONZERO exit with an artifact present is still reported as success.'
    )
    action = 'profile the hotspots'
    oversize_hint = 'lower top or focus on a specific function'
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
            'focus': {
                'type': 'string',
                'description': (
                    'Python only: a function name — additionally show its '
                    'callers/callees from pstats data.'
                ),
            },
            'top': {
                'type': 'integer',
                'description': (
                    'Number of top functions to show. Range 1-50. Default 15.'
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
        """Execute the profile-hotspots tool.

        Args:
            language: One of 'python', 'node', 'php' (required).
            target: Project-relative path to a script (optional).
            snippet: Source code to profile (optional).
            args: Arguments forwarded to the profiled program (optional).
            is_module: Python only: run target with -m (optional).
            focus: Python only: function name for callers/callees (optional).
            top: Number of top functions to show (optional, default 15).
            timeout: Max seconds to wait (optional, default 120).

        Returns:
            A ``ToolResult`` with the top-N hotspot table, stdout/stderr tails,
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
        focus_raw = kwargs.get('focus')
        focus = focus_raw if isinstance(focus_raw, str) else None
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

        if language != 'python' and (is_module or focus):
            return ToolResult.err(
                'is_module and focus are only valid for python.',
                code='bad-arguments',
                hint='Set language to "python" to use is_module or focus.',
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
            suffix = _SNIPPET_SUFFIXES.get(language, '')
            fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix='profile_hotspots_')
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
                if is_module:
                    cmd = shlex.join([
                        interpreter, '-m', 'cProfile', '-o',
                        os.path.join(artifact_dir, 'prof.out'),
                        '-m', script_path,
                    ] + args)
                else:
                    cmd = shlex.join([
                        interpreter, '-m', 'cProfile', '-o',
                        os.path.join(artifact_dir, 'prof.out'),
                        script_path,
                    ] + args)
                artifact_pattern = os.path.join(artifact_dir, 'prof.out')

            elif language == 'node':
                cmd = shlex.join([
                    interpreter, '--cpu-prof', '--cpu-prof-dir',
                    artifact_dir, script_path,
                ] + args)
                artifact_pattern = os.path.join(artifact_dir, '*.cpuprofile')

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
                stderr_tail = tail_lines(result.stderr)
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
                has_artifact = os.path.isfile(artifact_pattern)
            elif language == 'node':
                has_artifact = bool(_glob.glob(artifact_pattern))
            else:
                has_artifact = bool(_glob.glob(artifact_pattern))

            if not has_artifact:
                stderr_tail = tail_lines(result.stderr)
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
                table = _render_python_hotspots(artifact_pattern, top, focus)
                lines.append(table)

            elif language == 'node':
                merged = _render_node_hotspots(artifact_dir)
                table = _render_node_table(merged, top)
                lines.append(table)
                lines.append('')
                lines.append(
                    'Note: V8 --cpu-prof data is sampling-based and '
                    'approximate, unlike python/php exact counts.'
                )

            else:  # php
                table = _render_php_hotspots(artifact_dir, top)
                lines.append(table)

            if mutation_line:
                lines.append('')
                lines.append(mutation_line)

            # --- Stdout/stderr tails ---
            stdout_tail = tail_lines(result.stdout)
            stderr_tail = tail_lines(result.stderr)
            streams = format_streams(stdout_tail, stderr_tail)
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
