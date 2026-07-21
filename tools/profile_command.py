"""Profile command tool: run a shell command under resource measurement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.process import is_denied
from runtime.profiling import fmt_bytes, fmt_seconds, run_measured
from tools._snapshot import publish_snapshot_diff, render_mutation_line, snapshot_tree
from tools.base import Tool
from tools.result import ToolResult


def _format_streams(stdout_text: str, stderr_text: str) -> str:
    """Render captured stdout/stderr with section labels.

    The ``--- stderr ---`` section is only emitted when *stderr_text* is
    non-empty, so a command that produced no error output does not carry an
    empty labelled block.
    """
    if stderr_text:
        return f'--- stdout ---\n{stdout_text}\n--- stderr ---\n{stderr_text}'
    return f'--- stdout ---\n{stdout_text}'


class ProfileCommand(Tool):
    """Runs a shell command under resource measurement.

    The *cmd* argument is passed to ``shell=True`` via ``subprocess.Popen``, so
    all standard shell features (pipes, redirects, globs, variable expansion)
    are available.  A deny-list check blocks hazardous commands before execution.

    The tool reports wall time, user/sys CPU, peak RSS, sampled process-tree
    RSS/CPU, and process count.  When *repeats* is greater than 1, each run is
    measured independently and the result includes per-run wall times along
    with a median and spread, giving the model a stable picture of timing.

    Commands always run with the project root as the working directory, so a
    cd into the project is never needed.  The tree is snapshot once before all
    runs and once after, so files the profiled command writes are published on
    the mutation bus and named in the result without paying a per-repeat walk
    tax.
    """

    name = 'profile_command'
    summary = 'Run a shell command under resource measurement.'
    description = (
        'Runs a shell command under measurement and reports wall time, user/sys '
        'CPU, peak RSS, sampled process-tree RSS/CPU, and process count. '
        'Commands always run with the project root as the working directory, '
        'so a cd into the project is never needed.  When repeats>1 the result '
        'reports per-run walls with median and spread.'
    )
    action = 'measure the command'
    oversize_hint = 'pipe the output through head/tail or redirect it to a file and read a slice'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'cmd': {
                'type': 'string',
                'description': 'The shell command to execute.',
            },
            'repeats': {
                'type': 'integer',
                'description': (
                    'Number of measured runs. Default is 1. Use >1 for stable '
                    'wall time measurements.'
                ),
            },
            'timeout': {
                'type': 'integer',
                'description': (
                    'Maximum seconds to wait per run. Default is 120.'
                ),
            },
        },
        'required': ['cmd'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute a shell command under resource measurement.

        Args:
            cmd: The shell command to execute (required).
            repeats: Number of measured runs (optional, default 1, range 1-5).
            timeout: Max seconds per run (optional, default 120).

        Returns:
            A ``ToolResult`` with measured metrics, stdout/stderr tails, and
            any file mutations the command produced.
        """
        raw_cmd = kwargs.get('cmd')
        cmd = raw_cmd if isinstance(raw_cmd, str) else ''
        repeats: int = kwargs.get('repeats') if isinstance(kwargs.get('repeats'), int) else 1
        timeout: int = kwargs.get('timeout') if isinstance(kwargs.get('timeout'), int) else 120

        # --- Argument validation ---
        if not cmd:
            return ToolResult.err(
                'Missing or non-string "cmd" argument.',
                code='bad-arguments',
                hint='Provide a non-empty string for the cmd parameter.',
            )
        if repeats is None or not (1 <= repeats <= 5):
            return ToolResult.err(
                'repeats must be an integer between 1 and 5.',
                code='bad-arguments',
                hint='Set repeats to an integer in the range 1..5.',
            )
        if timeout <= 0:
            return ToolResult.err(
                'timeout must be a positive integer.',
                code='bad-arguments',
                hint='Set timeout to a positive integer (seconds).',
            )

        # --- Deny-list gate ---
        deny_reason = is_denied(cmd)
        if deny_reason is not None:
            return ToolResult.err(
                f'Destructive command blocked ({deny_reason}).',
                code='destructive-command-blocked',
                hint='Review the command and remove any patterns that match the deny-list.',
            )

        root = Path.cwd()
        before = snapshot_tree(root)

        # --- Execute measured runs ---
        runs = []
        timed_out = False
        for i in range(1, repeats + 1):
            result = run_measured(cmd, str(root), timeout_seconds=timeout)
            runs.append(result)
            if result.timed_out:
                timed_out = True
                break

        # Publish snapshot-diff mutation events once around all runs.
        mutation_line = ''
        if before is not None:
            after = snapshot_tree(root)
            if after is not None:
                created, deleted, changed = publish_snapshot_diff(before, after)
                mutation_line = render_mutation_line(root, created, deleted, changed)

        # --- Timeout path ---
        if timed_out:
            last = runs[-1]
            lines = ['Command timed out. Per-run wall times measured so far:']
            for idx, r in enumerate(runs, 1):
                lines.append(
                    f'run {idx}: wall '
                    f'{fmt_seconds(r.wall_s)}'
                )
            if mutation_line:
                lines.append(mutation_line)
            stdout_tail = '\n'.join(str(last.stdout).splitlines()[-20:])
            stderr_tail = '\n'.join(str(last.stderr).splitlines()[-20:])
            streams = _format_streams(stdout_tail, stderr_tail)
            lines.append(streams)
            return ToolResult.err(
                '\n'.join(lines),
                code='timeout',
                hint=f'Each run was allowed {timeout}s. Try a longer timeout or a smaller command.',
            )

        # --- Success path: render metrics ---
        lines = ['Measured resource usage:']
        for idx, r in enumerate(runs, 1):
            lines.append(
                f'run {idx}: wall '
                f'{fmt_seconds(r.wall_s)}, cpu user '
                f'{fmt_seconds(r.cpu_user_s)}, sys '
                f'{fmt_seconds(r.cpu_sys_s)}, peak rss '
                f'{fmt_bytes(r.max_rss_bytes)}'
            )

        # Median and spread across runs.
        if len(runs) > 1:
            walls = [r.wall_s for r in runs]
            sorted_walls = sorted(walls)
            n = len(sorted_walls)
            if n % 2 == 1:
                median = sorted_walls[n // 2]
            else:
                median = (sorted_walls[n // 2 - 1] + sorted_walls[n // 2]) / 2.0
            spread = (max(walls) - min(walls)) / median * 100 if median else 0.0
            lines.append(f'median wall {fmt_seconds(median)}, spread {spread:.1f}%')

        # Sampled process-tree data from the LAST run.
        last = runs[-1]
        sampled_peak = last.sampled_peak_rss_bytes or 0
        if sampled_peak > 0:
            mean_cpu_pct = last.sampled_mean_cpu_pct or 0.0
            max_procs = last.max_procs or 0
            lines.append(
                f'sampled: peak tree rss {fmt_bytes(sampled_peak)}, '
                f'mean cpu {mean_cpu_pct:.1f}%, max processes {max_procs}'
            )

        if mutation_line:
            lines.append(mutation_line)

        # Stdout/stderr tails from the LAST run, capped to 20 lines.
        stdout_tail = '\n'.join(str(last.stdout).splitlines()[-20:])
        stderr_tail = '\n'.join(str(last.stderr).splitlines()[-20:])
        streams = _format_streams(stdout_tail, stderr_tail)
        lines.append(streams)

        # Grounding line for nonzero exit codes.
        exit_codes = [r.exit_code for r in runs]
        exit_code = exit_codes[-1]
        if len(set(exit_codes)) > 1:
            # Different exit codes across runs — name each.
            exit_line = ', '.join(
                f'run {idx}: {r.exit_code}'
                for idx, r in enumerate(runs, 1)
            )
            lines.append(
                f'(exit codes {exit_line} — this command ran in working directory '
                f'{root}; commands always run there, a cd into the project is '
                f'never needed)'
            )
        elif exit_code != 0:
            lines.append(
                f'(exit code {exit_code} — this command ran in working directory '
                f'{root}; commands always run there, a cd into the project is '
                f'never needed)'
            )

        return ToolResult.ok(
            '\n'.join(lines),
            exit_code=exit_code,
            timed_out=False,
        )
