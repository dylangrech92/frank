"""Lint tool: run configured linters over one file or the whole project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools._lint import LintIssue, run_lint
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult

# Hard cap on rendered issue lines so a noisy project can never blow the
# context budget; anything past this is summarized in a "+N more" tail line.
_MAX_RENDERED_ISSUES = 150

_SEVERITY_RANK = {'error': 0, 'warning': 1, 'info': 2, 'hint': 3}


def _sort_key(issue: LintIssue) -> tuple[int, str, int, int]:
    return (_SEVERITY_RANK.get(issue.severity, 4), issue.path, issue.line, issue.col)


class Lint(Tool):
    """Runs configured linters (ruff/eslint/phpstan by default) over a file or the project.

    Results are normalized to one line per issue::

        <path>:<line>:<col> <rule> <severity> <message>

    sorted by severity (errors first), then path/line/col, and capped at a
    fixed count with a ``+N more`` tail line so output can never blow the
    context budget. When a configured language's linter binary is missing
    (or its output can't be parsed), that language is reported separately
    as unavailable rather than failing the whole call.
    """

    name = 'lint'
    description = (
        'Run configured static-analysis linters over a single file or the whole '
        'project. Defaults: ruff for Python, eslint for JavaScript/TypeScript, '
        'phpstan for PHP (see the `linters` config block). Returns normalized, '
        'severity-sorted issue lines capped at a fixed count. A language whose '
        'linter binary is missing (or whose output could not be parsed) is '
        'reported separately instead of failing the call.'
    )
    action = 'lint the file or project'
    oversize_hint = 'pass path to scope the lint to a single file or subdirectory'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': (
                    'File or directory relative to the project root to lint. '
                    'Defaults to the whole project when omitted.'
                ),
            },
        },
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the lint tool.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects optional
                ``path`` (str) -- a file or directory relative to the project
                root; defaults to the whole project.

        Returns:
            A ``ToolResult`` with the normalized, severity-sorted issue lines
            as its body on success (including an "unavailable" section when
            some configured language could not be linted), or an error when
            *path* escapes the project root or does not exist.
        """
        root = Path.cwd()
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''

        target_paths: list[str] | None = None
        if raw_path:
            try:
                resolved = resolve_in_root(root, raw_path)
            except ValueError as exc:
                return ToolResult.err(str(exc), code='path-escapes-root')

            if not resolved.exists():
                return ToolResult.err(f'{raw_path} does not exist.', code='not-found')

            target_paths = [str(resolved)]

        report = run_lint(target_paths, str(root))

        lines: list[str] = []

        sorted_issues = sorted(report.issues, key=_sort_key)
        shown = sorted_issues[:_MAX_RENDERED_ISSUES]
        for issue in shown:
            lines.append(
                f'{issue.path}:{issue.line}:{issue.col} {issue.rule} '
                f'{issue.severity} {issue.message}'
            )

        remaining = len(sorted_issues) - len(shown)
        if remaining > 0:
            lines.append(f'+{remaining} more')

        if report.unavailable:
            lines.append('unavailable:')
            for lang in sorted(report.unavailable):
                lines.append(f'  {lang}: {report.unavailable[lang]}')

        if not lines:
            return ToolResult.ok('no issues', count=0)

        return ToolResult.ok(
            '\n'.join(lines),
            count=len(report.issues),
            unavailable_count=len(report.unavailable),
        )
