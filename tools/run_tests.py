"""Run-tests tool: detect the test framework for a directory and run its tests.

Detects pytest, jest, or phpunit based on the project layout, then delegates to the
corresponding runner backend. Returns structured per-test results with pass/fail/status.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from runtime.tests import detect_framework, run_framework
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult

# Unexecutable / non-testable file suffixes.
_NO_TEST_SUFFICES = frozenset({'.html', '.htm', '.css', '.scss'})


class RunTests(Tool):
    """Detect the test framework for a directory and run its tests, returning structured per-test results.

    The ``path`` parameter must be relative to the project root and point to an
    existing file or directory inside the sandbox.  When omitted or empty it defaults
    to the project root (``Path.cwd()``).

    The optional ``pattern`` parameter filters which tests are run â€” exact interpretation
    depends on the framework (-k for pytest, -t for jest, --filter for phpunit).
    """

    name = 'run_tests'
    description = (
        'Detect the test framework (pytest, jest, or phpunit) for a directory '
        'and run its tests, returning structured per-test results. Optional pattern '
        'filters tests by name (-k / -t / --filter).'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Directory (or file) relative to the project root to run tests for. Defaults to the project root.',
            },
            'pattern': {
                'type': 'string',
                'description': 'Only run tests whose name matches this substring/expression.',
            },
        },
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the run-tests tool.

        Args:
            **kwargs: Parsed from LLM function-call payload.

        Returns:
            A ``ToolResult`` containing structured test output on success or an
            error message if the framework could not be detected or the path is invalid.
        """
        # --- 1. Read raw parameters ----------------------------------------
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else '.'
        if not raw_path:
            raw_path = '.'

        pattern_raw = kwargs.get('pattern') if isinstance(kwargs.get('pattern'), str) else ''
        if not pattern_raw:
            pattern_raw = None

        # --- 2. Resolve path ------------------------------------------------
        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        if not resolved.exists():
            return ToolResult.err(f'{resolved} does not exist.', code='not-found')

        # --- 3. Non-executable-language refusal -----------------------------
        if resolved.is_file():
            suffix = resolved.suffix.lower()
            if suffix in _NO_TEST_SUFFICES:
                dot_suffix = suffix.lstrip('.')
                return ToolResult.err(
                    f'no test framework for this language ({dot_suffix}); '
                    'html/css/scss files are not executable.',
                    code='no-test-framework',
                )

        # --- 4. Determine target directory ----------------------------------
        if resolved.is_dir():
            target_dir = str(resolved)
        else:
            target_dir = str(resolved.parent)

        # --- 5. Load test_runners config ------------------------------------
        config_path = Path(os.environ.get('CODING_AGENT_CONFIG', 'config.json'))
        test_runners: dict[str, Any] | None = None

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as fh:
                    cfg = json.load(fh)
                test_runners = cfg.get('test_runners')
            except (OSError, ValueError):
                test_runners = None

        # --- 6. Detect framework --------------------------------------------
        framework, reason = detect_framework(target_dir, str(Path.cwd()), test_runners)
        if framework is None:
            return ToolResult.err(
                f'no test framework detected: {reason}',
                code='no-test-framework',
            )

        # --- 7. Run the framework -------------------------------------------
        result = run_framework(framework, target_dir, str(Path.cwd()), pattern=pattern_raw, timeout=180)

        # --- 8. Build body --------------------------------------------------
        s = result['summary']
        header = (
            f"{framework}: {s['passed']} passed, {s['failed']} failed, "
            f"{s['skipped']} skipped, {s['errors']} errors ({s['total']} total)"
        )

        lines: list[str] = [header]

        for t in result.get('tests', []):
            if t['status'] == 'passed':
                continue
            lines.append(f"{t['status'].upper()} {t['name']}")
            msg = t.get('message', '') or ''
            if msg:
                lines.append(f'  {msg[:400]}')

        note = result.get('note', '') or ''
        if note:
            lines.append(f'note: {note}')

        degraded = result.get('degraded', False)
        if degraded and 'degraded' not in note.lower():
            lines.append('note: degraded parse \u2014 per-test detail may be incomplete.')

        body = '\n'.join(lines)

        # --- 9. Return ------------------------------------------------------
        return ToolResult.ok(
            body,
            framework=framework,
            degraded=degraded,
            passed=s['passed'],
            failed=s['failed'],
            skipped=s['skipped'],
            errors=s['errors'],
            total=s['total'],
        )

    # --- exception guard: never let an error escape ---------------------------

    # Note: the above method is intentionally written to never let exceptions
    # propagate. Every step returns a ToolResult directly.  Should anything truly
    # unexpected slip through (e.g. a bug in detect_framework or run_framework),
    # the caller wraps it; we do not need extra machinery here.
