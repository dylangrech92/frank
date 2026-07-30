"""Run-tests tool: detect the test framework for a directory and run its tests.

Detects pytest, jest, or phpunit based on the project layout, then delegates to the
corresponding runner backend. Returns structured per-test results with pass/fail/status.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from runtime.tests import detect_framework, run_framework
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult

# Unexecutable / non-testable file suffixes.
_NO_TEST_SUFFICES = frozenset({'.html', '.htm', '.css', '.scss'})

# Matches one entry inside pytest's "warnings summary" section, e.g.:
#   /path/to/mod.py:5: DeprecationWarning: old_api is deprecated
_DEPRECATION_WARNING_RE = re.compile(
    r'^\s*(\S+:\d+):\s*(DeprecationWarning|PendingDeprecationWarning):\s*(.+?)\s*$'
)

# Cap on how many unique deprecation entries are rendered per run.
_DEPRECATIONS_CAP = 10


def _parse_pytest_deprecations(output_tail: str, cap: int = _DEPRECATIONS_CAP) -> list[str]:
    """Extract unique Deprecation/PendingDeprecationWarning entries from pytest output.

    Pytest renders a ``warnings summary`` section near the end of its output with
    one indented ``<path>:<line>: <Category>: <message>`` line per warning
    occurrence (preceded by the originating test's node id). This scans that
    section only for the two deprecation categories -- other warning categories
    (e.g. ``UserWarning``) are intentionally left out of scope for this tool.
    This never adds ``-W error``; it only reads whatever pytest already printed.

    Args:
        output_tail: Tail of the combined stdout+stderr from a pytest run.
        cap: Maximum number of unique entries to return (default 10).

    Returns:
        A list of ``"<origin> <category>: <message>"`` strings, deduplicated
        and capped at *cap* entries, in first-seen order. Empty when no
        deprecation warnings are present.
    """
    if 'warnings summary' not in output_tail:
        return []

    seen: dict[str, None] = {}
    for line in output_tail.splitlines():
        match = _DEPRECATION_WARNING_RE.match(line)
        if not match:
            continue

        origin, category, message = match.group(1), match.group(2), match.group(3)
        entry = f'{origin} {category}: {message[:200]}'
        if entry not in seen:
            seen[entry] = None
            if len(seen) >= cap:
                break

    return list(seen)


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
    action = 'run the tests'
    oversize_hint = 'use pattern to run a subset'
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

        if framework == 'pytest':
            deprecations = _parse_pytest_deprecations(str(result.get('output_tail', '') or ''))
            if deprecations:
                lines.append('deprecations:')
                for entry in deprecations:
                    lines.append(f'  {entry}')
                if len(deprecations) >= _DEPRECATIONS_CAP:
                    lines.append(f'  ... capped at {_DEPRECATIONS_CAP} unique entries')

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
