"""Find-dead-code tool: sweep a Python file/project for unused code via vulture."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult

# Directories never walked while sniffing a directory target for Python files.
_IGNORED_DIRS = frozenset({'.git', '.coding_agent', '__pycache__', 'node_modules', '.venv', 'venv'})

# vulture's Item.get_report() renders exactly:
#   "{path}:{lineno}: {message} ({confidence}% confidence[, N lines])"
# where message defaults to "unused {typ} '{name}'" but is a free-form string
# for the "unreachable_code" item type (e.g. "unreachable code after 'return'").
_VULTURE_LINE_RE = re.compile(
    r"^(?P<path>.+):(?P<line>\d+):\s(?P<message>.+?)\s"
    r"\((?P<conf>\d+)% confidence(?:,\s\d+\s(?:line|lines))?\)$"
)


def _dir_has_python(directory: Path) -> bool:
    """Return True as soon as a ``.py`` file is found under *directory*.

    Skips common vendored/build/vcs directories so the sniff stays cheap.
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for fn in filenames:
            if Path(fn).suffix == '.py':
                return True
    return False


def _parse_vulture_output(stdout: str) -> list[dict[str, Any]]:
    """Parse vulture's line-oriented report into structured findings.

    Args:
        stdout: Raw stdout from a vulture invocation.

    Returns:
        A list of dicts with keys ``path``, ``line``, ``message``, ``confidence``.
        Lines that do not match vulture's report format (e.g. stray warnings) are
        silently skipped rather than raising — this parser must never crash on
        unexpected vulture output.
    """
    findings: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        match = _VULTURE_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        findings.append({
            'path': match.group('path'),
            'line': int(match.group('line')),
            'message': match.group('message'),
            'confidence': int(match.group('conf')),
        })
    return findings


class FindDeadCode(Tool):
    """Sweeps a Python file or the project for unused/unreachable code via vulture.

    Runs `vulture <https://github.com/jendrikseipp/vulture>`_ (when installed on
    ``PATH``, never auto-installed) against a Python file or a directory that
    contains Python files. Findings are reported as *potentially* dead —
    dynamic dispatch, exports, and reflection can keep "unused" code alive, so
    verify before deleting. Non-Python targets have no adapter here.
    """

    name = 'find_dead_code'
    parallel_safe = True  # spawns a read-only subprocess
    summary = 'Sweep a Python file/project for unused/unreachable code (vulture).'
    description = (
        'Sweeps a Python file or the whole project for dead code using vulture. '
        'Findings are reported as potentially dead — verify before deleting. '
        'Non-Python targets are not supported.'
    )
    action = 'sweep for dead code'
    oversize_hint = 'narrow the scan to a subdirectory or a single file'
    alternative = 'find_references for a single known symbol'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': (
                    'File or directory to scan, relative to the project root. '
                    'Defaults to the whole project.'
                ),
            },
        },
        'required': [],
    }

    MAX_FINDINGS = 200

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the dead-code sweep.

        Args:
            **kwargs: Parsed from LLM function-call payload. Optional ``path``
                (str, default the whole project).

        Returns:
            A ``ToolResult`` with a rendered findings list (or a "none found"
            message) on success, or an error when the target is not Python,
            vulture is not installed, or the scan itself fails.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else '.'

        root = Path.cwd()
        try:
            target = resolve_in_root(root, raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        if not target.exists():
            return ToolResult.err(
                f'{target} does not exist.',
                code='not-found',
            )

        is_py = (
            target.suffix == '.py' if target.is_file()
            else _dir_has_python(target)
        )
        if not is_py:
            return ToolResult.err(
                'no dead-code adapter applies to this target (no Python files found)',
                code='no-adapter',
                hint=(
                    'This tool sweeps Python only. For a TS/JS project, run knip '
                    'or ts-prune via the run_command tool.'
                ),
            )

        vulture_bin = shutil.which('vulture')
        if vulture_bin is None:
            return ToolResult.err(
                'the vulture binary was not found on PATH',
                code='missing-engine',
                hint='Install it with: pip install vulture',
            )

        return self._run_vulture(vulture_bin, target, root)

    def _run_vulture(self, vulture_bin: str, target: Path, root: Path) -> ToolResult:
        """Run vulture against *target* and render its findings.

        Args:
            vulture_bin: Absolute path to the vulture executable.
            target: Resolved absolute path to scan (file or directory).
            root: Project root, used as the subprocess cwd so vulture emits
                root-relative paths.

        Returns:
            A ``ToolResult`` with rendered findings, a "no dead code" message,
            or an error when vulture itself fails (bad syntax, missing target).
        """
        rel = os.path.relpath(str(target), root) or '.'

        result = subprocess.run(
            [vulture_bin, rel],
            capture_output=True,
            text=True,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
        )

        # vulture exit codes: 0 = clean, 3 = findings reported, anything else
        # (1 typically) = a scan error (missing target, syntax error, ...).
        if result.returncode not in (0, 3):
            message = result.stdout.strip() or result.stderr.strip() or (
                f'vulture exited with code {result.returncode}'
            )
            return ToolResult.err(
                f'vulture scan failed: {message}',
                code='dead-code-scan-failed',
            )

        if result.returncode == 0:
            return ToolResult.ok(
                f'no dead code found by vulture in {rel}',
                path=rel,
                adapter='vulture',
                count=0,
            )

        findings = _parse_vulture_output(result.stdout)
        if not findings:
            # Findings were reported (exit 3) but the report format didn't match
            # what we parse for — surface the raw output rather than hide it.
            return ToolResult.ok(
                result.stdout.strip() or 'vulture reported findings in an unrecognized format',
                path=rel,
                adapter='vulture',
            )

        total = len(findings)
        capped = findings[: self.MAX_FINDINGS]
        lines = [
            f"{f['path']}:{f['line']}: {f['message']} ({f['confidence']}% confidence)"
            for f in capped
        ]
        if total > len(capped):
            lines.append(
                f'-- showing {len(capped)} of {total} findings; '
                f'narrow path to see the rest'
            )
        else:
            lines.append(f"-- {total} finding" + ('' if total == 1 else 's'))

        return ToolResult.ok(
            '\n'.join(lines),
            path=rel,
            adapter='vulture',
            count=total,
        )
