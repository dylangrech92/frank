"""Shared lint-runner layer used by the ``lint`` tool (and, later, reactive
lint-delta injection in ``agent.py``).

The leading underscore keeps this module out of tool discovery by the registry
(mirrors ``tools/_sandbox.py``).

Exports the pinned interface other slices build against::

    run_lint(paths, project_root) -> LintReport
    LintReport.issues: list[LintIssue]        # path, line, col, rule, severity, message, source
    LintReport.unavailable: dict[str, str]    # language -> reason (missing binary etc.)

Do not change this shape without updating every caller.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from runtime.process import run_one_shot

# File extension (lowercase, with leading dot) -> language key used to look up
# the `linters` config block and to select a result parser.
EXTENSION_LANGUAGE: dict[str, str] = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.php': 'php',
}

# Fallback linter commands used when config.json has no (or an incomplete)
# `linters` block. Kept in sync with the defaults documented in config.json.
DEFAULT_LINTERS: dict[str, dict[str, list[str]]] = {
    'python': {'command': ['ruff', 'check', '--output-format', 'json']},
    'javascript': {'command': ['eslint', '--format', 'json']},
    'typescript': {'command': ['eslint', '--format', 'json']},
    'php': {'command': ['phpstan', 'analyse', '--error-format', 'json', '--no-progress', '--level', '0']},
}

# Directory names skipped while scanning for languages present in a target
# directory (whole-project / subdirectory lint modes).
_SKIP_DIRS = frozenset({
    '.git', 'node_modules', 'venv', '.venv', '__pycache__', 'dist', 'build',
    'vendor', '.tox', '.mypy_cache', '.pytest_cache', '.ruff_cache',
})

_LINT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class LintIssue:
    """A single normalized lint finding."""

    path: str  # project-root-relative path
    line: int
    col: int
    rule: str
    severity: str  # 'error' | 'warning' | 'info' | 'hint'
    message: str
    source: str  # linter name, e.g. 'ruff', 'eslint', 'phpstan'


@dataclass(slots=True)
class LintReport:
    """The result of a :func:`run_lint` call."""

    issues: list[LintIssue] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)


def _relpath(filename: str, project_root: Path) -> str:
    """Return *filename* relative to *project_root*, falling back to *filename* as-is."""
    try:
        return os.path.relpath(filename, project_root)
    except ValueError:
        return filename


def _parse_ruff(stdout: str, project_root: Path) -> list[LintIssue]:
    """Parse ``ruff check --output-format json`` output."""
    data = json.loads(stdout) if stdout.strip() else []
    if not isinstance(data, list):
        raise ValueError('unexpected ruff output shape (expected a JSON array)')

    issues: list[LintIssue] = []
    for item in data:
        loc = item.get('location') or {}
        issues.append(LintIssue(
            path=_relpath(str(item.get('filename', '')), project_root),
            line=int(loc.get('row') or 0),
            col=int(loc.get('column') or 0),
            rule=str(item.get('code') or ''),
            severity=str(item.get('severity') or 'error').lower(),
            message=str(item.get('message') or ''),
            source='ruff',
        ))
    return issues


def _parse_eslint(stdout: str, project_root: Path) -> list[LintIssue]:
    """Parse ``eslint --format json`` output."""
    data = json.loads(stdout) if stdout.strip() else []
    if not isinstance(data, list):
        raise ValueError('unexpected eslint output shape (expected a JSON array)')

    severity_map = {1: 'warning', 2: 'error'}
    issues: list[LintIssue] = []
    for file_entry in data:
        rel = _relpath(str(file_entry.get('filePath', '')), project_root)
        for m in file_entry.get('messages') or []:
            issues.append(LintIssue(
                path=rel,
                line=int(m.get('line') or 0),
                col=int(m.get('column') or 0),
                rule=str(m.get('ruleId') or 'syntax-error'),
                severity=severity_map.get(m.get('severity'), 'error'),
                message=str(m.get('message') or ''),
                source='eslint',
            ))
    return issues


def _parse_phpstan(stdout: str, project_root: Path) -> list[LintIssue]:
    """Parse ``phpstan analyse --error-format json`` output."""
    data = json.loads(stdout) if stdout.strip() else {}
    if not isinstance(data, dict):
        raise ValueError('unexpected phpstan output shape (expected a JSON object)')

    files = data.get('files') or {}
    if not isinstance(files, dict):
        raise ValueError('unexpected phpstan output shape (missing "files" object)')

    issues: list[LintIssue] = []
    for filepath, info in files.items():
        rel = _relpath(str(filepath), project_root)
        for m in (info or {}).get('messages') or []:
            issues.append(LintIssue(
                path=rel,
                line=int(m.get('line') or 0),
                col=0,
                rule=str(m.get('identifier') or 'phpstan'),
                severity='error',
                message=str(m.get('message') or ''),
                source='phpstan',
            ))
    return issues


_PARSERS: dict[str, Callable[[str, Path], list[LintIssue]]] = {
    'python': _parse_ruff,
    'javascript': _parse_eslint,
    'typescript': _parse_eslint,
    'php': _parse_phpstan,
}


def _load_linters_config(project_root: Path) -> dict[str, dict]:
    """Load the ``linters`` config block for *project_root*.

    Reads ``config.json`` (or the path in ``CODING_AGENT_CONFIG``) the same
    way ``tools/run_tests.py`` reads ``test_runners`` -- directly from the
    JSON file rather than through the ``Config`` dataclass, so this module has
    no import-time dependency on the harness's config-loading path. Missing
    file, missing key, or a malformed block all fall back to
    :data:`DEFAULT_LINTERS`; entries present in the file override the default
    for that language only.
    """
    config_path = Path(os.environ.get('CODING_AGENT_CONFIG', 'config.json'))
    if not config_path.is_absolute():
        config_path = project_root / config_path

    configured: dict = {}
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            block = data.get('linters')
            if isinstance(block, dict):
                configured = block
        except (OSError, ValueError):
            configured = {}

    merged = {lang: dict(entry) for lang, entry in DEFAULT_LINTERS.items()}
    for lang, entry in configured.items():
        if isinstance(entry, dict) and isinstance(entry.get('command'), list):
            merged[lang] = entry
    return merged


def _detect_languages_in_dir(directory: Path) -> set[str]:
    """Walk *directory* (skipping common noise dirs) and return languages present."""
    found: set[str] = set()
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith('.')]
        for fname in files:
            lang = EXTENSION_LANGUAGE.get(Path(fname).suffix.lower())
            if lang:
                found.add(lang)
        if found >= set(_PARSERS):
            break  # every lintable language already found; no need to keep walking
    return found


def run_lint(paths: list[str] | None, project_root: str) -> LintReport:
    """Run configured linters over *paths* (or the whole project) and normalize results.

    Args:
        paths: File and/or directory paths (absolute, or relative to
            *project_root*) to lint. ``None`` or an empty list lints the
            entire project.
        project_root: The project root; linter subprocesses run with this as
            their working directory, and every reported path is relative to it.

    Returns:
        A :class:`LintReport` with normalized ``issues`` and an
        ``unavailable`` map of language -> reason (e.g. a missing binary,
        a timeout, or unparseable linter output) for languages that were
        detected but could not be linted.
    """
    project_root_path = Path(project_root).resolve()
    linters_cfg = _load_linters_config(project_root_path)

    targets = paths if paths else [str(project_root_path)]

    # Group every target by the language(s) it contributes, so each linter
    # runs once per call across all of that language's targets.
    lang_targets: dict[str, list[str]] = {}
    for target in targets:
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = project_root_path / target_path
        target_path = target_path.resolve()

        if target_path.is_dir():
            for lang in _detect_languages_in_dir(target_path):
                rel = _relpath(str(target_path), project_root_path)
                lang_targets.setdefault(lang, [])
                if rel not in lang_targets[lang]:
                    lang_targets[lang].append(rel)
        elif target_path.is_file():
            lang = EXTENSION_LANGUAGE.get(target_path.suffix.lower())
            if lang is None:
                continue  # unsupported extension -- not a lint failure
            rel = _relpath(str(target_path), project_root_path)
            lang_targets.setdefault(lang, [])
            if rel not in lang_targets[lang]:
                lang_targets[lang].append(rel)
        # nonexistent targets are silently skipped; the caller (the `lint`
        # tool) is responsible for existence validation and user-facing errors

    issues: list[LintIssue] = []
    unavailable: dict[str, str] = {}

    for lang, rel_targets in lang_targets.items():
        entry = linters_cfg.get(lang)
        if entry is None:
            continue  # no linter configured for this language at all

        command = entry.get('command')
        if not command:
            unavailable[lang] = 'no command configured'
            continue

        binary = command[0]
        binary_path = shutil.which(binary)
        if binary_path is None and not (os.path.isabs(binary) and os.path.exists(binary)):
            unavailable[lang] = f"'{binary}' not found on PATH"
            continue

        cmd_tokens = list(command) + rel_targets
        cmd_str = ' '.join(shlex.quote(tok) for tok in cmd_tokens)

        result = run_one_shot(cmd_str, str(project_root_path), timeout_seconds=_LINT_TIMEOUT_SECONDS)
        if result.get('timed_out'):
            unavailable[lang] = f"'{binary}' timed out after {_LINT_TIMEOUT_SECONDS}s"
            continue

        parser = _PARSERS.get(lang)
        if parser is None:
            unavailable[lang] = f'no result parser for language {lang!r}'
            continue

        try:
            lang_issues = parser(str(result.get('stdout', '')), project_root_path)
        except (ValueError, TypeError, KeyError):
            stderr_text = str(result.get('stderr', '') or '')
            first_line = next((ln for ln in stderr_text.splitlines() if ln.strip()), '')
            reason = first_line or f"'{binary}' exited {result.get('exit_code')} with unparseable output"
            unavailable[lang] = reason
            continue

        issues.extend(lang_issues)

    return LintReport(issues=issues, unavailable=unavailable)
