"""Find files tool: search for files by name or glob pattern using ripgrep."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._sandbox import resolve_in_root
from tools._rg import IGNORE_GLOB, locate_rg, strip_dot_prefix


# Characters whose presence marks a pattern as a real glob rather than a plain
# substring; when none appear the pattern is wrapped as *pattern* so a literal-
# minded model still gets useful matches.
_GLOB_METACHARS = frozenset('*?[]{}')

# Maximum number of paths returned; beyond this the model should narrow the
# pattern rather than drown in results.
_MAX_RESULTS = 200


class FindFiles(Tool):
    """Finds files by name or glob pattern, respecting gitignore rules.

    Matches file *paths* (not contents) case-insensitively against a glob such
    as ``*scheduler*``, ``*.sh`` or ``src/**/*.ts``, returning paths relative to
    the project root.  A pattern with no glob metacharacters is treated as a
    substring (wrapped as ``*pattern*``).  The *path* parameter scopes the
    search to a subdirectory.

    For file contents use find; for code symbols use find_symbol.
    """

    name = 'find_files'
    description = (
        'Find files by NAME/glob pattern (gitignore-aware), returning paths '
        'relative to the project root. Use find to search file contents, '
        'find_symbol for code symbols.'
    )
    action = 'find files'
    oversize_hint = 'narrow the pattern or pass path to limit the scope'
    alternative = 'find (file contents) or find_symbol (code symbols)'
    parallel_safe = True  # spawns its own ripgrep subprocess, reads only
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'pattern': {
                'type': 'string',
                'description': (
                    'Filename glob to match case-insensitively, e.g. '
                    '"*scheduler*", "*.sh" or "src/**/*.ts". A pattern with no '
                    'glob metacharacters is matched as a substring.'
                ),
            },
            'path': {
                'type': 'string',
                'description': (
                    'Subdirectory relative to the project root to limit the '
                    'search to. Must be a relative path.'
                ),
            },
        },
        'required': ['pattern'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the find_files tool, listing files whose path matches *pattern*.

        Args:
            **kwargs: Parsed from the LLM function-call payload.  Expects
                *pattern* (required glob) and optional *path* (subdirectory to
                scope the search).

        Returns:
            On success, a ``ToolResult.ok`` whose body is the newline-joined
            matching paths (relative to the project root, sorted), or a clear
            no-matches message.  On failure, a ``ToolResult.err`` with a
            kebab-case error code.
        """
        raw_pattern = kwargs.get('pattern')
        pattern = raw_pattern if isinstance(raw_pattern, str) else ''
        if not pattern.strip():
            return ToolResult.err(
                'pattern must be a non-empty glob or substring.',
                code='empty-pattern',
            )

        raw_path: str | None = kwargs.get('path')
        root = Path.cwd()

        # Resolve optional path argument under project root
        search_dir = root
        relative_path: str | None = None
        if raw_path is not None:
            try:
                search_dir = resolve_in_root(root, raw_path)
            except ValueError:
                return ToolResult.err(
                    f'{raw_path} escapes the project root.',
                    code='path-escapes-root',
                )

            if not search_dir.is_dir():
                return ToolResult.err(
                    f'{raw_path} is not an existing directory.',
                    code='not-a-directory',
                )

            relative_path = str(search_dir.relative_to(root))

        rg, rg_error = locate_rg()
        if rg is None:
            return rg_error  # type: ignore[return-value]

        scope = relative_path if relative_path is not None else '.'

        # A pattern with no glob metacharacters is a literal the model likely
        # meant as a substring — wrap it so *pattern* matches anywhere in a name.
        glob = pattern if any(c in _GLOB_METACHARS for c in pattern) else f'*{pattern}*'

        # rg matches the include glob case-insensitively (--iglob), but a
        # command-line inclusion glob *overrides* .gitignore in ripgrep — a
        # broad pattern would surface gitignored files (e.g. __pycache__/*.pyc).
        # To stay gitignore-aware we intersect the glob matches with the plain
        # `rg --files` universe (no inclusion glob, so .gitignore is fully
        # honoured). rg does both the glob semantics and the ignore logic; we
        # only take the overlap.
        matched, err = self._rg_files(rg, root, scope, ['--iglob', glob])
        if err is not None:
            return err

        if not matched:  # glob matched nothing — no need to enumerate the tree
            return ToolResult.ok(
                f'No files match {pattern!r}.',
                match_count=0,
            )

        allowed, err = self._rg_files(rg, root, scope, [])
        if err is not None:
            return err

        paths = sorted(matched & allowed)
        total = len(paths)

        if total == 0:  # every glob match was gitignored — report as no match
            return ToolResult.ok(
                f'No files match {pattern!r}.',
                match_count=0,
            )

        truncated = total > _MAX_RESULTS
        body = '\n'.join(paths[:_MAX_RESULTS])
        if truncated:
            body += (
                f'\n\n… {total - _MAX_RESULTS} more match(es) not shown; '
                f'narrow the pattern to see the rest.'
            )

        return ToolResult.ok(
            body,
            match_count=total,
            truncated=truncated,
        )

    @staticmethod
    def _rg_files(
        rg: str, root: Path, scope: str, extra: list[str]
    ) -> tuple[set[str], ToolResult | None]:
        """Run ``rg --files`` under *scope* and return its paths as a set.

        Always excludes the harness state dir (``IGNORE_GLOB``); *extra* carries
        any additional flags such as the ``--iglob`` include pattern.  Exactly
        one path arg is appended so rg searches the working directory, not stdin.

        Args:
            rg: Absolute path to the ripgrep binary.
            root: Project root (the subprocess cwd).
            scope: The single path argument — a relative subdir or ``'.'``.
            extra: Additional rg flags inserted before the ignore glob.

        Returns:
            ``(paths, None)`` on success (an empty set means rg matched nothing),
            or ``(set(), error_result)`` when rg exits with a real error code.
        """
        result = subprocess.run(
            [rg, '--files', *extra, '--glob', IGNORE_GLOB, scope],
            capture_output=True,
            text=True,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
        )

        # Exit 0: files listed. Exit 1: none matched (not an error). Above 1: rg
        # failed for a real reason (bad glob, unreadable path, ...).
        if result.returncode not in (0, 1):
            return set(), ToolResult.err(
                result.stderr.strip() or f'rg exited with code {result.returncode}.',
                code='search-failed',
            )

        return set(strip_dot_prefix(result.stdout.splitlines())), None
