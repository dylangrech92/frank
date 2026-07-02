"""List files tool: walks the project tree and renders an indented overview."""

import fnmatch
import os
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult, truncate
from tools._sandbox import resolve_in_root


_LISTED_PATHS = ('.git', '.coding_agent', '__pycache__')


def _load_gitignore_patterns(root: Path) -> list[str]:
    """Load glob patterns from the ``.gitignore`` at *root*, if present.

    Skips blank lines and lines starting with ``#``, keeps trailing-slash
    directory-only patterns as-is, and returns everything else verbatim so
    :func:`fnmatch` can match against file/basename later.

    Args:
        root: Project root (where ``.gitignore`` is expected).

    Returns:
        A list of pattern strings ready for use with :func:`_should_skip`.
    """
    gitignore = root / '.gitignore'
    if not gitignore.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        patterns.append(stripped)
    return patterns


def _should_skip(name: str, is_dir: bool, patterns: list[str]) -> bool:
    """Return ``True`` when *name* should be skipped during tree walking.

    Args:
        name: The basename of the file or directory entry.
        is_dir: Whether *name* refers to a directory.
        patterns: Parsed :file:`.gitignore` patterns (may also be empty for
            built-in exclusions).

    Returns:
        ``True`` if the entry should be excluded, ``False`` otherwise.
    """
    # Built-in exclusions — always applied regardless of .gitignore
    if name in _LISTED_PATHS:
        return True

    if not patterns:
        return False

    for pat in patterns:
        # Trailing slash: directory-only pattern
        if pat.endswith('/'):
            if is_dir and fnmatch.fnmatch(name, pat.rstrip('/')):
                return True
            continue

        # Directories and files: match against full name
        if fnmatch.fnmatch(name, pat):
            return True

    return False


class ListFiles(Tool):
    """Lists the project file tree while respecting gitignore rules.

    Walks the target directory recursively, rendering each entry indented by
    its depth, grouped with directories first then files within each level.
    Truncates output at 20000 characters when exceeded.
    """

    name = 'list_files'
    description = (
        'Lists the project file tree while respecting gitignore rules.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'The subdirectory to list relative to the project root.',
            },
        },
        'required': [],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, rendering an indented file-tree string.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Expects an
                optional ``path`` key (default ``'.'``).

        Returns:
            A ``ToolResult`` with the tree as its body on success, or an
            error when the path escapes the root or does not exist.
        """
        # Default to current working directory
        raw_path: str = kwargs.get('path', '.') if isinstance(kwargs.get('path'), str) else '.'
        target_root = Path.cwd()

        try:
            target = resolve_in_root(target_root, raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        if not target.exists() or not target.is_dir():
            return ToolResult.err(
                f'{target} does not exist or is not a directory.',
                code='not-a-directory',
            )

        patterns = _load_gitignore_patterns(target_root)
        lines: list[str] = []
        entries = 0

        def _recurse(dir_path: Path, depth: int) -> None:
            nonlocal entries
            indent = '  ' * depth
            for entry in sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
                name = entry.name
                if _should_skip(name, entry.is_dir(), patterns):
                    continue
                lines.append(f'{indent}{name}/' if entry.is_dir() else f'{indent}{name}')
                entries += 1
                if entry.is_dir():
                    _recurse(entry, depth + 1)

        _recurse(target, 0)

        body_list, truncated = truncate('\n'.join(lines), 20_000)
        if truncated:
            body_list += '\noutput truncated'

        return ToolResult.ok(body_list, entries=entries)
