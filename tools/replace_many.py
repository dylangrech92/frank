"""Replace many tool: replaces every occurrence of a literal string across the project."""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from tools._edit import finalize_write, looks_line_numbered
from tools._read_registry import check_fresh
from tools.base import Tool
from tools.result import ToolResult

# Directories always skipped during the tree walk.
_SKIP_DIRS = ('.git', '.coding_agent', '__pycache__')


class ReplaceMany(Tool):
    """Replaces every occurrence of a literal string across the project.

    Walks the project tree (respecting built-in skip directories), reads each
    matching text file, counts occurrences of *search*, replaces them with the
    given text, and reports per-file replacement counts.  Binary or unreadable
    files are silently skipped. Symlinked files whose resolved paths escape outside
    the project root are also silently skipped to prevent sandbox escapes.
    """

    name = 'replace_many'
    summary = 'Replace every occurrence of a literal string across the project.'
    description = (
        'Replaces every occurrence of a literal string across the project and reports '
        'per-file replacement counts. When a ``glob`` pattern is provided, only file names '
        'matching that glob pattern are touched; otherwise all text files in the tree are '
        "considered. The search string must be the raw file text — do not include read_file's "
        "display-only line-number prefixes (the '     1\\t' column)."
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'search': {
                'type': 'string',
                'description': 'The exact literal text to find in every file.',
            },
            'replace': {
                'type': 'string',
                'description': 'The replacement text.',
            },
            'glob': {
                'type': 'string',
                'description': (
                    "A filename pattern such as '*.py' to limit which files are touched. "
                    'Matched against each file name with fnmatch; when omitted all text '
                    'files are candidates.'
                ),
            },
        },
        'required': ['search', 'replace'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, replacing every occurrence of *search* across files.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``search``
                (required, the exact literal text to find), ``replace`` (required,
                the replacement text), and an optional ``glob`` pattern.

        Returns:
            A ``ToolResult`` with a body listing per-file replacement counts on success,
            or the string "no occurrences were found" when none matched.
        """
        search = kwargs.get('search', '') if isinstance(kwargs.get('search'), str) else ''
        replace = kwargs.get('replace', '') if isinstance(kwargs.get('replace'), str) else ''
        glob_pattern: str | None = kwargs.get('glob') if isinstance(kwargs.get('glob'), str) else None

        # Validate that search is non-empty.
        if not search:
            return ToolResult.err(
                'The ``search`` parameter is required and must be a non-empty string.',
                code='bad-arguments',
            )

        root = Path.cwd()
        real_root = root.resolve()

        # Accumulate per-file results as (relative_path, count) tuples.
        changed_files: list[tuple[str, int]] = []
        # Files skipped because this session's view of them is stale (another
        # process modified them after this session last read them).
        skipped_stale: list[str] = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skip directories in-place so os.walk descends no further.
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

            for filename in filenames:
                # Apply glob filter when provided.
                if glob_pattern is not None and not fnmatch(filename, glob_pattern):
                    continue

                real_file = Path(os.path.join(dirpath, filename)).resolve()
                if real_file != real_root and not real_file.is_relative_to(real_root):
                    continue

                rel_path = os.path.relpath(os.path.join(dirpath, filename), root)

                try:
                    content = real_file.read_text(encoding='utf-8')
                except (UnicodeDecodeError, OSError):
                    # Treat as binary or unreadable.
                    continue

                count = content.count(search)
                if count == 0:
                    continue

                # Skip files this session read earlier but that changed on
                # disk since -- another process may have modified them.
                # ('unread' is not gated here: replace_many by design sweeps
                # files the caller never individually read, so that would
                # defeat its purpose.)
                if check_fresh(real_file) == 'stale':
                    skipped_stale.append(rel_path)
                    continue

                new_content = content.replace(search, replace, count)
                real_file.write_text(new_content, encoding='utf-8')

                # Re-stamp the read registry and emit exactly one mutation
                # event per changed file.
                finalize_write(real_file)

                changed_files.append((rel_path, count))

        if not changed_files:
            if skipped_stale:
                return ToolResult.err(
                    'No files were changed; all matching files changed on disk after you last read '
                    'them: ' + ', '.join(skipped_stale),
                    code='file-changed-on-disk',
                    hint='Re-read the affected files with read_file, then re-apply the replacement.',
                )
            body = 'No occurrences were found.'
            if looks_line_numbered(search):
                # The model likely pasted read_file's numbered output into the
                # search string; point it at the real cause.
                body += (
                    " Your search text includes read_file's line-number prefixes "
                    "(the '     1\\t' column) — those are display-only. Strip them so "
                    "the search matches the real file content."
                )
            return ToolResult.ok(
                body,
                files_changed=0,
                total_replacements=0,
            )

        total = sum(c for _, c in changed_files)
        body_lines = [f'{path}: {cnt}' for path, cnt in changed_files]
        if skipped_stale:
            body_lines.append(
                'Skipped (changed on disk since last read): ' + ', '.join(skipped_stale)
            )

        return ToolResult.ok(
            '\n'.join(body_lines),
            files_changed=len(changed_files),
            total_replacements=total,
            files_skipped_stale=len(skipped_stale),
        )
