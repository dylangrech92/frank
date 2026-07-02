"""Replace many tool: replaces every occurrence of a literal string across the project."""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from tools._sandbox import emit_mutation
from tools.base import Tool
from tools.result import ToolResult, truncate

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
    description = (
        'Replaces every occurrence of a literal string across the project and reports '
        'per-file replacement counts. When a ``glob`` pattern is provided, only file names '
        'matching that glob pattern are touched; otherwise all text files in the tree are '
        'considered.'
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
            or the string "no occurrences were found" when none matched.  The body is
            truncated at 20 000 characters when exceeded.
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

                new_content = content.replace(search, replace, count)
                real_file.write_text(new_content, encoding='utf-8')

                # Emit exactly one mutation event per changed file.
                emit_mutation('changed', real_file)

                changed_files.append((rel_path, count))

        if not changed_files:
            return ToolResult.ok(
                'No occurrences were found.',
                files_changed=0,
                total_replacements=0,
            )

        total = sum(c for _, c in changed_files)
        body_lines = [f'{path}: {cnt}' for path, cnt in changed_files]
        body_str, truncated = truncate('\n'.join(body_lines), 20_000)
        if truncated:
            body_str += '\noutput truncated'

        return ToolResult.ok(
            body_str,
            files_changed=len(changed_files),
            total_replacements=total,
        )
