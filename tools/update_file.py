"""Update file tool: overwrites an existing file with entirely new content inside the project."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._edit import finalize_write, freshness_gate, resolve_existing_file


class UpdateFile(Tool):
    """Overwrites an existing file with entirely new content (full overwrite, not a patch).

    The path must be relative to the project root.  Only regular files can be
    updated; directories and other special paths are rejected.
    """

    name = 'update_file'
    summary = 'Overwrite an existing file with new content (full overwrite).'
    description = (
        'Overwrites an existing file with entirely new content (full overwrite, not a patch) — '
        'use it for a full rewrite of a small file. For a targeted change to part of a file prefer '
        'replace_one or edit_lines; to make a brand-new file use create_file. '
        'The path must be relative to the project root.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root of the file to overwrite.',
            },
            'content': {
                'type': 'string',
                'description': 'New content to write to the file, replacing all existing content.',
            },
        },
        'required': ['path', 'content'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, overwriting an existing file with entirely new content.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path``
                (required, relative to project root) and ``content`` (required).

        Returns:
            A ``ToolResult`` naming the updated path on success, or an error
            when the path escapes root, the file does not exist, or the target
            is not a regular file.
        """
        path_arg = kwargs.get('path')
        raw_path = path_arg if isinstance(path_arg, str) else ''
        content = kwargs.get('content', '')

        # Resolve under root and confirm the target is an existing regular file.
        resolved, error = resolve_existing_file(raw_path)
        if error is not None:
            return error
        assert resolved is not None

        # Refuse to overwrite a stale or never-read view of the file --
        # another process may have changed it since this session last saw it.
        stale_error = freshness_gate(resolved, raw_path)
        if stale_error is not None:
            return stale_error

        # Guard against a destructive partial-edit (TKT-1468): the model
        # sometimes calls update_file (a full overwrite) with only a code
        # SNIPPET when it means to make a small edit, silently destroying a
        # large working file (observed: an 11.5 KB game.js overwritten with a
        # 226-byte fragment to "fix" a lint hint). Refuse when the new content
        # is a tiny fraction of a substantial existing file and steer to the
        # targeted-edit tools. The threshold is deliberately conservative
        # (<5% of a >=4 KB file) so legitimate rewrites are unaffected; an
        # intentional full shortening can still use delete_file + create_file.
        try:
            existing_size = resolved.stat().st_size
        except OSError:
            existing_size = 0
        new_size = len(content.encode('utf-8'))
        if existing_size >= 4000 and new_size * 20 < existing_size:
            return ToolResult.err(
                f'{raw_path}: refusing to overwrite a {existing_size}-byte file '
                f'with {new_size} bytes — this looks like a partial edit (a '
                f'snippet), not a full rewrite, and would destroy the existing '
                f'content. Use replace_one / replace_many for a targeted edit, '
                f'or delete_file + create_file if you genuinely mean to replace '
                f'the whole file.',
                code='destructive-partial-overwrite',
                hint='Use replace_one / replace_many for small edits.',
            )

        # Write content as UTF-8
        resolved.write_text(content, encoding='utf-8')

        # Re-stamp the read registry and emit exactly one mutation event.
        finalize_write(resolved)

        return ToolResult.ok(
            f'File updated: {raw_path}',
            path=raw_path,
            bytes_written=len(content.encode('utf-8')),
        )
