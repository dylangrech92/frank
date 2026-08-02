"""Write file tool: writes a new or existing file with the given content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools._edit import finalize_write, freshness_gate
from tools._sandbox import emit_mutation, resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


class WriteFile(Tool):
    """Writes content to a new or existing file inside the project.

    Creates missing parent directories as needed. When the target path names an
    existing regular file the freshness gate is enforced — a never-read or
    stale file refuses the overwrite. Directories are rejected with a
    ``not-a-file`` error.

    Decides between create-vs-overwrite based on whether the file already
    exists; call this whenever you want content at a path.
    """

    name = 'write_file'
    alternative = 'edit_file'
    description = (
        'Writes content to a new or existing file inside the project. '
        'Creates missing parent directories. Refuses to overwrite a file that '
        'has not been freshly read; use edit_file for targeted in-place edits. '
        'Pass an empty string as ``contents`` to truncate a file to zero bytes.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root where the file should be written.',
            },
            'contents': {
                'type': 'string',
                'description': 'Content to write to the file. Empty string truncates the file to 0 bytes.',
            },
        },
        'required': ['path', 'contents'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, writing *contents* to *path*.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path``
                (required, relative to project root) and ``contents``
                (required; empty string is valid and truncates to 0 bytes).

        Returns:
            A ``ToolResult`` confirming creation or overwrite on success, or
            an error when the path escapes the project root, names a
            directory, or the destination is a never-read/stale file.
        """
        path_arg = kwargs.get('path')
        raw_path = path_arg if isinstance(path_arg, str) else ''
        contents = kwargs.get('contents', '')
        # Resolve under project root -- this is the one spot that produces
        # the path-escapes-root error with the canonical message from the
        # sandbox resolver.
        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        # Directories are not writable via this tool; refuse before attempting
        # any freshness check or write attempt.
        if resolved.exists() and not resolved.is_file():
            return ToolResult.err(
                f'{raw_path} is a directory, not a file.',
                code='not-a-file',
            )

        if resolved.exists():
            # File already exists: enforce the freshness gate exactly as our
            # own overwrite guard does, then overwrite and re-stamp via finalize_write.
            stale_error = freshness_gate(resolved, raw_path)
            if stale_error is not None:
                return stale_error

            # Guard against a destructive partial-edit (TKT-1468): the model
            # sometimes calls write_file (a full overwrite) with only a code
            # snippet when it means to make a small edit, silently destroying a
            # large working file. Refuse when the new content is a tiny fraction
            # of a substantial existing file and steer to the targeted-edit
            # tools. The threshold is deliberately conservative (<5% of a >=4 KB
            # file) so legitimate rewrites are unaffected; an intentional full
            # shortening can still use write_file with an empty contents.
            try:
                existing_size = resolved.stat().st_size
            except OSError:
                existing_size = 0
            new_size = len(contents.encode('utf-8'))
            if existing_size >= 4000 and new_size * 20 < existing_size:
                return ToolResult.err(
                    f'{raw_path}: refusing to overwrite a {existing_size}-byte file '
                    f'with {new_size} bytes — this looks like a partial edit (a '
                    f'snippet), not a full rewrite, and would destroy the existing '
                    f'content. Use edit_file for a targeted edit, or delete_file '
                    f'first if you genuinely mean to replace the whole file.',
                    code='destructive-partial-overwrite',
                    hint='Use edit_file for small edits.',
                )

            resolved.write_text(contents, encoding='utf-8')
            finalize_write(resolved)
            return ToolResult.ok(
                f'File updated: {raw_path}',
                path=raw_path,
                bytes_written=len(contents.encode('utf-8')),
                overwritten=True,
            )

        # Path does not exist yet: create it (making parent directories as
        # needed) and emit a 'created' mutation event.  We do not call
        # finalize_write here -- its body emits 'changed', but newly created
        # files dispatch on mutation-kind ('created' vs 'changed') downstream.
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(contents, encoding='utf-8')
        emit_mutation('created', resolved)
        return ToolResult.ok(
            f'File created: {raw_path}',
            path=raw_path,
            bytes_written=len(contents.encode('utf-8')),
            created=True,
        )
