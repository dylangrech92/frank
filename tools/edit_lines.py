"""Edit lines tool: replaces a 1-based inclusive line range of a file with new text."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._edit import finalize_write, freshness_gate, resolve_existing_file
from tools._sandbox import emit_mutation

# Lines of context shown on either side of the changed region in the preview.
_PREVIEW_CONTEXT = 3


class EditLines(Tool):
    """Replaces a 1-based inclusive line range of a file with new text.

    A line-anchored alternative to a full-file overwrite: the caller names the
    line range to replace instead of restating the whole file.  Insertion is
    expressed as an empty range (``end_line = start_line - 1``); deletion is
    expressed by passing an empty ``new_text`` for the range to remove.  The
    path must be relative to the project root and the file must have been read
    this session; only regular files can be edited.
    """

    name = 'edit_lines'
    summary = 'Replace a 1-based inclusive line range of a file with new text.'
    description = (
        'Replaces lines start_line..end_line (1-based, inclusive) of a file with new_text. '
        'To INSERT text before an existing line N without deleting anything, set start_line=N '
        'and end_line=N-1 (an empty range replaces zero lines); to insert at the very top set '
        'start_line=1 and end_line=0; to append at the end set start_line to one past the last '
        'line and end_line to the last line. To DELETE lines start_line..end_line, pass an empty '
        'string as new_text. new_text may span multiple lines and need not carry '
        "a trailing newline. Do not include read_file's display-only line-number prefixes in "
        'new_text. The path must be relative to the project root; read the file first. After '
        'an edit that changes the line count, line numbers below it shift — re-read the file '
        'before editing it again (a further edit without a re-read is refused).'
    )
    alternative = 'replace_one or update_file'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root of the file to edit.',
            },
            'start_line': {
                'type': 'integer',
                'description': '1-based first line of the range to replace (inclusive).',
            },
            'end_line': {
                'type': 'integer',
                'description': (
                    '1-based last line of the range to replace (inclusive). Set to '
                    'start_line - 1 to insert without deleting any existing line.'
                ),
            },
            'new_text': {
                'type': 'string',
                'description': (
                    'Text to write in place of the replaced range. Pass an empty string '
                    'to delete the replaced range.'
                ),
            },
        },
        'required': ['path', 'start_line', 'end_line', 'new_text'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, replacing a line range of a file with new text.

        Args:
            **kwargs: Parsed from LLM function-call payload. Expects ``path``
                (required, relative to project root), ``start_line`` and
                ``end_line`` (required, 1-based inclusive; ``end_line =
                start_line - 1`` inserts without deleting), and ``new_text``
                (required, the replacement text).

        Returns:
            A ``ToolResult`` echoing a cat -n style numbered preview of the
            changed region on success, or an error when the path escapes root,
            the file does not exist / is not a file, the session's view is stale
            or never read, or the line range is out of bounds.
        """
        path_arg = kwargs.get('path')
        raw_path = path_arg if isinstance(path_arg, str) else ''
        start_line = kwargs.get('start_line')
        end_line = kwargs.get('end_line')
        raw_new_text = kwargs.get('new_text')
        new_text = raw_new_text if isinstance(raw_new_text, str) else ''

        # Resolve under root and confirm the target is an existing regular file.
        resolved, error = resolve_existing_file(raw_path)
        if error is not None:
            return error
        assert resolved is not None

        # Refuse to edit against a stale or never-read view of the file --
        # another process may have changed it since this session last saw it.
        stale_error = freshness_gate(resolved, raw_path)
        if stale_error is not None:
            return stale_error

        # Read the file as UTF-8. ``lines`` holds the logical lines the model
        # sees numbered (splitlines drops the line terminators); ``had_trailing``
        # records whether the file ends with a newline so we can preserve it.
        content = resolved.read_text(encoding='utf-8')
        lines = content.splitlines()
        total_lines = len(lines)
        had_trailing = content.endswith('\n')

        # Validate the range. ``start_line`` may run one past the last line (to
        # append) and ``end_line`` may sit at ``start_line - 1`` (to insert
        # without deleting).
        assert isinstance(start_line, int) and isinstance(end_line, int)
        if start_line < 1:
            return ToolResult.err(
                f'Invalid line range: start_line={start_line} must be >= 1.',
                code='bad-range',
            )
        if start_line > total_lines + 1:
            return ToolResult.err(
                f'Invalid line range: start_line={start_line}, file has {total_lines} lines. '
                'start_line must not exceed total_lines + 1 (one past the last line, to append).',
                code='bad-range',
            )
        if end_line < start_line - 1:
            return ToolResult.err(
                f'Invalid line range: end_line={end_line} must be >= start_line - 1 '
                f'({start_line - 1}). Use end_line = start_line - 1 to insert without deleting.',
                code='bad-range',
            )
        if end_line > total_lines:
            return ToolResult.err(
                f'Invalid line range: end_line={end_line}, file has {total_lines} lines. '
                'end_line must not exceed the file length.',
                code='bad-range',
            )

        # Build the new logical lines. splitlines() normalizes new_text so a
        # trailing newline in it never spawns a spurious blank line; an empty
        # new_text deletes the range.
        inserted = new_text.splitlines()
        before = lines[: start_line - 1]
        after = lines[end_line:]
        new_lines = before + inserted + after

        new_content = '\n'.join(new_lines)
        # Preserve the file's original trailing-newline behavior.
        if had_trailing and new_content and not new_content.endswith('\n'):
            new_content += '\n'

        resolved.write_text(new_content, encoding='utf-8')

        # A length-changing edit shifts the line numbers of everything below
        # it, so any further line-anchored edit against the old numbering would
        # land on the wrong lines. Skip the read-registry re-stamp in that case:
        # the next write-tool call hits the file-changed-on-disk gate and is
        # forced through a fresh read_file (fresh numbers). A same-length edit
        # leaves numbering intact and keeps the normal re-stamp.
        line_delta = len(inserted) - (end_line - start_line + 1)
        if line_delta == 0:
            finalize_write(resolved)
        else:
            emit_mutation('changed', resolved)

        preview = self._preview(new_content.splitlines(), start_line, len(inserted))
        if end_line >= start_line and not inserted:
            body = f'edit_lines: deleted lines {start_line}-{end_line} in {raw_path}.'
        elif end_line >= start_line:
            body = (
                f'edit_lines: replaced lines {start_line}-{end_line} in {raw_path} '
                f'({len(inserted)} line(s) written).'
            )
        else:
            body = (
                f'edit_lines: inserted at line {start_line} in {raw_path} '
                f'({len(inserted)} line(s) written).'
            )
        if line_delta != 0:
            body += (
                f' Line numbers below line {start_line} have shifted by '
                f'{line_delta:+d} — re-read the file with read_file before '
                'making another edit to it.'
            )
        if preview:
            body += '\n' + preview

        return ToolResult.ok(
            body,
            path=raw_path,
            lines_written=len(inserted),
            total_lines=len(new_lines),
        )

    @staticmethod
    def _preview(new_lines: list[str], region_start: int, inserted_count: int) -> str:
        """Render a cat -n style numbered preview of the changed region ± context.

        Args:
            new_lines: The logical lines of the file after the edit.
            region_start: 1-based first line of the changed region.
            inserted_count: Number of lines written (0 for a pure deletion).

        Returns:
            A newline-joined ``     N\\t<line>`` block spanning the changed
            region plus ``_PREVIEW_CONTEXT`` lines on either side, or an empty
            string when there is nothing to show.
        """
        region_end = region_start + inserted_count - 1  # < region_start on deletion
        ctx_start = max(1, region_start - _PREVIEW_CONTEXT)
        ctx_end = min(len(new_lines), region_end + _PREVIEW_CONTEXT)
        if ctx_end < ctx_start:
            return ''
        window = new_lines[ctx_start - 1: ctx_end]
        return '\n'.join(
            f'{ctx_start + offset:6d}\t{line}' for offset, line in enumerate(window)
        )
