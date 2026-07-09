"""Read file tool: reads contents of a file inside the project with optional line paging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from tools._read_registry import record_read
from tools._sandbox import resolve_in_root


def _number_lines(lines: list[str], first_lineno: int) -> str:
    """Render *lines* cat -n style: ``     N\\t<line>`` starting at *first_lineno*.

    The line number is right-aligned in a width-6 column so numbers line up, and
    reflects the true file line number (not a page-relative index) so a paged
    read still shows the real line numbers.
    """
    return '\n'.join(
        f'{first_lineno + offset:6d}\t{line}' for offset, line in enumerate(lines)
    )


class ReadFile(Tool):
    """Reads a file inside the project, returning its contents with optional line range paging.

    The path must be relative to the project root.  Optional *start_line* and
    *end_line* parameters let the caller page through large files one chunk at a
    time (both are 1-based inclusive).
    """

    name = 'read_file'
    summary = 'Read a file\u2019s contents with optional line-range paging.'
    description = (
        'Reads a file inside the project returning its contents with optional line range. '
        'Output is cat -n style: every line is prefixed with its true 1-based file line number '
        'and a tab. The line numbers are display-only \u2014 never include them in file content you '
        'write, nor in search strings for replace_one/replace_many. '
        'The path must be relative to the project root.'
    )
    action = 'read the file'
    oversize_hint = 'use start_line/end_line for a smaller range, or find_symbol to jump to the symbol first'
    alternative = 'list_files to check the path exists'
    parallel_safe = True  # pure filesystem read
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root.',
            },
            'start_line': {
                'type': 'integer',
                'description': '1-based first line to include (inclusive).',
            },
            'end_line': {
                'type': 'integer',
                'description': '1-based last line to include (inclusive).',
            },
        },
        'required': ['path'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, reading a file and optionally slicing by line range.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Expects ``path``
                (required), and optional ``start_line`` and ``end_line`` (both
                1-based inclusive).

        Returns:
            A ``ToolResult`` with the file content as its body on success, or an
            error when the path escapes root, is not a regular file, exceeds the
            character limit without paging, or has invalid range parameters.
        """
        path_arg = kwargs.get('path')
        raw_path = path_arg if isinstance(path_arg, str) else ''
        start_line: int | None = kwargs.get('start_line')
        end_line: int | None = kwargs.get('end_line')

        # Resolve to absolute path under project root
        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.err(
                f'{resolved} does not exist or is not a regular file.',
                code='not-a-file',
            )

        content = resolved.read_text(encoding='utf-8', errors='replace')

        # Enforce size limit when no paging parameters are provided. The cap is
        # measured against the RAW content, not the line-numbered output built
        # for the body below, so numbering never shrinks the effective budget.
        if start_line is None and end_line is None:
            MAX_CHARACTERS = 50000
            if len(content) > MAX_CHARACTERS:
                lines_count = len(content.splitlines())
                return ToolResult.err(
                    f'{raw_path} is too large ({len(content)} characters, {lines_count} '
                    'lines). Use find_symbol to jump to the relevant symbol, then '
                    'read_file with start_line and end_line for just that range.',
                    code='file-too-large',
                    hint='Use find_symbol to locate the symbol, then read_file with start_line/end_line for its range.',
                )

            # Content fits without paging — return the full file, cat -n style.
            file_lines = content.splitlines()
            lines_count = len(file_lines)
            record_read(resolved)
            return ToolResult.ok(_number_lines(file_lines, 1), total_lines=lines_count)

        if start_line is not None and end_line is not None:
            # Both bounds given — validate below
            if (
                not isinstance(start_line, int)
                or not isinstance(end_line, int)
                or isinstance(start_line, bool)
                or isinstance(end_line, bool)
                or start_line < 1
                or end_line < 1
                or start_line > end_line
            ):
                return ToolResult.err(
                    f'Invalid line range: start_line={start_line}, end_line={end_line}. '
                    'Both must be positive integers with start_line <= end_line.',
                    code='bad-range',
                )

        elif start_line is not None and end_line is None:
            # Only start_line — default end_line to total line count
            lines_count = len(content.splitlines())
            if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
                return ToolResult.err(
                    f'Invalid line range: start_line={start_line}. '
                    'start_line must be a positive integer.',
                    code='bad-range',
                )
            end_line = lines_count

        elif start_line is None and end_line is not None:
            # Only end_line — default start_line to 1
            max_lines = len(content.splitlines())
            if not isinstance(end_line, int) or isinstance(end_line, bool) or end_line < 1:
                return ToolResult.err(
                    f'Invalid line range: end_line={end_line}. '
                    'end_line must be a positive integer.',
                    code='bad-range',
                )
            if end_line > max_lines:
                return ToolResult.err(
                    f'Invalid line range: end_line={end_line}, file has {max_lines} lines. '
                    'end_line must not exceed the file length.',
                    code='bad-range',
                )
            start_line = 1

        # At this point we have both bounds (possibly defaulted above)
        assert start_line is not None and end_line is not None

        # Range read — slice by lines and re-check size guard on the RAW slice
        # (numbering is display-only, so it must not count against the cap).
        lines = content.splitlines()
        total_lines = len(lines)
        sliced = lines[(start_line - 1): end_line]  # type: ignore[index]
        returned = '\n'.join(sliced)
        MAX_CHARACTERS = 50000
        if len(returned) > MAX_CHARACTERS:
                return ToolResult.err(
                    f'{raw_path} is too large ({len(returned)} characters for the requested line range). '
                    'Use find_symbol to jump to the relevant symbol and request a narrower range.',
                    code='file-too-large',
                    hint='Use find_symbol to locate the symbol, then read_file with a narrower start_line/end_line range.',
                )
        record_read(resolved)
        # Number from the true first line of the slice, not page-relative.
        return ToolResult.ok(
            _number_lines(sliced, start_line), total_lines=total_lines, returned_lines=len(sliced)
        )
