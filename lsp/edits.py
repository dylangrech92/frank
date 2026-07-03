"""Shared LSP WorkspaceEdit applier – validate, compute, and atomically apply text edits.

Exposes a single public API: ``apply_workspace_edit(workspace_edit, root_path)`` which
validates every URI is inside *root_path*, computes the new file contents in memory,
then writes all changes atomically with full rollback on failure.
"""

from __future__ import annotations

import os

from lsp.manager import uri_to_path


class WorkspaceEditError(Exception):
    """Raised when a workspace edit cannot be applied."""


def collect_text_edits(workspace_edit: dict, root_path: str) -> dict[str, list[dict]]:
    """Merge LSP ``changes`` and ``documentChanges`` into a single path mapping.

    Every URI is converted to an absolute file path via :func:`uri_to_path`.
    Each resulting path is validated to be inside *root_path* and to exist as a regular file.

    Args:
        workspace_edit: An LSP ``WorkspaceEdit`` dict (may have ``"changes"`` and/or ``"documentChanges"``).
        root_path: The workspace root directory for containment checks.

    Returns:
        A mapping of absolute file path to a list of TextEdit dicts.

    Raises:
        WorkspaceEditError: When an unsupported resource operation is encountered, a path leaves the workspace root, or a path does not point to an existing regular file.
    """
    result: dict[str, list[dict]] = {}

    # --- 'changes' shape: uri -> [TextEdit, ...] --------------------------------
    for uri, edits in (workspace_edit.get('changes') or {}).items():
        path = uri_to_path(uri)
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(root_path)

        if real_path != real_root and not real_path.startswith(real_root + os.sep):
            raise WorkspaceEditError(f'edit target escapes workspace root: {path}')

        if not os.path.isfile(path):
            raise WorkspaceEditError(f'edit target is not an existing file: {path}')

        result.setdefault(path, []).extend(edits or [])

    # --- 'documentChanges' shape: list[TextDocumentEdit | CreateFile | RenameFile | DeleteFile] ---
    for entry in (workspace_edit.get('documentChanges') or []):
        if 'textDocument' not in entry:
            kind = entry.get('kind', 'unknown')
            raise WorkspaceEditError(f'unsupported resource operation in workspace edit: {kind}')

        uri = entry['textDocument']['uri']
        path = uri_to_path(uri)
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(root_path)

        if real_path != real_root and not real_path.startswith(real_root + os.sep):
            raise WorkspaceEditError(f'edit target escapes workspace root: {path}')

        if not os.path.isfile(path):
            raise WorkspaceEditError(f'edit target is not an existing file: {path}')

        result.setdefault(path, []).extend(entry.get('edits') or [])

    return result


def apply_text_edits(text: str, edits: list[dict]) -> str:
    """Apply a list of TextEdit dicts to *text*, returning the modified string.

    Builds a line-start offset table from ``splitlines(keepends=True)`` and converts
    each edit's (line, character) range into absolute byte offsets. Edits are clamped
    to document bounds before overlap detection.

    Args:
        text: The original document text.
        edits: A list of TextEdit dicts with ``"range"`` ({'start': {...}, 'end': ...}) and ``"newText"`` keys.

    Returns:
        The modified text with all edits applied.

    Raises:
        WorkspaceEditError: When any two edits overlap after offset conversion.
    """
    lines = text.splitlines(True)
    offsets: list[int] = []
    current = 0
    for line in lines:
        offsets.append(current)
        current += len(line)
    offsets.append(current)  # sentinel – one past the last character

    def _clamp(value: int, max_val: int) -> int:
        """Clamp *value* to [0, max_val]."""
        if value < 0:
            return 0
        return min(value, max_val)

    converted: list[tuple[int, int, str]] = []
    for edit in edits:
        try:
            start_line = edit['range']['start']['line']
            start_char = edit['range']['start']['character']
            end_line = edit['range']['end']['line']
            end_char = edit['range']['end']['character']
        except (KeyError, TypeError):
            continue

        new_text = edit.get('newText', '')

        line_count = len(lines)
        clamped_start_line = _clamp(start_line, line_count)
        clamped_end_line = _clamp(end_line, line_count)

        max_col = line_count  # safety net for empty / partial files

        start_off = offsets[_clamp(clamped_start_line, max_col)] + _clamp(start_char, len(lines[clamped_start_line]) if clamped_start_line < line_count else 0) if clamped_start_line < line_count else offsets[-1]
        end_off = offsets[_clamp(clamped_end_line, max_col)] + _clamp(end_char, len(lines[clamped_end_line]) if clamped_end_line < line_count else 0) if clamped_end_line < line_count else offsets[-1]

        # Clamp absolute offsets to document length
        doc_len = offsets[-1]
        start_off = min(start_off, doc_len)
        end_off = min(end_off, doc_len)

        converted.append((start_off, end_off, new_text))

    # Sort descending by (start_offset, end_offset) so later edits don't invalidate earlier positions.
    converted.sort(key=lambda e: (e[0], e[1]), reverse=True)

    prev_start = len(text)  # upper bound from the "first" (largest start) edit
    for start_off, end_off, new_text in converted:
        if end_off > prev_start:
            raise WorkspaceEditError('overlapping text edits')
        prev_start = start_off
        text = text[:start_off] + new_text + text[end_off:]

    return text


def apply_workspace_edit(workspace_edit: dict, root_path: str) -> list[str]:
    """Validate then atomically apply an LSP ``WorkspaceEdit`` to disk files.

    Phase 1 validates every path and computes the new file contents entirely in memory.
    Phase 2 writes each updated file, emitting a mutation event after every successful write.
    If any write fails, all previously-written files are rolled back and an error is raised.

    Args:
        workspace_edit: An LSP ``WorkspaceEdit`` dict (may have ``"changes"`` and/or ``"documentChanges"``).
        root_path: The workspace root directory for containment checks.

    Returns:
        A sorted list of file paths written, each relative to *root_path*.

    Raises:
        WorkspaceEditError: On validation failure, or when a write fails (with rollback).
    """
    path_edits = collect_text_edits(workspace_edit, root_path)

    # Phase 1 – validate & compute in memory
    new_texts: dict[str, str] = {}
    try:
        for path, edits in path_edits.items():
            with open(path, 'r', encoding='utf-8') as f:
                original_text = f.read()
            new_texts[path] = apply_text_edits(original_text, edits)
    except WorkspaceEditError:
        raise
    except Exception as exc:
        target_path = ''
        for path in path_edits:
            if path not in new_texts:
                target_path = path
                break
        raise WorkspaceEditError(f'validate error on {target_path}: {exc}') from exc

    # Phase 2 – write with rollback-on-failure
    from tools._sandbox import emit_mutation  # pylint: disable=import-outside-toplevel

    originals: dict[str, str] = {}
    for path in new_texts:
        with open(path, 'r', encoding='utf-8') as f:
            originals[path] = f.read()

    written_paths: list[str] = []
    try:
        for path, new_text in new_texts.items():
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_text)
            written_paths.append(path)
            emit_mutation('changed', path)
    except Exception as exc:
        # Restore the saved original content of every file already written.
        for wpath in written_paths:
            try:
                with open(wpath, 'w', encoding='utf-8') as f:
                    f.write(originals[wpath])
            except Exception:  # noqa: E722
                pass

        raise WorkspaceEditError(f'failed to write {exc}; all changes rolled back') from exc

    return sorted(os.path.relpath(path, root_path) for path in written_paths)
