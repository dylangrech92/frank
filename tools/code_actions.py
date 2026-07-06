"""Code-actions tool: list or apply IDE code actions via LSP textDocument/codeAction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lsp.manager import LSPUnavailableError, path_to_uri
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult


class CodeActions(Tool):
    """List or apply a code action for a file or line range.

    Two-step flow: call once to list available actions (quick fixes, organize
    imports, refactorings), then call again with ``apply`` set to the exact title
    of the desired action to execute it.
    """

    name = 'code_actions'
    summary = 'List/apply code actions (quick fixes, refactorings).'
    description = (
        'List code actions (quick fixes, organize imports, refactorings) '
        'available for a file or line range, or apply one by passing its '
        'exact title as ``apply``. Two-step flow: call once to list, '
        'call again with apply.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root.',
            },
            'start_line': {
                'type': 'integer',
                'description': '1-based start line (inclusive). Defaults to 1.',
            },
            'end_line': {
                'type': 'integer',
                'description': '1-based inclusive end line. Defaults to last line of the file.',
            },
            'only': {
                'type': 'string',
                'description': 'Filter by a single code-action kind (e.g. \'quickfix\', \'source.organizeImports\').',
            },
            'apply': {
                'type': 'string',
                'description': 'Exact title of a listed action to apply.',
            },
        },
        'required': ['path'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the code-actions tool for the given path and line range.

        Args:
            **kwargs: Parsed from LLM function-call payload. Requires ``path``
                (str). Optionally ``start_line``, ``end_line``, ``only``, and ``apply``.

        Returns:
            A ``ToolResult`` listing available actions or confirming an action was applied,
            or an error when validation fails or the language server is unavailable.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''

        # --- validate path ---------------------------------------------------
        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.err(
                f'{resolved} does not exist or is not a regular file.',
                code='not-a-file',
            )

        # --- read file line count for defaults -------------------------------
        with open(resolved, 'r', encoding='utf-8') as f:
            file_lines = len(f.readlines())
        num_file_lines = max(file_lines, 1)

        # --- validate start_line / end_line (default to 1 and last line) -----
        line_raw = kwargs.get('start_line')
        end_line_raw = kwargs.get('end_line')

        if line_raw is None:
            line_raw = 1
        if end_line_raw is None:
            end_line_raw = num_file_lines

        for label, value in [('start_line', line_raw), ('end_line', end_line_raw)]:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                return ToolResult.err(
                    f'{label} must be a positive integer, got {value!r}.',
                    code='bad-arguments',
                )

        line: int = line_raw  # type: ignore[assignment]
        end_line: int = end_line_raw  # type: ignore[assignment]

        if end_line > num_file_lines:
            end_line = num_file_lines

        if end_line < line:
            return ToolResult.err(
                f'end_line ({end_line}) must be >= start_line ({line}).',
                code='bad-arguments',
            )

        # --- validate only / apply -------------------------------------------
        only_raw = kwargs.get('only')
        apply_raw = kwargs.get('apply')

        if (
            only_raw is not None
            and (not isinstance(only_raw, str) or not only_raw.strip())
        ):
            return ToolResult.err(
                'only must be a non-empty string.',
                code='bad-arguments',
            )

        if (
            apply_raw is not None
            and (not isinstance(apply_raw, str) or not apply_raw.strip())
        ):
            return ToolResult.err(
                'apply must be a non-empty string.',
                code='bad-arguments',
            )

        # --- language server availability ------------------------------------
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.MANAGER is None:
            return ToolResult.err(
                'no language servers are running',
                code='lsp-unavailable',
            )

        MANAGER = main_module.MANAGER
        language = MANAGER.language_for_path(str(resolved))
        if language is None:
            return ToolResult.err(
                f'no language server configured for this file type '
                f'({Path(raw_path).suffix})',
                code='lsp-unavailable',
                hint=f'Try get_diagnostics for a supported file type.',
            )

        try:
            client = MANAGER.get_client(language)
        except LSPUnavailableError as exc:
            return ToolResult.err(
                str(exc),
                code='lsp-unavailable',
            )

        # --- ensure document is open/synced ----------------------------------
        uri = path_to_uri(str(resolved))
        if uri not in MANAGER._open_docs:
            MANAGER._did_open(str(resolved))

        # --- build request params --------------------------------------------
        range_params: dict[str, Any] = {
            'start': {'line': line - 1, 'character': 0},
            'end': {'line': end_line, 'character': 0},
        }

        # Pick up stored diagnostics that overlap the requested line window.
        from diagnostics import STORE  # pylint: disable=import-outside-toplevel

        included_diagnostics: list[dict] = []
        with STORE.condition:
            all_diags = list(STORE._diags.get(uri, []))

        for diag in all_diags:
            try:
                diag_range = diag.get('range', {})
                diag_start_line = diag_range.get('start', {}).get('line', -1)
                diag_end_line = diag_range.get('end', {}).get('line', -1)
            except (AttributeError, TypeError):
                continue

            if isinstance(diag_start_line, int) and isinstance(diag_end_line, int):
                if diag_start_line <= end_line - 1 and diag_end_line >= line - 1:
                    included_diagnostics.append(diag)

        context: dict[str, Any] = {'diagnostics': included_diagnostics}
        if only_raw is not None:
            context['only'] = [only_raw]

        try:
            result = client.request(
                'textDocument/codeAction',
                {'textDocument': {'uri': uri}, 'range': range_params, 'context': context},
                timeout=10.0,
            )
        except TimeoutError:
            return ToolResult.err(
                'language server did not answer in time',
                code='lsp-timeout',
            )
        except Exception as exc:
            return ToolResult.err(
                f'the {language} language server does not '
                f'support textDocument/codeAction (or the request failed: {exc})',
                code='lsp-capability',
            )

        # --- handle empty / None result --------------------------------------
        if not result:
            only_suffix = (f' (kind {only_raw})' if only_raw else '')
            return ToolResult.ok(f'no code actions available for {raw_path}{only_suffix}')

        # --- normalise entries -----------------------------------------------
        normalised: list[dict[str, Any]] = []
        for entry in result:
            if not isinstance(entry, dict):
                continue
            title = entry.get('title', 'untitled')
            kind = ''
            if 'kind' in entry:
                kind = entry['kind']
            elif 'command' in entry and 'edit' not in entry:
                kind = 'command'
            normalised.append({'title': title, 'kind': kind, 'entry': entry})

        # --- apply mode ------------------------------------------------------
        if apply_raw is not None:
            chosen: dict[str, Any] | None = None
            for item in normalised:
                if item['title'] == apply_raw:
                    chosen = item
                    break

            if chosen is None:
                return ToolResult.err(
                    f"no action titled {apply_raw!r}; call code_actions without apply to list titles",
                    code='bad-arguments',
                )

            entry = chosen['entry']
            title = chosen['title']

            # Resolve if it looks like a CodeAction (has no 'edit' and is not a bare Command).
            if 'edit' not in entry:
                entry_is_command = isinstance(entry.get('command'), str)

                if not entry_is_command:
                    try:
                        resolved_action = client.request(
                            'codeAction/resolve', entry, timeout=10.0,
                        )
                        if isinstance(resolved_action, dict):
                            entry = resolved_action
                    except Exception:
                        pass

            # Apply via 'edit'
            if 'edit' in entry:
                from lsp.edits import WorkspaceEditError, apply_workspace_edit  # pylint: disable=import-outside-toplevel

                try:
                    files = apply_workspace_edit(entry['edit'], str(Path.cwd()))
                except WorkspaceEditError as exc:
                    return ToolResult.err(str(exc), code='edit-failed')

                body = f"applied {title!r} to {len(files)} file(s):\n"
                for rel in files:
                    body += f'{rel}\n'
                return ToolResult.ok(body, count=len(files))

            # Fall through: no 'edit' after resolve.
            if 'command' in entry:
                return ToolResult.err(
                    f'action {title!r} is a server command, which this tool does not execute',
                    code='lsp-capability',
                )

            return ToolResult.err(
                f'action {title!r} returned no edit to apply',
                code='lsp-capability',
            )

        # --- list mode -------------------------------------------------------
        lines_out: list[str] = []
        for item in normalised:
            title = item['title']
            kind = item['kind']
            if kind:
                lines_out.append(f'- {title} [{kind}]')
            else:
                lines_out.append(f'- {title}')

        count = len(normalised)
        body = '\n'.join(lines_out) + f'\n-- {count} action(s)'
        return ToolResult.ok(body, count=count)
