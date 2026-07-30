"""Format-file tool: format a source file in place using the language server or a configured CLI formatter.

Formatter is dispatched either via the LSP ``textDocument/formatting`` request or, when no language
server client is available or capable, via a module-level CLI fallback table keyed by file suffix.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from lsp.manager import LSPUnavailableError, path_to_uri
from tools._sandbox import emit_mutation, resolve_in_root
from tools.base import Tool
from tools.result import ToolResult

# CLI fallback formatters keyed by language identifier. Only languages whose server cannot or does
# not expose a formatter need a CLI backing here.
_CLI_FORMATTERS: dict[str, list[str]] = {'python': ['black', '--quiet']}


class FormatFile(Tool):
    """Formats *path* in place using the language-server formatter or a CLI fallback.

    The ``path`` parameter must be relative to the project root and point to an
    existing file inside the sandbox.  If a language server client is available and
    capable, the LSP ``textDocument/formatting`` request is used.  Otherwise, when
    a language-specific CLI formatter exists in :data:`_CLI_FORMATTERS`, it is
    invoked directly as a fallback.
    """

    name = 'format'
    description = (
        'Format a source file in place using the language server\'s formatter, '
        'or a configured CLI formatter for languages whose server cannot format '
        '(e.g. black for Python).'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path relative to the project root.',
            },
        },
        'required': ['path'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the format tool for *path*.

        Args:
            **kwargs: Parsed from LLM function-call payload. Requires ``path`` (str).

        Returns:
            A ``ToolResult`` confirming formatting on success, or explaining
            why formatting could not proceed.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else ''

        # --- validate path -------------------------------------------------
        if not raw_path:
            return ToolResult.err(
                'path is required and must be a non-empty string.',
                code='bad-arguments',
            )

        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.err(
                f'{resolved} does not exist or is not a regular file.',
                code='not-a-file',
            )

        # --- language detection --------------------------------------------
        import main as main_module  # pylint: disable=import-outside-toplevel

        MANAGER = main_module.MANAGER
        language = None

        if MANAGER is not None:
            language = MANAGER.language_for_path(str(resolved))

        # --- CLI fallback path (no server or no LSP capability) -------------
        if language is not None:
            cli_lang = language
        else:
            cli_lang = {'.py': 'python'}.get(Path(raw_path).suffix)
        command = _CLI_FORMATTERS.get(cli_lang) if cli_lang else None

        if command is not None:
            return self._run_cli(command, raw_path, resolved)

        # --- LSP path when MANAGER exists ----------------------------------
        if MANAGER is None:
            return ToolResult.err(
                'no language servers are running',
                code='lsp-unavailable',
            )

        if language is None:
            return ToolResult.err(
                f'no language server configured for this file type '
                f'({Path(raw_path).suffix})',
                code='lsp-unavailable',
            )

        try:
            client = MANAGER.get_client(language)
        except LSPUnavailableError as exc:
            return ToolResult.err(str(exc), code='lsp-unavailable')

        # --- ensure document is open/synced --------------------------------
        uri = path_to_uri(str(resolved))
        if uri not in MANAGER._open_docs:
            MANAGER._did_open(str(resolved))

        # --- textDocument/formatting ---------------------------------------
        METHOD = 'textDocument/formatting'
        params = {
            'textDocument': {'uri': uri},
            'options': {'tabSize': 4, 'insertSpaces': True},
        }

        try:
            result = client.request(METHOD, params, timeout=15.0)
        except TimeoutError:
            return ToolResult.err(
                'language server did not answer in time',
                code='lsp-timeout',
            )
        except Exception as exc:
            return ToolResult.err(
                f'the {language} language server does not '
                f'support {METHOD} (or the request failed: {exc})',
                code='lsp-capability',
            )

        # --- process result ------------------------------------------------
        if result is None or result == []:
            return ToolResult.ok(f'{raw_path} is already formatted (no edits returned)')

        # --- apply text edits ----------------------------------------------
        from lsp.edits import apply_text_edits  # pylint: disable=import-outside-toplevel

        try:
            with open(resolved, 'r', encoding='utf-8') as fh:
                text = fh.read()
        except OSError as exc:
            return ToolResult.err(f'could not read {raw_path}: {exc}', code='edit-failed')

        try:
            new_text = apply_text_edits(text, result)
        except Exception as exc:
            return ToolResult.err(f'could not apply formatting edits: {exc}', code='edit-failed')

        if new_text == text:
            return ToolResult.ok(f'{raw_path} is already formatted')

        with open(resolved, 'w', encoding='utf-8') as fh:
            fh.write(new_text)

        emit_mutation('changed', str(resolved))

        return ToolResult.ok(
            f'formatted {raw_path} ({len(result)} edit(s) applied)',
            count=len(result),
        )

    # -----------------------------------------------------------------------
    # CLI fallback helper
    # -----------------------------------------------------------------------

    def _run_cli(
        self,
        command: list[str],
        raw_path: str,
        resolved: Path,
    ) -> ToolResult:
        """Run the CLI *command* on *resolved* as a formatting fallback.

        Args:
            command: The CLI formatter invocation (e.g. ``['black', '--quiet']``).
            raw_path: The original relative path string for error messages.
            resolved: The absolute, sandbox-resolved path.

        Returns:
            A ``ToolResult`` indicating success, timeout, missing binary, or non-zero exit.
        """
        try:
            proc = subprocess.run(
                command + [str(resolved)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            return ToolResult.err(
                f'{command[0]} is not installed',
                code='lsp-unavailable',
            )
        except subprocess.TimeoutExpired:
            return ToolResult.err(
                'formatter timed out after 30 seconds',
                code='lsp-timeout',
            )

        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip()
            return ToolResult.err(
                f'{command[0]} failed: {stderr}',
                code='edit-failed',
            )

        emit_mutation('changed', str(resolved))

        return ToolResult.ok(
            f'formatted {raw_path} with {command[0]}',
            formatter=command[0],
        )
