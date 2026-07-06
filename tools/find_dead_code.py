"""Find-dead-code tool: sweep a file/project for unreachable or unused code."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from lsp.locations import flatten_symbols, normalize_locations
from lsp.manager import LSPUnavailableError, path_to_uri
from tools._sandbox import resolve_in_root
from tools.base import Tool
from tools.result import ToolResult

# Directories never walked while sniffing a directory target's language mix.
_IGNORED_DIRS = frozenset({'.git', '.coding_agent', '__pycache__', 'node_modules', '.venv', 'venv'})

_PY_SUFFIXES = frozenset({'.py'})
_TS_SUFFIXES = frozenset({'.ts', '.tsx', '.js', '.jsx'})

# vulture's Item.get_report() renders exactly:
#   "{path}:{lineno}: {message} ({confidence}% confidence[, N lines])"
# where message defaults to "unused {typ} '{name}'" but is a free-form string
# for the "unreachable_code" item type (e.g. "unreachable code after 'return'").
_VULTURE_LINE_RE = re.compile(
    r"^(?P<path>.+):(?P<line>\d+):\s(?P<message>.+?)\s"
    r"\((?P<conf>\d+)% confidence(?:,\s\d+\s(?:line|lines))?\)$"
)

# Symbol kinds that are structural containers, not something that can itself be
# "unused" — skipped when walking document_symbols for the LSP fallback.
_STRUCTURAL_KINDS = frozenset({'file', 'module', 'namespace', 'package'})


def _has_files_with_suffix(directory: Path, suffixes: frozenset[str]) -> bool:
    """Return True as soon as a file with a matching suffix is found under *directory*.

    Skips common vendored/build/vcs directories so the sniff stays cheap.
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for fn in filenames:
            if Path(fn).suffix in suffixes:
                return True
    return False


def _parse_vulture_output(stdout: str) -> list[dict[str, Any]]:
    """Parse vulture's line-oriented report into structured findings.

    Args:
        stdout: Raw stdout from a vulture invocation.

    Returns:
        A list of dicts with keys ``path``, ``line``, ``message``, ``confidence``.
        Lines that do not match vulture's report format (e.g. stray warnings) are
        silently skipped rather than raising — this parser must never crash on
        unexpected vulture output.
    """
    findings: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        match = _VULTURE_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        findings.append({
            'path': match.group('path'),
            'line': int(match.group('line')),
            'message': match.group('message'),
            'confidence': int(match.group('conf')),
        })
    return findings


class FindDeadCode(Tool):
    """Sweeps a file or the project for unreachable/unused code.

    Adapter selection (``mode='auto'``, the default):

    - Python targets: `vulture <https://github.com/jendrikseipp/vulture>`_ when
      installed on ``PATH``.
    - TS/JS targets: ``knip`` or ``ts-prune`` when installed on ``PATH`` (never
      auto-installed); otherwise skipped with a clear note.
    - Any LSP-supported language, as a fallback (or when no CLI adapter applies):
      walks ``textDocument/documentSymbol`` for the target file and counts
      ``textDocument/references`` per symbol (excluding the declaration itself).
      Zero-reference symbols are reported as *potentially* dead with a caveat —
      this path only ever inspects a single file, not a whole directory.

    Pass ``mode='vulture'`` or ``mode='lsp'`` to force a specific adapter.
    """

    name = 'find_dead_code'
    parallel_safe = True  # spawns a read-only subprocess or issues read-only LSP queries
    summary = 'Sweep a file/project for unreachable or unused code.'
    description = (
        'Sweeps a file or the whole project for dead code. Uses vulture for Python, '
        'knip/ts-prune for TS/JS when installed, and falls back to an LSP-based '
        'document_symbols + find_references sweep for any other supported language. '
        'Findings are reported as potentially dead — verify before deleting.'
    )
    action = 'sweep for dead code'
    oversize_hint = 'narrow the scan to a subdirectory or a single file'
    alternative = 'find_references for a single known symbol'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': (
                    'File or directory to scan, relative to the project root. '
                    'Defaults to the whole project.'
                ),
            },
            'mode': {
                'type': 'string',
                'description': (
                    "Adapter selection: 'auto' (default) picks vulture/knip/ts-prune "
                    "or the LSP fallback based on the target; 'vulture' forces the "
                    "vulture adapter; 'lsp' forces the document_symbols/find_references "
                    "fallback and requires path to point at a single file."
                ),
            },
        },
        'required': [],
    }

    MAX_FINDINGS = 200

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the dead-code sweep.

        Args:
            **kwargs: Parsed from LLM function-call payload. Optional ``path``
                (str, default the whole project) and optional ``mode`` (str,
                one of ``'auto'``, ``'vulture'``, ``'lsp'``; default ``'auto'``).

        Returns:
            A ``ToolResult`` with a rendered findings list (or a "none found"
            message) on success, or an error when validation fails, the
            requested adapter is unavailable, or the scan itself fails.
        """
        raw_path = kwargs.get('path') if isinstance(kwargs.get('path'), str) else '.'
        mode = kwargs.get('mode') if isinstance(kwargs.get('mode'), str) else 'auto'

        if mode not in ('auto', 'vulture', 'lsp'):
            return ToolResult.err(
                f"mode must be one of 'auto', 'vulture', 'lsp', got {mode!r}",
                code='bad-arguments',
            )

        root = Path.cwd()
        try:
            target = resolve_in_root(root, raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code='path-escapes-root')

        if not target.exists():
            return ToolResult.err(
                f'{target} does not exist.',
                code='not-found',
            )

        if mode == 'lsp':
            if not target.is_file():
                return ToolResult.err(
                    'lsp mode requires path to point at a single file '
                    '(document_symbols is scoped per-file).',
                    code='not-a-file',
                )
            return self._lsp_fallback(target, root)

        if mode == 'vulture':
            vulture_bin = shutil.which('vulture')
            if vulture_bin is None:
                return ToolResult.err(
                    'the vulture binary was not found on PATH',
                    code='missing-engine',
                    hint='Install it with: pip install vulture',
                )
            return self._run_vulture(vulture_bin, target, root)

        # --- mode == 'auto' ---------------------------------------------------
        is_py = (
            target.suffix in _PY_SUFFIXES if target.is_file()
            else _has_files_with_suffix(target, _PY_SUFFIXES)
        )
        if is_py:
            vulture_bin = shutil.which('vulture')
            if vulture_bin is not None:
                return self._run_vulture(vulture_bin, target, root)
            note = 'vulture is not installed (pip install vulture) — Python sweep skipped.'
            if target.is_file():
                return self._lsp_fallback(target, root, note=note)
            return ToolResult.ok(note, adapter='none')

        is_ts = (
            target.suffix in _TS_SUFFIXES if target.is_file()
            else _has_files_with_suffix(target, _TS_SUFFIXES)
        )
        if is_ts:
            ts_bin = shutil.which('knip') or shutil.which('ts-prune')
            if ts_bin is not None:
                return self._run_ts_adapter(ts_bin, target, root)
            note = (
                'knip/ts-prune are not installed and were not auto-installed — '
                'TS/JS dead-code sweep skipped.'
            )
            if target.is_file():
                return self._lsp_fallback(target, root, note=note)
            return ToolResult.ok(note, adapter='none')

        # Neither Python nor TS/JS: fall back to the LSP path for a single file
        # (any other LSP-supported language); a directory has no adapter to try.
        if target.is_file():
            return self._lsp_fallback(target, root)

        return ToolResult.err(
            'no dead-code adapter applies to this directory (no Python or TS/JS '
            'files found)',
            code='no-adapter',
            hint='Pass path to a single file to use the LSP-based fallback.',
        )

    # ------------------------------------------------------------------
    # vulture adapter
    # ------------------------------------------------------------------

    def _run_vulture(self, vulture_bin: str, target: Path, root: Path) -> ToolResult:
        """Run vulture against *target* and render its findings.

        Args:
            vulture_bin: Absolute path to the vulture executable.
            target: Resolved absolute path to scan (file or directory).
            root: Project root, used as the subprocess cwd so vulture emits
                root-relative paths.

        Returns:
            A ``ToolResult`` with rendered findings, a "no dead code" message,
            or an error when vulture itself fails (bad syntax, missing target).
        """
        rel = os.path.relpath(str(target), root) or '.'

        result = subprocess.run(
            [vulture_bin, rel],
            capture_output=True,
            text=True,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
        )

        # vulture exit codes: 0 = clean, 3 = findings reported, anything else
        # (1 typically) = a scan error (missing target, syntax error, ...).
        if result.returncode not in (0, 3):
            message = result.stdout.strip() or result.stderr.strip() or (
                f'vulture exited with code {result.returncode}'
            )
            return ToolResult.err(
                f'vulture scan failed: {message}',
                code='dead-code-scan-failed',
            )

        if result.returncode == 0:
            return ToolResult.ok(
                f'no dead code found by vulture in {rel}',
                path=rel,
                adapter='vulture',
                count=0,
            )

        findings = _parse_vulture_output(result.stdout)
        if not findings:
            # Findings were reported (exit 3) but the report format didn't match
            # what we parse for — surface the raw output rather than hide it.
            return ToolResult.ok(
                result.stdout.strip() or 'vulture reported findings in an unrecognized format',
                path=rel,
                adapter='vulture',
            )

        total = len(findings)
        capped = findings[: self.MAX_FINDINGS]
        lines = [
            f"{f['path']}:{f['line']}: {f['message']} ({f['confidence']}% confidence)"
            for f in capped
        ]
        if total > len(capped):
            lines.append(
                f'-- showing {len(capped)} of {total} findings; '
                f'narrow path to see the rest'
            )
        else:
            lines.append(f"-- {total} finding" + ('' if total == 1 else 's'))

        return ToolResult.ok(
            '\n'.join(lines),
            path=rel,
            adapter='vulture',
            count=total,
        )

    # ------------------------------------------------------------------
    # knip / ts-prune adapter
    # ------------------------------------------------------------------

    def _run_ts_adapter(self, ts_bin: str, target: Path, root: Path) -> ToolResult:
        """Run whichever of knip/ts-prune is available on *target*.

        Both tools are project-config-driven (they discover the surrounding
        ``tsconfig.json``/``package.json`` themselves), so the process is run
        from the project root with no target argument beyond that context.

        Args:
            ts_bin: Absolute path to the knip or ts-prune executable.
            target: Resolved absolute path (used only to compute a relative
                label — both tools scan from the project root).
            root: Project root, used as the subprocess cwd.

        Returns:
            A ``ToolResult`` wrapping the tool's raw stdout, or an error when
            the process itself fails to run.
        """
        adapter_name = Path(ts_bin).name
        rel = os.path.relpath(str(target), root) or '.'

        try:
            result = subprocess.run(
                [ts_bin],
                capture_output=True,
                text=True,
                cwd=str(root),
                stdin=subprocess.DEVNULL,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.err(
                f'{adapter_name} did not finish in time',
                code='dead-code-scan-timeout',
            )
        except OSError as exc:
            return ToolResult.err(
                f'{adapter_name} failed to run: {exc}',
                code='dead-code-scan-failed',
            )

        output = (result.stdout or '').strip()
        if not output:
            output = (result.stderr or '').strip()

        if not output:
            return ToolResult.ok(
                f'no dead code found by {adapter_name}',
                path=rel,
                adapter=adapter_name,
                count=0,
            )

        lines = output.splitlines()[: self.MAX_FINDINGS]
        body = '\n'.join(lines)
        if len(output.splitlines()) > len(lines):
            body += f'\n-- output truncated to {len(lines)} lines; narrow path to see the rest'

        return ToolResult.ok(body, path=rel, adapter=adapter_name)

    # ------------------------------------------------------------------
    # LSP fallback adapter
    # ------------------------------------------------------------------

    def _lsp_fallback(self, resolved: Path, root: Path, *, note: str | None = None) -> ToolResult:
        """Sweep a single file for zero-reference symbols via the LSP layer.

        Reuses the exact document_symbols / find_references request shapes
        used by ``tools/document_symbols.py`` and ``tools/find_references.py``
        against the already-running LSP manager — no separate client is spawned.

        Args:
            resolved: Resolved absolute path to the target file.
            root: Project root, used to render a relative path.
            note: Optional note (e.g. "vulture not installed") prepended to a
                successful result's body.

        Returns:
            A ``ToolResult`` listing potentially dead symbols with a
            verify-before-deleting caveat, a "none found" message, or an error
            when no language server is available for this file type.
        """
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
                f'no language server configured for this file type ({resolved.suffix})',
                code='lsp-unavailable',
                hint='Try get_diagnostics for a supported file type.',
            )

        try:
            client = MANAGER.get_client(language)
        except LSPUnavailableError as exc:
            return ToolResult.err(str(exc), code='lsp-unavailable')

        uri = path_to_uri(str(resolved))
        MANAGER.ensure_document_open(str(resolved))

        SYMBOL_METHOD = 'textDocument/documentSymbol'
        try:
            sym_result = client.request(
                SYMBOL_METHOD,
                {'textDocument': {'uri': uri}},
                timeout=10.0,
            )
        except TimeoutError:
            return ToolResult.err(
                'language server did not answer in time',
                code='lsp-timeout',
            )
        except Exception as exc:
            return ToolResult.err(
                f'the {language} language server does not support {SYMBOL_METHOD} '
                f'(or the request failed: {exc})',
                code='lsp-capability',
            )

        rel = os.path.relpath(str(resolved), root)
        syms = flatten_symbols(sym_result)
        candidates = [s for s in syms if s.get('kind') not in _STRUCTURAL_KINDS]

        if not candidates:
            body = f'no symbols to check in {rel}'
            return ToolResult.ok(self._with_note(body, note), path=rel, adapter='lsp-fallback', count=0)

        # Some servers report flat SymbolInformation whose range starts at the
        # declaration keyword (e.g. column 0 of "def foo(...)"), not the name
        # itself — a references request there resolves nothing. Re-locate the
        # identifier's own column on its reported line so positions always land
        # on the symbol name.
        try:
            file_lines = resolved.read_text(errors='replace').splitlines()
        except OSError:
            file_lines = []

        REFS_METHOD = 'textDocument/references'
        dead: list[tuple[str, str, int, int]] = []
        checked = 0
        for sym in candidates:
            line0 = sym.get('line', 0)
            char0 = self._identifier_column(file_lines, line0, sym.get('name', ''), sym.get('character', 0))
            try:
                refs_result = client.request(
                    REFS_METHOD,
                    {
                        'textDocument': {'uri': uri},
                        'position': {'line': line0, 'character': char0},
                        # excludeDeclaration -- we only want *other* usages
                        'context': {'includeDeclaration': False},
                    },
                    timeout=10.0,
                )
            except Exception:
                # Skip symbols the server can't answer for rather than aborting
                # the whole sweep over one uncooperative position.
                continue

            checked += 1
            refs = normalize_locations(refs_result)
            if not refs:
                dead.append((sym.get('name', ''), sym.get('kind', 'symbol'), line0 + 1, char0 + 1))

        if not dead:
            body = f'no potentially dead symbols found in {rel} ({checked} symbols checked)'
            return ToolResult.ok(self._with_note(body, note), path=rel, adapter='lsp-fallback', count=0)

        total = len(dead)
        capped = dead[: self.MAX_FINDINGS]
        lines = [
            f"{rel}:{ln}:{col} potentially dead {kind} '{name}'"
            for name, kind, ln, col in capped
        ]
        if total > len(capped):
            lines.append(
                f'-- showing {len(capped)} of {total} potentially dead symbols; '
                f'narrow path to see the rest'
            )
        else:
            lines.append(f"-- {total} potentially dead symbol" + ('' if total == 1 else 's'))
        lines.append('verify before deleting (dynamic dispatch, exports, reflection)')

        return ToolResult.ok(
            self._with_note('\n'.join(lines), note),
            path=rel,
            adapter='lsp-fallback',
            count=total,
        )

    @staticmethod
    def _identifier_column(file_lines: list[str], line0: int, name: str, fallback_char: int) -> int:
        """Return the column where *name* starts on line *line0*, defaulting to *fallback_char*.

        Args:
            file_lines: The target file's content, split into lines.
            line0: 0-based line index reported by the language server.
            name: The symbol's identifier text to locate.
            fallback_char: Column to use when *name* cannot be found on the line.

        Returns:
            The 0-based column of *name* on the line, or *fallback_char* when the
            line is out of range, *name* is empty, or it isn't found there.
        """
        if not name or not (0 <= line0 < len(file_lines)):
            return fallback_char
        idx = file_lines[line0].find(name, fallback_char)
        if idx == -1:
            idx = file_lines[line0].find(name)
        return idx if idx != -1 else fallback_char

    @staticmethod
    def _with_note(body: str, note: str | None) -> str:
        """Prepend *note* (e.g. a skipped-adapter explanation) to *body* when set."""
        if not note:
            return body
        return f'{note}\n{body}'
