"""Find-symbol tool: fuzzy-search symbols by name across the whole workspace."""

from __future__ import annotations

from typing import Any

from lsp.locations import flatten_symbols, render_locations, render_symbol_line
from lsp.manager import LSPUnavailableError, uri_to_path
from tools.base import Tool
from tools.result import ToolResult


VALID_ACTIONS = ("search", "definition", "references", "implementations",
                 "type_definition", "hover")

_LOCATION_LSP_METHOD: dict[str, str] = {
    "definition": "textDocument/definition",
    "references": "textDocument/references",
    "implementations": "textDocument/implementation",
    "type_definition": "textDocument/typeDefinition",
}


def _check_action(action: Any) -> str | None:
    """Return error message if *action* is invalid; otherwise ``None``."""
    if not isinstance(action, str) or action not in VALID_ACTIONS:
        allowed = ", ".join(VALID_ACTIONS)
        return (
            f'action must be one of {allowed}; '
            f'got {action!r}.'
        )
    return None


def _sym_matches_path(sym: dict, relative_path: str) -> bool:
    uri = sym.get("uri")
    if not isinstance(uri, str):
        return False
    try:
        abs_p = uri_to_path(uri)
    except Exception:  # noqa: E722
        return False
    norm = str(abs_p).replace("\\", "/")
    target = relative_path.lstrip("/").replace("\\", "/")
    return norm == target or norm.endswith("/" + target)


def _render_hover_response(response: Any) -> str:
    """Return the textual/markdown contents of an LSP ``HoverResponse``."""
    if not isinstance(response, dict):
        return repr(response) if response is not None else ""
    contents = response.get("contents")
    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents.strip()
    if isinstance(contents, dict):
        kind = contents.get("kind")
        if kind in ("markdown", "plaintext") and isinstance(contents.get("value"), str):
            return contents["value"].strip()
        if "code" in contents and isinstance(contents["code"], str):
            lang = contents.get("language") or "text"
            return f"```{lang}\n{contents['code']}\n```".strip()
        return repr(contents)
    if isinstance(contents, list):
        parts = [_render_hover_response(c) for c in contents]
        return "\n".join(p for p in parts if p)
    return repr(contents)


class FindSymbol(Tool):
    """Fuzzy-search symbols by name across the whole workspace (like IDE ctrl+t).

    By default runs a ``workspace/symbol`` query across the entire project — files that
    have never been opened in the editor are still covered because the Language Server
    Protocol initialize handshake advertises ``rootUri`` and/or ``workspaceFolders`` so
    servers can index the whole tree without per-file synchronization.

    With ``action`` set to one of ``"definition"`` / ``"references"`` /
    ``"implementations"`` / ``"type_definition"`` / ``"hover"``, narrows the lookup to
    the matching symbol at its concrete position and dispatches the corresponding LSP
    code-navigation request. Supply ``path`` (relative to the project root) when the
    query resolves to multiple files so the right candidate is chosen.
    """

    name = 'find_symbol'
    parallel_safe = True  # read-only LSP query; client-map read is lock-guarded (F3)
    description = (
        'Fuzzy-search symbols by name across the whole workspace (like an IDE\'s '
        'ctrl+T). Queries the entire project tree via the LSP initialize handshake '
        '(rootUri/workspaceFolders) so files never opened in an editor are still '
        'covered. Use ``action`` to perform definition/references/implementations/'
        'type_definition/hover lookups on a matched symbol, or omit it for the '
        'flat workspace listing.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Symbol name or fragment to search for.',
            },
            'action': {
                'type': 'string',
                'enum': list(VALID_ACTIONS),
                'default': 'search',
                'description': (
                    '"search" (default) returns a flat list of matches across the '
                    'whole workspace; "definition"/"references"/"implementations"/'
                    '"type_definition"/"hover" narrows to the symbol at its '
                    'location and issues the corresponding LSP request.'
                ),
            },
            'path': {
                'type': 'string',
                'description': (
                    'Optional file path relative to the project root, used only '
                    'to disambiguate when the query matches symbols in several files.'
                ),
            },
        },
        'required': ['query'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the find-symbol tool.

        Args:
            **kwargs: Parsed from LLM function-call payload.  Requires ``query`` (str).
                Optional: ``action`` (defaults to ``search``), ``path`` (filter).

        Returns:
            On the default action, a flat ``ToolResult.ok`` list of matching symbols.  For
            the remaining actions, a ``ToolResult`` with definition/reference/etc. info
            at the symbol's position, or an error explaining why.
        """
        raw_query = kwargs.get('query')
        if not isinstance(raw_query, str) or not raw_query.strip():
            return ToolResult.err(
                'query must be a non-empty string.',
                code='bad-arguments',
            )

        action = kwargs.get('action') or 'search'
        path = kwargs.get('path')

        # Validate action FIRST — even a perfectly good query against an unrecognized
        # action must produce a ``bad-arguments`` error (per spec).
        action_err = _check_action(action)
        if action_err is not None:
            return ToolResult.err(action_err, code='bad-arguments')

        # --- language server availability ------------------------------------
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.MANAGER is None:
            return ToolResult.err(
                'no language servers are running',
                code='lsp-unavailable',
            )

        MANAGER = main_module.MANAGER
        # Thread-safe snapshot: plain dict iteration would race a concurrent
        # spawn mutating _clients under _spawn_lock (F3).
        clients = MANAGER.snapshot_clients()
        if not clients:
            return ToolResult.err(
                'no language servers are running',
                code='lsp-unavailable',
            )

        # Deduplicate by client identity, sorted by language name.
        seen: set[int] = set()
        deduped: list[tuple[str, Any]] = []
        for lang, client in clients:
            cid = id(client)
            if cid not in seen:
                seen.add(cid)
                deduped.append((lang, client))
        deduped.sort(key=lambda pair: pair[0])

        # --- query each client -----------------------------------------------
        merged: list[Any] = []
        for _, client in deduped:
            try:
                result = client.request(
                    'workspace/symbol',
                    {'query': raw_query},
                    timeout=10.0,
                )
                if isinstance(result, list):
                    merged.extend(result)
            except Exception:
                pass

        syms = flatten_symbols(merged)
        if not syms:
            # "search" reports an empty result as a successful search -- finding
            # nothing is a legitimate answer to "what is named X". The other
            # actions were asked to operate ON a symbol, so a missing one is a
            # failed precondition: the model needs to tell "your name was wrong"
            # apart from "this symbol genuinely has no references".
            if action == 'search':
                return ToolResult.ok(f'no symbols matching {raw_query!r}')
            return ToolResult.err(
                f'no symbol matching {raw_query!r} — cannot resolve a position '
                f'to ask for its {action}.',
                code='symbol-not-found',
                hint='Try find_symbol with the default action to see what exists.',
            )

        # --- Plain "search": byte-identical to the historical implementation ----
        if action == 'search':
            lines = [render_symbol_line(s, str(MANAGER._root_path)) for s in syms]
            n = len(syms)
            singular = 'symbol' if n == 1 else 'symbols'
            footer = f'-- {n} {singular}'
            body = '\n'.join(lines + [footer])
            return ToolResult.ok(body, count=n)

        # --- Disambiguate among multiple candidates --------------------------
        filtered = syms
        if path is not None:
            filtered = [
                s for s in syms
                if _sym_matches_path(s, path)
            ]

        # Nothing matches after applying the path filter.
        if not filtered:
            msg = f"No symbol '{raw_query}' found"
            if path is not None:
                msg += f" in path {path!r}"
            return ToolResult.err(msg, code='symbol-not-found')
        # Multiple viable candidates still remain — refuse to guess silently.
        if len(filtered) > 1:
            lines = [f"- {render_symbol_line(s, str(MANAGER._root_path))}"
                     for s in filtered]
            body = (
                f'Multiple matches for "{raw_query}":\n'
                + "\n".join(lines)
                + (
                    "\n\n"
                    "Please specify the desired file with the ``path`` parameter "
                    "to disambiguate."
                )
            )
            return ToolResult.err(body, code='ambiguous-symbol')

        # Exactly one match — route the corresponding LSP request.
        sym = filtered[0]
        uri = sym.get('uri')
        line = sym.get('line', 0)
        character = sym.get('character', 0)
        # flatten_symbols yields raw 0-based LSP coordinates, which map 1-to-1 onto
        # the wire protocol. These came from the server, not from the model, so no
        # "- 1" shift belongs here — that conversion is only for 1-based positions
        # a caller typed in.
        base_params: dict[str, Any] = {
            'textDocument': {'uri': uri},
            'position': {'line': line, 'character': character},
        }

        # Locate the responsible client and make sure the document is open.
        if not isinstance(uri, str):
            return ToolResult.err(
                'symbol has no URI — cannot dispatch LSP request',
                code='symbol-not-found',
            )
        try:
            abs_path = uri_to_path(uri)
        except Exception:  # noqa: E722
            return ToolResult.err(
                'cannot translate URI %r to a file path' % uri,
                code='lsp-capability',
            )

        abs_str = str(abs_path)
        MANAGER.ensure_document_open(abs_str)

        try:
            language = MANAGER.language_for_path(abs_str)
            if language is None:
                msg_part = (
                    abs_str.split('.')[-1] if '.' in abs_str else 'unknown'
                )
                return ToolResult.err(
                    'no language server configured for this file type '
                    '(' + msg_part + ')',
                    code='lsp-unavailable',
                    hint='Try get_diagnostics for a supported file type.',
                )
            client = MANAGER.get_client(language)
        except LSPUnavailableError as exc:
            return ToolResult.err(str(exc), code='lsp-unavailable')

        if action == 'hover':
            try:
                hover = client.request(
                    'textDocument/hover',
                    base_params,
                    timeout=10.0,
                )
            except TimeoutError:
                return ToolResult.err(
                    'language server did not answer in time',
                    code='lsp-timeout',
                )
            except Exception as exc:
                return ToolResult.err(
                    f'the {language} language server did not '
                    f'support textDocument/hover '
                    f'(or the request failed: {exc})',
                    code='lsp-capability',
                )
            text = _render_hover_response(hover)
            return ToolResult.ok(text)

        method = _LOCATION_LSP_METHOD[action]
        params = dict(base_params)
        if action == 'references':
            # LSP names this member "context", not "options" (ReferenceParams
            # in the specification). Under the wrong key the server falls back
            # to its default and silently drops the declaration from results.
            params['context'] = {'includeDeclaration': True}

        try:
            result = client.request(method, params, timeout=10.0)
        except TimeoutError:
            return ToolResult.err(
                'language server did not answer in time',
                code='lsp-timeout',
            )
        except Exception as exc:
            return ToolResult.err(
                f'the {language} language server does not '
                f'support {method} (or the request failed: {exc})',
                code='lsp-capability',
            )

        if result is None:
            return ToolResult.ok(f'no {method} result for {raw_query!r}')
        if isinstance(result, list) and len(result) == 0:
            return ToolResult.ok(f'no {method} result for {raw_query!r}')

        root = str(MANAGER._root_path)
        lines = render_locations(result, root)
        if not lines:
            return ToolResult.ok(f'no {method} result for {raw_query!r}')

        return ToolResult.ok('\n'.join(lines))
