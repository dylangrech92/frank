"""Dispatch-level check of find_symbol's action routing (no LLM, no language server).

``find_symbol`` absorbed six tools that were deleted (go_to_definition,
go_to_implementation, go_to_type_definition, find_references, hover,
signature_help). All of them required the caller to already know a
file:line:column, so the model had to run find_symbol first anyway; the
``action`` parameter now does the position lookup and the navigation request in
one call.

What this pins is the *routing layer* find_symbol owns -- the part that turns a
name plus an action into a correctly-shaped LSP request -- not the language
server's answer. A fake client records every request instead of talking to a
real server, which keeps the checks deterministic and toolchain-free while
still catching the failures that actually occurred during the fold:

    - ``references`` must send its flag under the key ``context``. The LSP
      specification names that member of ReferenceParams "context"; under any
      other key a real server silently falls back to its default and drops the
      declaration from the results. Nothing about the response shape reveals
      the mistake, so only the recorded request can catch it.
    - a query matching no symbol must be an ERROR for every action except
      ``search``. "Nothing is named X" is a legitimate answer to a search, but
      the other actions were asked to operate ON a symbol, so a missing one is
      a failed precondition -- reporting success there tells the model "this
      symbol has no references" when the truth is "you spelled it wrong".
    - positions come from the server's own index and are already 0-based, so
      they must be forwarded verbatim. The deleted tools subtracted 1 because
      their coordinates were typed by the model; carrying that shift over here
      would silently query the wrong column.
    - each action must reach its own LSP method, and an ambiguous name must be
      refused rather than resolved to an arbitrary first match.

Exits 0 on success, 1 on any assertion failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


# --- fake LSP surface --------------------------------------------------------

class FakeClient:
    """Records every request and replays canned responses."""

    def __init__(self, symbols: list[dict], responses: dict):
        self.symbols = symbols
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def request(self, method: str, params: dict, timeout: float = 10.0):
        self.calls.append((method, params))
        if method == 'workspace/symbol':
            q = params.get('query', '')
            return [s for s in self.symbols if q.lower() in s['name'].lower()]
        return self.responses.get(method)


class FakeManager:
    def __init__(self, root: Path, client: FakeClient):
        self._root_path = root
        self._client = client
        self.opened: list[str] = []

    def snapshot_clients(self):
        return [('python', self._client)]

    def ensure_document_open(self, path: str) -> None:
        self.opened.append(path)

    def language_for_path(self, path: str):
        return 'python' if path.endswith('.py') else None

    def get_client(self, language: str):
        return self._client


def _sym(name: str, rel: str, line: int, char: int, root: Path, kind: int = 12) -> dict:
    return {
        'name': name,
        'kind': kind,
        'location': {
            'uri': f'file://{root}/{rel}',
            'range': {
                'start': {'line': line, 'character': char},
                'end': {'line': line, 'character': char + len(name)},
            },
        },
    }


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()

        # "widget" is unique; "run" appears in two files (the ambiguous case).
        symbols = [
            _sym('widget', 'pkg/core.py', 41, 4, root),
            _sym('run', 'pkg/alpha.py', 10, 0, root),
            _sym('run', 'pkg/beta.py', 20, 0, root),
        ]
        loc = [{
            'uri': f'file://{root}/pkg/core.py',
            'range': {'start': {'line': 41, 'character': 4},
                      'end': {'line': 41, 'character': 10}},
        }]
        responses = {
            'textDocument/definition': loc,
            'textDocument/references': loc,
            'textDocument/implementation': loc,
            'textDocument/typeDefinition': loc,
            'textDocument/hover': {
                'contents': {'kind': 'markdown', 'value': '```python\ndef widget() -> int\n```'},
            },
        }

        client = FakeClient(symbols, responses)
        manager = FakeManager(root, client)

        import main as main_module
        from tools import registry
        from tools.registry import dispatch

        registry.activate_mode('research')
        main_module.MANAGER = manager

        def last_request(method: str):
            for m, p in reversed(client.calls):
                if m == method:
                    return p
            return None

        # ---- 1. default action still returns the flat workspace listing -------
        res = dispatch('find_symbol', {'query': 'widget'})
        check(
            res.status == 'success' and 'widget' in str(res.body),
            f'default search should list matches, got status={res.status!r} '
            f'body={str(res.body)[:200]!r}',
        )
        check(
            '-- 1 symbol' in str(res.body),
            f'default search lost its count footer: {str(res.body)[:200]!r}',
        )

        # ---- 2. each action reaches its own LSP method ------------------------
        for action, method in (
            ('definition', 'textDocument/definition'),
            ('references', 'textDocument/references'),
            ('implementations', 'textDocument/implementation'),
            ('type_definition', 'textDocument/typeDefinition'),
            ('hover', 'textDocument/hover'),
        ):
            client.calls.clear()
            res = dispatch('find_symbol', {'query': 'widget', 'action': action})
            check(
                res.status == 'success',
                f'action={action!r} failed: status={res.status!r} '
                f'code={res.code!r} body={str(res.body)[:200]!r}',
            )
            check(
                any(m == method for m, _ in client.calls),
                f'action={action!r} did not issue {method}; '
                f'issued {[m for m, _ in client.calls]}',
            )

        # ---- 3. references sends includeDeclaration under "context" ----------
        # The bug this catches shipped once: the flag was sent as "options", so
        # the server used its default and the declaration vanished from results.
        client.calls.clear()
        dispatch('find_symbol', {'query': 'widget', 'action': 'references'})
        params = last_request('textDocument/references')
        check(params is not None, 'references issued no textDocument/references request')
        if params is not None:
            check(
                'context' in params,
                f'references must send its flag under "context" (LSP ReferenceParams); '
                f'sent keys {sorted(params)}',
            )
            check(
                params.get('context', {}).get('includeDeclaration') is True,
                f'references must set context.includeDeclaration=True, '
                f'got {params.get("context")!r}',
            )

        # ---- 4. server-supplied positions are forwarded unshifted -------------
        client.calls.clear()
        dispatch('find_symbol', {'query': 'widget', 'action': 'definition'})
        params = last_request('textDocument/definition')
        check(params is not None, 'definition issued no request')
        if params is not None:
            pos = params.get('position', {})
            check(
                pos.get('line') == 41 and pos.get('character') == 4,
                f'position must match the indexed symbol verbatim (0-based, no -1 '
                f'shift): expected line=41 character=4, got {pos!r}',
            )

        # ---- 5. a missing symbol is an error for every action but search ------
        res = dispatch('find_symbol', {'query': 'zzz_absent'})
        check(
            res.status == 'success',
            f'search for a missing name is a valid empty answer, not an error; '
            f'got status={res.status!r} code={res.code!r}',
        )
        for action in ('definition', 'references', 'implementations',
                       'type_definition', 'hover'):
            res = dispatch('find_symbol', {'query': 'zzz_absent', 'action': action})
            check(
                res.status == 'error' and res.code == 'symbol-not-found',
                f'action={action!r} on a missing symbol must be symbol-not-found '
                f'(otherwise "you misspelled it" reads as "it has no results"); '
                f'got status={res.status!r} code={res.code!r}',
            )

        # ---- 6. an ambiguous name is refused, not silently resolved -----------
        client.calls.clear()
        res = dispatch('find_symbol', {'query': 'run', 'action': 'references'})
        check(
            res.status == 'error' and res.code == 'ambiguous-symbol',
            f'an ambiguous name must be refused, got status={res.status!r} '
            f'code={res.code!r}',
        )
        check(
            'path' in str(res.body),
            f'the ambiguity error must tell the caller to pass path: '
            f'{str(res.body)[:200]!r}',
        )
        check(
            not any(m == 'textDocument/references' for m, _ in client.calls),
            'an ambiguous name must not issue a navigation request for an '
            'arbitrarily chosen candidate',
        )

        # ---- 7. path disambiguates, and picks the right one ------------------
        client.calls.clear()
        res = dispatch('find_symbol', {
            'query': 'run', 'action': 'references', 'path': 'pkg/beta.py',
        })
        check(
            res.status == 'success',
            f'path should have disambiguated, got status={res.status!r} '
            f'code={res.code!r} body={str(res.body)[:200]!r}',
        )
        params = last_request('textDocument/references')
        if params is not None:
            check(
                params['textDocument']['uri'].endswith('pkg/beta.py'),
                f'path=pkg/beta.py selected the wrong candidate: '
                f'{params["textDocument"]["uri"]!r}',
            )
            check(
                params['position']['line'] == 20,
                f'path=pkg/beta.py used the other candidate\'s position: '
                f'{params["position"]!r}',
            )

        # ---- 8. an unknown action is rejected before any LSP traffic ---------
        client.calls.clear()
        res = dispatch('find_symbol', {'query': 'widget', 'action': 'teleport'})
        check(
            res.status == 'error' and res.code == 'bad-arguments',
            f'an unknown action must be bad-arguments, got status={res.status!r} '
            f'code={res.code!r}',
        )
        check(
            client.calls == [],
            f'an unknown action must be rejected before any LSP request; '
            f'issued {[m for m, _ in client.calls]}',
        )

        # ---- 9. hover renders the markdown body, not the raw envelope --------
        res = dispatch('find_symbol', {'query': 'widget', 'action': 'hover'})
        check(
            'def widget() -> int' in str(res.body),
            f'hover must render the contents value, got {str(res.body)[:200]!r}',
        )
        check(
            'contents' not in str(res.body) and 'markdown' not in str(res.body),
            f'hover leaked the raw LSP envelope instead of rendering it: '
            f'{str(res.body)[:200]!r}',
        )

    for f in failures:
        print(f'FAIL: {f}')
    if failures:
        print(f'\n{len(failures)} check(s) failed')
        return 1
    print('find_symbol action routing: all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
