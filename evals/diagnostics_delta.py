"""Check that the injected LSP diagnostics summary is a delta, not a total (no LLM).

After a mutation round the harness appends the language servers' diagnostics to
the tool result.  Reporting the whole store made that a project-wide total
attached to the model's own write, which reads as feedback on that write.
Observed in the wild: a code-mode run confirmed all four files it had edited were
clean via ``get_diagnostics``, then had ``⚠ 65 errors in 10 files`` appended to
its next successful write.  Given two contradictory signals the model went
hunting for errors it had not caused, guessed at unrelated files, and filed a
fabricated defect report.

    (a) a clean edit against a project full of pre-existing diagnostics appends
        nothing at all.
    (b) a diagnostic the round actually introduced is still reported, counted
        alone, and named -- so the count is attributable.
    (c) breakage in a file the round never edited still surfaces (a delta, not a
        filter on the edited paths).
    (d) a length-changing edit that shifts pre-existing diagnostics onto new
        lines introduces nothing, while a genuinely added duplicate of an
        existing message still counts.
    (e) severity/deprecated/file-cap rendering, and severities 3 and 4 never
        reaching a summary at all.

Exits 0 on success, 1 on any assertion failure. Runs with the repo root on
``sys.path`` (evals/run.py inserts it before exec'ing this file).  Drives a real
``DiagnosticsStore`` with real ``textDocument/publishDiagnostics`` payloads --
the store is the unit under test, so nothing about it is stubbed.
"""

from __future__ import annotations

import sys


def _diag(line: int, msg: str, severity: int | None = 1, tags: list[int] | None = None) -> dict:
    """Build one LSP diagnostic exactly as lsp/manager.py delivers it."""
    diagnostic: dict = {
        'severity': severity,
        'message': msg,
        'range': {
            'start': {'line': line, 'character': 0},
            'end': {'line': line, 'character': 1},
        },
    }
    if tags:
        diagnostic['tags'] = tags
    return diagnostic


EDITED = 'file:///project/tests/Feature/ContactTest.php'


def _noisy_store(store_cls):
    """Ten untouched files carrying 20 pre-existing errors, plus one clean edited file."""
    store = store_cls()
    for n in range(10):
        store.handle_publish({
            'uri': f'file:///project/app/Untouched{n}.php',
            'diagnostics': [_diag(1, "Undefined type 'Foo'"), _diag(2, "Undefined method 'bar'")],
        })
    store.handle_publish({'uri': EDITED, 'diagnostics': []})
    return store


def main() -> int:
    from diagnostics import DiagnosticsStore

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # ---- (a) a clean edit against a noisy project stays silent. ------------
    store = _noisy_store(DiagnosticsStore)
    baseline = store.snapshot_issues()
    store.handle_publish({'uri': EDITED, 'diagnostics': []})
    out = store.summary(baseline)
    check(
        out is None,
        f'a clean edit against 20 pre-existing errors must append nothing, got {out!r}',
    )

    # ---- (b) a genuinely new diagnostic is reported and attributed. --------
    store = _noisy_store(DiagnosticsStore)
    baseline = store.snapshot_issues()
    store.handle_publish({'uri': EDITED, 'diagnostics': [_diag(11, "Undefined variable '$reponse'")]})
    out = str(store.summary(baseline))
    check('1 new error' in out, f'the 1 new error must be counted alone, got {out!r}')
    check('ContactTest.php' in out, f'the offending file must be named, got {out!r}')
    check('Untouched' not in out, f'a file the round did not break must not be named, got {out!r}')

    # ---- (c) breakage in an unedited file still surfaces. ------------------
    store = _noisy_store(DiagnosticsStore)
    baseline = store.snapshot_issues()
    store.handle_publish({
        'uri': 'file:///project/app/Untouched3.php',
        'diagnostics': [_diag(1, "Undefined type 'Foo'"), _diag(2, "Undefined method 'bar'"),
                        _diag(9, 'Call to undefined method contactForm()')],
    })
    out = str(store.summary(baseline))
    check('1 new error' in out, f'breakage in an unedited file must surface, got {out!r}')
    check('Untouched3.php' in out, f'the broken unedited file must be named, got {out!r}')

    # ---- (d) shifted lines are not new; added duplicates are. --------------
    store = DiagnosticsStore()
    store.handle_publish({'uri': EDITED, 'diagnostics': [_diag(40, 'pre-existing A'),
                                                        _diag(50, 'pre-existing B')]})
    baseline = store.snapshot_issues()
    store.handle_publish({'uri': EDITED, 'diagnostics': [_diag(42, 'pre-existing A'),
                                                        _diag(52, 'pre-existing B')]})
    out = store.summary(baseline)
    check(out is None, f'a pre-existing diagnostic that merely shifted line is not new, got {out!r}')

    store.handle_publish({'uri': EDITED, 'diagnostics': [_diag(42, 'pre-existing A'),
                                                        _diag(52, 'pre-existing B'),
                                                        _diag(60, 'pre-existing A')]})
    out = str(store.summary(baseline))
    check('1 new error' in out, f'a genuinely added duplicate message must count, got {out!r}')

    # ---- (e) rendering. ----------------------------------------------------
    store = DiagnosticsStore()
    baseline = store.snapshot_issues()
    store.handle_publish({'uri': 'file:///project/a.php', 'diagnostics': [
        _diag(1, 'boom', severity=None), _diag(2, 'careful', severity=2),
        _diag(3, 'old api', tags=[2])]})
    out = str(store.summary(baseline))
    check('2 new errors' in out, f'a missing severity counts as an error, got {out!r}')
    check('1 new warning' in out, f'severity 2 counts as a warning, got {out!r}')
    check('[deprecated]' in out, f'the deprecated tag must surface, got {out!r}')

    store = DiagnosticsStore()
    baseline = store.snapshot_issues()
    for n in range(6):
        store.handle_publish({'uri': f'file:///project/f{n}.php', 'diagnostics': [_diag(1, f'e{n}')]})
    out = str(store.summary(baseline))
    check('+3 more' in out, f'the file list must cap at 3 with a +N more tail, got {out!r}')

    store = DiagnosticsStore()
    baseline = store.snapshot_issues()
    store.handle_publish({'uri': 'file:///project/a.php', 'diagnostics': [
        _diag(1, 'fyi', severity=3), _diag(2, 'hint', severity=4)]})
    out = store.summary(baseline)
    check(out is None, f'severities 3 and 4 must never reach a summary, got {out!r}')

    for failure in failures:
        print(f'FAIL: {failure}', file=sys.stderr)
    print(f'diagnostics_delta: {len(failures)} failure(s)')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
