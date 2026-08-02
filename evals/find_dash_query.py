"""Dispatch-level check that find treats a dash-leading query as a search term (no LLM).

The find tool shells out to ripgrep.  A pattern passed positionally is parsed by
rg's own flag parser, so a query the model genuinely wants to search for --
``--disable-voice``, ``-Wall``, any CLI flag -- came back as
``search-failed: rg: unrecognized flag``.  Observed in the wild: a research-mode
run searching an installer for ``--disable-voice`` got that error and concluded
the harness was broken.  The same hazard applies to the path argument, which
needs a ``--`` terminator to be read as a path rather than a flag.

    (a) a ``--`` and a ``-`` leading query both match, literally and fuzzily.
    (b) a dash-leading directory name is accepted as the path argument.
    (c) ordinary queries are unaffected: a match still reports its file:line,
        and a genuine no-match is still success with match_count 0.

Exits 0 on success, 1 on any assertion failure. Runs with the repo root on
``sys.path`` (evals/run.py inserts it before exec'ing this file); all file work
happens inside a fresh temp dir that becomes the project root via chdir.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    from tools import registry
    from tools.registry import dispatch

    failures: list[str] = []

    # find ships in every mode via modes._COMMON_TOOLS; 'code' is as good as any.
    registry.activate_mode('code')

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix='find-dash-eval-') as tmp:
        root = Path(tmp)
        os.chdir(root)
        try:
            _run_checks(dispatch, root, check)
        finally:
            os.chdir(original_cwd)

    for failure in failures:
        print(f'FAIL: {failure}', file=sys.stderr)
    print(f'find_dash_query: {len(failures)} failure(s)')
    return 1 if failures else 0


def _run_checks(dispatch, root: Path, check) -> None:
    """Drive find over a project whose content and layout both carry dashes."""
    (root / 'build.sh').write_text(
        'run --disable-voice here\ncc -Wall main.c\nplain line\n', encoding='utf-8'
    )

    # ---- (a) dash-leading queries are search terms, not flags. -------------
    for query, expected in (('--disable-voice', 'build.sh:1'), ('-Wall', 'build.sh:2')):
        res = dispatch('find', {'query': query})
        check(
            res.status == 'success' and expected in str(res.body),
            f'query {query!r} must match {expected}, '
            f'got status={res.status!r} code={res.code!r} body={str(res.body)[:120]!r}',
        )

    res = dispatch('find', {'query': '--disable-voice', 'fuzzy': True})
    check(
        res.status == 'success' and 'build.sh:1' in str(res.body),
        f'fuzzy dash query must match, got status={res.status!r} code={res.code!r}',
    )

    # ---- (b) a dash-leading directory is a path, not a flag. ---------------
    odd = root / '-tmpdir'
    odd.mkdir()
    (odd / 'x.txt').write_text('needle here\n', encoding='utf-8')
    res = dispatch('find', {'query': 'needle', 'path': '-tmpdir'})
    check(
        res.status == 'success' and 'x.txt' in str(res.body),
        f'dash-leading path must be searched, got status={res.status!r} code={res.code!r}',
    )

    # ---- (c) ordinary queries are unchanged. -------------------------------
    res = dispatch('find', {'query': 'plain'})
    check(
        res.status == 'success' and 'build.sh:3' in str(res.body),
        f'plain query regressed: status={res.status!r} body={str(res.body)[:120]!r}',
    )
    res = dispatch('find', {'query': 'nothingmatchesthis'})
    check(
        res.status == 'success' and res.meta.get('match_count') == 0,
        f'a genuine no-match must stay success/0, got status={res.status!r} meta={res.meta!r}',
    )


if __name__ == '__main__':
    raise SystemExit(main())
