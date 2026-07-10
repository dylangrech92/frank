"""Dispatch-level check of the edit_lines tool and read_file numbering (no LLM).

E3 gives the model a line-anchored edit path so it stops rewriting whole files
with update_file. This script drives that surface directly, in a throwaway temp
project, and asserts its load-bearing properties without any live LLM:

    (a) a range replacement swaps exactly the named lines and leaves the rest
        untouched, and the success body carries a cat -n numbered preview.
    (b) a pure insertion (end_line = start_line - 1) adds lines without deleting
        an adjacent one; insert-at-top (start_line=1, end_line=0) prepends.
    (c) out-of-range calls are rejected with code=bad-range (start<1,
        end<start-1, start>total+1, end>total).
    (d) the freshness gates fire: an unread file is rejected not-read-yet, and a
        file changed on disk after the last read is rejected file-changed-on-disk.
    (e) trailing-newline behavior is preserved: a file with a trailing newline
        keeps it, one without stays without, and new_text's own trailing newline
        never spawns a spurious blank line.
    (f) read_file emits cat -n output whose numbers are the TRUE file line
        numbers even when paging with start_line.
    (g) a length-changing edit does not re-stamp the read registry (a second
        line-anchored edit against shifted numbering is refused until re-read,
        and the success body says so), while same-length edits chain freely.

Exits 0 on success, 1 on any assertion failure. Runs with the repo root on
``sys.path`` (evals/run.py inserts it before exec'ing this file); all file work
happens inside a fresh temp dir that becomes the project root via chdir.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

_NUMBERED_LINE = re.compile(r'^\s*\d+\t', re.MULTILINE)


def main() -> int:
    from tools.registry import discover, dispatch

    discover()
    failures: list[str] = []

    # edit_lines is gated (not pinned) on purpose — load it before dispatching,
    # exactly as a protocol-following model would.
    loaded = dispatch('load_tool', {'name': 'edit_lines'})
    if loaded.status != 'success':
        print(f'FAIL: load_tool(edit_lines) failed: {loaded.body!r}')
        return 1

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix='edit-lines-eval-') as tmp:
        root = Path(tmp)
        os.chdir(root)
        try:
            _run_checks(dispatch, root, check)
        finally:
            os.chdir(original_cwd)

    for f in failures:
        print(f'FAIL: {f}')
    if not failures:
        print('edit_lines inline checks passed')
    return 1 if failures else 0


def _run_checks(dispatch, root: Path, check) -> None:  # type: ignore[no-untyped-def]
    # ---- (d) unread gate: edit before any read is rejected. ----------------
    (root / 'unread.py').write_text('a\nb\nc\n', encoding='utf-8')
    res = dispatch('edit_lines', {'path': 'unread.py', 'start_line': 1, 'end_line': 1, 'new_text': 'X'})
    check(
        res.status == 'error' and res.code == 'not-read-yet',
        f'unread file must be rejected not-read-yet, got status={res.status!r} code={res.code!r}',
    )

    # ---- (a) range replacement + numbered preview. -------------------------
    (root / 'range.py').write_text('one\ntwo\nthree\nfour\nfive\n', encoding='utf-8')
    dispatch('read_file', {'path': 'range.py'})
    res = dispatch('edit_lines', {'path': 'range.py', 'start_line': 2, 'end_line': 3, 'new_text': 'TWO\nTHREE'})
    check(res.status == 'success', f'range replace failed: {res.body!r}')
    check(
        (root / 'range.py').read_text(encoding='utf-8') == 'one\nTWO\nTHREE\nfour\nfive\n',
        f'range replace produced wrong content: {(root / "range.py").read_text(encoding="utf-8")!r}',
    )
    check(
        _NUMBERED_LINE.search(str(res.body)) is not None,
        f'range replace body lacks a cat -n numbered preview: {res.body!r}',
    )
    # Preview must number from the true file lines (line 2 changed → context
    # begins at line 1) and show the new text on its real line numbers.
    check('     2\tTWO' in str(res.body), f'preview missing numbered new line "     2\\tTWO": {res.body!r}')

    # ---- (b) pure insertion (end = start - 1) inserts without deleting. -----
    (root / 'ins.py').write_text('alpha\nbeta\ngamma\n', encoding='utf-8')
    dispatch('read_file', {'path': 'ins.py'})
    res = dispatch('edit_lines', {'path': 'ins.py', 'start_line': 2, 'end_line': 1, 'new_text': 'INSERTED'})
    check(res.status == 'success', f'insertion failed: {res.body!r}')
    check(
        (root / 'ins.py').read_text(encoding='utf-8') == 'alpha\nINSERTED\nbeta\ngamma\n',
        f'insertion corrupted adjacent lines: {(root / "ins.py").read_text(encoding="utf-8")!r}',
    )

    # ---- (b) insert-at-top (start=1, end=0). -------------------------------
    (root / 'top.py').write_text('first\nsecond\n', encoding='utf-8')
    dispatch('read_file', {'path': 'top.py'})
    res = dispatch('edit_lines', {'path': 'top.py', 'start_line': 1, 'end_line': 0, 'new_text': 'HEADER'})
    check(res.status == 'success', f'insert-at-top failed: {res.body!r}')
    check(
        (root / 'top.py').read_text(encoding='utf-8') == 'HEADER\nfirst\nsecond\n',
        f'insert-at-top wrong content: {(root / "top.py").read_text(encoding="utf-8")!r}',
    )

    # ---- (c) bad-range codes. ----------------------------------------------
    (root / 'bad.py').write_text('l1\nl2\nl3\n', encoding='utf-8')  # 3 lines
    dispatch('read_file', {'path': 'bad.py'})
    bad_cases = [
        ({'start_line': 0, 'end_line': 0}, 'start<1'),
        ({'start_line': 3, 'end_line': 1}, 'end<start-1'),
        ({'start_line': 5, 'end_line': 4}, 'start>total+1'),
        ({'start_line': 1, 'end_line': 9}, 'end>total'),
    ]
    for args, label in bad_cases:
        res = dispatch('edit_lines', {'path': 'bad.py', 'new_text': 'x', **args})
        check(
            res.status == 'error' and res.code == 'bad-range',
            f'bad-range case {label} ({args}): expected code=bad-range, got '
            f'status={res.status!r} code={res.code!r}',
        )
    # File must be untouched after the rejected calls.
    check(
        (root / 'bad.py').read_text(encoding='utf-8') == 'l1\nl2\nl3\n',
        'a rejected bad-range call mutated the file',
    )

    # ---- (d) stale gate: file changed on disk after last read. -------------
    (root / 'stale.py').write_text('p\nq\nr\n', encoding='utf-8')
    dispatch('read_file', {'path': 'stale.py'})
    # Mutate on disk behind the session's back (distinct mtime + size).
    (root / 'stale.py').write_text('p\nq\nr\nEXTRA\n', encoding='utf-8')
    res = dispatch('edit_lines', {'path': 'stale.py', 'start_line': 1, 'end_line': 1, 'new_text': 'P'})
    check(
        res.status == 'error' and res.code == 'file-changed-on-disk',
        f'stale file must be rejected file-changed-on-disk, got status={res.status!r} code={res.code!r}',
    )

    # ---- (e) trailing-newline preservation. --------------------------------
    # File WITH a trailing newline keeps it; new_text's own trailing newline
    # must not add a blank line.
    (root / 'nl.py').write_text('x\ny\nz\n', encoding='utf-8')
    dispatch('read_file', {'path': 'nl.py'})
    res = dispatch('edit_lines', {'path': 'nl.py', 'start_line': 2, 'end_line': 2, 'new_text': 'Y\n'})
    check(
        (root / 'nl.py').read_text(encoding='utf-8') == 'x\nY\nz\n',
        f'trailing-newline file corrupted: {(root / "nl.py").read_text(encoding="utf-8")!r}',
    )
    # File WITHOUT a trailing newline stays without one.
    (root / 'nonl.py').write_text('x\ny\nz', encoding='utf-8')
    dispatch('read_file', {'path': 'nonl.py'})
    res = dispatch('edit_lines', {'path': 'nonl.py', 'start_line': 3, 'end_line': 3, 'new_text': 'Z'})
    check(
        (root / 'nonl.py').read_text(encoding='utf-8') == 'x\ny\nZ',
        f'no-trailing-newline file gained/lost a newline: {(root / "nonl.py").read_text(encoding="utf-8")!r}',
    )

    # ---- (h) deletion: empty new_text removes the named range. --------------
    (root / 'del.py').write_text('keep1\ndrop1\ndrop2\nkeep2\n', encoding='utf-8')
    dispatch('read_file', {'path': 'del.py'})
    res = dispatch('edit_lines', {'path': 'del.py', 'start_line': 2, 'end_line': 3, 'new_text': ''})
    check(res.status == 'success', f'deletion failed: {res.body!r}')
    check(
        (root / 'del.py').read_text(encoding='utf-8') == 'keep1\nkeep2\n',
        f'deletion produced wrong content: {(root / "del.py").read_text(encoding="utf-8")!r}',
    )
    # Trailing newline is preserved across the deletion.
    check(
        (root / 'del.py').read_text(encoding='utf-8').endswith('\n'),
        'deletion dropped the trailing newline',
    )
    # Body reads as a deletion (not "replaced") and warns of the negative shift.
    check(
        'deleted lines 2-3' in str(res.body),
        f'deletion body must say "deleted lines 2-3": {res.body!r}',
    )
    check(
        'replaced lines' not in str(res.body),
        f'deletion body must not claim a replacement: {res.body!r}',
    )
    check(
        'shifted by -2' in str(res.body),
        f'deletion body must warn about the negative line shift: {res.body!r}',
    )

    # The tool's own description must advertise empty-new_text deletion so the
    # model can discover the affordance without guessing.
    from tools.edit_lines import EditLines

    check(
        'To DELETE lines start_line..end_line, pass an empty string as new_text.'
        in EditLines.description,
        f'EditLines.description does not advertise empty-new_text deletion: {EditLines.description!r}',
    )

    # ---- (g) anchor-shift guard: a length-changing edit must NOT re-stamp ---
    # the read registry, so a second line-anchored edit against the now-shifted
    # numbering is refused until the model re-reads; a same-length edit keeps
    # numbering intact and chains freely.
    (root / 'shift.py').write_text('a\nb\nc\nd\n', encoding='utf-8')
    dispatch('read_file', {'path': 'shift.py'})
    res = dispatch('edit_lines', {'path': 'shift.py', 'start_line': 2, 'end_line': 1, 'new_text': 'NEW'})
    check(res.status == 'success', f'shift-guard setup insertion failed: {res.body!r}')
    check(
        'shifted by +1' in str(res.body),
        f'length-changing edit body must warn about shifted line numbers: {res.body!r}',
    )
    res = dispatch('edit_lines', {'path': 'shift.py', 'start_line': 4, 'end_line': 4, 'new_text': 'D'})
    check(
        res.status == 'error' and res.code == 'file-changed-on-disk',
        'second line-anchored edit after a length-changing one must be refused '
        f'until re-read, got status={res.status!r} code={res.code!r}',
    )
    dispatch('read_file', {'path': 'shift.py'})
    res = dispatch('edit_lines', {'path': 'shift.py', 'start_line': 4, 'end_line': 4, 'new_text': 'C'})
    check(
        res.status == 'success'
        and (root / 'shift.py').read_text(encoding='utf-8') == 'a\nNEW\nb\nC\nd\n',
        f'edit after re-read failed or hit wrong lines: {(root / "shift.py").read_text(encoding="utf-8")!r}',
    )
    # Same-length edits chain without a re-read (numbering unchanged) and
    # carry no shift warning.
    res = dispatch('edit_lines', {'path': 'shift.py', 'start_line': 1, 'end_line': 1, 'new_text': 'A'})
    check(res.status == 'success', f'first same-length chained edit failed: {res.body!r}')
    check('shifted by' not in str(res.body), f'same-length edit must not warn about shift: {res.body!r}')
    res = dispatch('edit_lines', {'path': 'shift.py', 'start_line': 5, 'end_line': 5, 'new_text': 'DD'})
    check(
        res.status == 'success'
        and (root / 'shift.py').read_text(encoding='utf-8') == 'A\nNEW\nb\nC\nDD\n',
        f'same-length edits must chain without re-read: {(root / "shift.py").read_text(encoding="utf-8")!r}',
    )

    # ---- (f) read_file cat -n numbering with true line numbers under paging.
    (root / 'numbered.py').write_text('L1\nL2\nL3\nL4\nL5\nL6\n', encoding='utf-8')
    res = dispatch('read_file', {'path': 'numbered.py'})
    check(res.status == 'success', f'read_file full failed: {res.body!r}')
    check('     1\tL1' in str(res.body), f'full read missing "     1\\tL1": {res.body!r}')
    check('     6\tL6' in str(res.body), f'full read missing "     6\\tL6": {res.body!r}')

    res = dispatch('read_file', {'path': 'numbered.py', 'start_line': 3, 'end_line': 5})
    check(res.status == 'success', f'read_file paged failed: {res.body!r}')
    body = str(res.body)
    check('     3\tL3' in body, f'paged read must number from TRUE line 3: {body!r}')
    check('     5\tL5' in body, f'paged read missing "     5\\tL5": {body!r}')
    # Page-relative numbering (starting at 1) would be a regression.
    check('     1\tL3' not in body, f'paged read wrongly numbered page-relative: {body!r}')


if __name__ == '__main__':
    sys.exit(main())
