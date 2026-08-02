"""Dispatch-level check of the consolidated file-tools surface: write_file + edit_file (no LLM).

These tools replaced five legacy tools (create_file, update_file, replace_one,
replace_many, edit_lines) with two: write_file (new file or full overwrite with
parent-dirs creation) and edit_file (single-occurrence search/replace). This
script drives both surfaces directly via ``dispatch(...)`` and asserts on
their load-bearing behaviors:

    write_file
    - creates a new file including missing parent directories (e.g. "a/b/c.py")
    - overwrites an existing file once it has been freshly read
    - returns code ``not-read-yet`` on an existing file that was never read
    - returns code ``not-a-file`` when *path* names a directory
    - empty-string ``contents`` truncates a small existing file to 0 bytes
    - refuses a tiny payload aimed at a large file with
      ``destructive-partial-overwrite``, AND asserts the original bytes still
      sit on disk after the refusal (a guard that writes anyway would defeat
      this check's purpose)

    edit_file
    - replaces exactly one unique match successfully
    - returns code ``not-found`` when the search string is absent
    - returns code ``not-unique`` when the search string matches >1 AND the
      file is byte-for-byte unchanged afterward
    - returns code ``not-read-yet`` on a file that was never read
    - refuses to write when the file changed on disk after being read,
      returning ``file-changed-on-disk`` (without touching the file on disk)

Exits 0 on success, 1 on any assertion failure. Runs with the repo root on
``sys.path`` (evals/run.py inserts it before exec'ing this file); all file work
happens inside a fresh temp dir that becomes the project root via chdir. The
temp dir is ``.resolve()``'d before being used as the chdir target so the
read-registry key alignment holds on systems where ``/var`` (etc.) is a
symlink to ``/private/var``.
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
    registry.activate_mode('code')

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix='file-ops-eval-') as tmp:
        root = Path(tmp).resolve()
        os.chdir(root)
        try:
            _run_checks(dispatch, root, check)
        finally:
            os.chdir(original_cwd)

    for f in failures:
        print(f'FAIL: {f}')
    if not failures:
        print('file_ops inline checks passed')
    return 1 if failures else 0


def _run_checks(dispatch, root: Path, check) -> None:  # type: ignore[no-untyped-def]
    # ---- write_file: create-new with nested parent dirs --------------------
    res = dispatch('write_file', {
        'path': 'a/b/c.py',
        'contents': '# module c\nprint(1)\n',
    })
    check(res.status == 'success', f'create-new-nested failed: {res.body!r}')
    # is_file() must short-circuit the read: this is the only assertion in the
    # script whose target exists solely because the tool created it, so an
    # unguarded read_text turns a real failure into a traceback and the
    # collected FAIL lines are never printed.
    nested = root / 'a/b/c.py'
    check(
        nested.is_file()
        and nested.read_text(encoding='utf-8') == '# module c\nprint(1)\n',
        'nested create produced wrong content',
    )

    # ---- write_file: overwrite an existing file after a read ---------------
    (root / 'owrite.txt').write_text('hello world\n', encoding='utf-8')
    dispatch('read_file', {'path': 'owrite.txt'})
    res = dispatch('write_file', {'path': 'owrite.txt', 'contents': 'overwritten\n'})
    check(res.status == 'success', f'overwrite-after-read failed: {res.body!r}')
    check(
        (root / 'owrite.txt').read_text(encoding='utf-8') == 'overwritten\n',
        'overwrite produced wrong content',
    )

    # ---- write_file: not-read-yet on an existing unwritten file -----------
    (root / 'unread_write.txt').write_text('initial content\n', encoding='utf-8')
    res = dispatch('write_file', {'path': 'unread_write.txt', 'contents': 'NEW\n'})
    check(
        res.status == 'error' and res.code == 'not-read-yet',
        f'unread write: expected not-read-yet, got status={res.status!r} code={res.code!r}',
    )

    # ---- write_file: not-a-file when *path* names a directory -------------
    (root / 'adir').mkdir()
    res = dispatch('write_file', {'path': 'adir', 'contents': 'x'})
    check(
        res.status == 'error' and res.code == 'not-a-file',
        f'dir-target: expected not-a-file, got status={res.status!r} code={res.code!r}',
    )

    # ---- write_file: empty contents truncates a small existing file -------
    (root / 'small_trunc.txt').write_text('tiny\n', encoding='utf-8')
    dispatch('read_file', {'path': 'small_trunc.txt'})
    res = dispatch('write_file', {'path': 'small_trunc.txt', 'contents': ''})
    check(res.status == 'success', f'empty-contents truncate failed: {res.body!r}')
    check(
        (root / 'small_trunc.txt').stat().st_size == 0,
        'empty-contents write did not truncate the file to 0 bytes',
    )

    # ---- write_file: destructive-partial-overwrite guard -------------------
    # The guard fires when existing_size >= 4000 AND new_size*20 < existing_size.
    # We build a 5001-byte fixture so the guard triggers; a legitimate full
    # rewrite is intentionally larger than 5 % of existing size and therefore
    # allowed — this makes the check *fail* if the guard ever silently writes
    # (so we explicitly verify the bytes stayed put).
    big_content = 'X' * 5000 + '\n'
    (root / 'big.txt').write_text(big_content, encoding='utf-8')
    dispatch('read_file', {'path': 'big.txt'})

    original_bytes = big_content.encode('utf-8')

    tiny_payload = 'SHORT SNIPPET'
    res = dispatch('write_file', {
        'path': 'big.txt',
        'contents': tiny_payload,
    })
    check(
        res.status == 'error' and res.code == 'destructive-partial-overwrite',
        f'destructive-partial-overwrite guard refused incorrectly: '
        f'status={res.status!r} code={res.code!r} body={res.body!r}',
    )

    # THE KEY ASSERTION: the original bytes survived the refused call.
    actual_after_refusal = (root / 'big.txt').read_bytes()
    check(
        actual_after_refusal == original_bytes,
        (
            'guard refused but wrote anyway: expected '
            f'{len(original_bytes)} bytes on disk, got {len(actual_after_refusal)}; '
            f'head={actual_after_refusal[:40]!r}'
        ),
    )

    # Explicit "legitimate full rewrite" case that should NOT trigger the
    # guard (new_size * 20 < existing_size must evaluate to False): a 5001-byte
    # payload replacing a 5001-byte file means 5001*20 = 100020 >> 5001, so the
    # guard is not triggered and the write proceeds normally.
    legitimate_rewrite = '=' * 5000 + '\n'
    res = dispatch('write_file', {
        'path': 'big.txt',
        'contents': legitimate_rewrite,
    })
    check(
        res.status == 'success',
        f'legitimate full rewrite was blocked: {res.body!r}',
    )

    # ---- edit_file: unique match replacement ------------------------------
    (root / 'uni_edit.txt').write_text('one\ntwo\nthree\n', encoding='utf-8')
    dispatch('read_file', {'path': 'uni_edit.txt'})
    res = dispatch('edit_file', {
        'path': 'uni_edit.txt',
        'search': 'two',
        'replace': 'TWO',
    })
    check(res.status == 'success', f'unique edit failed: {res.body!r}')
    check(
        (root / 'uni_edit.txt').read_text(encoding='utf-8') == 'one\nTWO\nthree\n',
        'unique edit produced wrong content',
    )

    # ---- edit_file: not-found when search string is absent ----------------
    (root / 'nf_edit.txt').write_text('foo bar baz\n', encoding='utf-8')
    dispatch('read_file', {'path': 'nf_edit.txt'})
    res = dispatch('edit_file', {
        'path': 'nf_edit.txt',
        'search': 'XYZNOTHERE',
        'replace': 'REPLACED',
    })
    check(
        res.status == 'error' and res.code == 'not-found',
        f'edit_file not-found expected, got status={res.status!r} code={res.code!r}',
    )

    # ---- edit_file: not-unique AND file byte-for-byte unchanged -----------
    (root / 'dup_edit.txt').write_text('aa\nbb\naa\ncc\n', encoding='utf-8')
    dispatch('read_file', {'path': 'dup_edit.txt'})
    before = (root / 'dup_edit.txt').read_bytes()
    res = dispatch('edit_file', {
        'path': 'dup_edit.txt',
        'search': 'aa',
        'replace': 'AA',
    })
    check(
        res.status == 'error' and res.code == 'not-unique',
        f'edit_file not-unique expected, got status={res.status!r} code={res.code!r}',
    )
    after = (root / 'dup_edit.txt').read_bytes()
    check(
        before == after,
        f'not-unique edit mutated the file (before!=after)',
    )

    # ---- edit_file: not-read-yet on a file never opened -------------------
    (root / 'uredit.txt').write_text('abc\n', encoding='utf-8')
    res = dispatch('edit_file', {
        'path': 'uredit.txt',
        'search': 'abc',
        'replace': 'xyz',
    })
    check(
        res.status == 'error' and res.code == 'not-read-yet',
        f'edit_file unread expected not-read-yet, got status={res.status!r} code={res.code!r}',
    )

    # ---- freshness gate: file changed on disk (edit_file) ----------
    # Both edit_file and write_file funnel through tools._edit.freshness_gate,
    # which compares the persisted mtime+size against the current stat. Writing
    # an additional trailing line guarantees the size changes in lockstep with
    # mtime, so the check cannot be satisfied trivially on platforms whose
    # mtime resolution is coarser than one second.
    (root / 'stale_edit.txt').write_text('p\nq\nr\n', encoding='utf-8')
    dispatch('read_file', {'path': 'stale_edit.txt'})
    (root / 'stale_edit.txt').write_text('p\nq\nr\nEXTRA\n', encoding='utf-8')
    res = dispatch('edit_file', {
        'path': 'stale_edit.txt',
        'search': 'q',
        'replace': 'Q',
    })
    check(
        res.status == 'error' and res.code == 'file-changed-on-disk',
        f'stale edit_file expected file-changed-on-disk, '
        f'got status={res.status!r} code={res.code!r}',
    )
    stale_edit_actual = (root / 'stale_edit.txt').read_bytes()
    check(
        stale_edit_actual == b'p\nq\nr\nEXTRA\n',
        (
            'edit_file refused but still wrote: '
            f'expected b\'p\\nq\\nr\\nEXTRA\\n\', '
            f'got {stale_edit_actual!r}'
        ),
    )
    body_str = str(res.body)
    check(
        'your own last edit' in body_str and 'another process' in body_str,
        (
            f'stale edit_file message did not name both causes: body={body_str!r}; '
            f'gate cannot distinguish a model re-write from an external process, '
            f'so singling out only one would send the model hunting a phantom cause '
            f'instead of simply re-reading.'
        ),
    )

    # ---- freshness gate: file changed on disk (write_file) -----------
    (root / 'stale_write.txt').write_text('aaa\n', encoding='utf-8')
    dispatch('read_file', {'path': 'stale_write.txt'})
    (root / 'stale_write.txt').write_text('aaa\nbbb\n', encoding='utf-8')
    res = dispatch('write_file', {
        'path': 'stale_write.txt',
        'contents': 'REPLACED\n',
    })
    check(
        res.status == 'error' and res.code == 'file-changed-on-disk',
        f'stale write_file expected file-changed-on-disk, '
        f'got status={res.status!r} code={res.code!r}',
    )
    stale_write_actual = (root / 'stale_write.txt').read_bytes()
    check(
        stale_write_actual == b'aaa\nbbb\n',
        (
            'write_file refused but still wrote: '
            f'expected b\'aaa\\nbbb\\n\', '
            f'got {stale_write_actual!r}'
        ),
    )


if __name__ == '__main__':
    sys.exit(main())
