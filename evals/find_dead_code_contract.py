"""Dispatch-level contract check for the find_dead_code tool (no LLM).

Drives the real production hot path — ``tools.registry.dispatch`` — exactly as a
live model turn does per tool call, so this exercises arg validation, the
not-loaded/PINNED gate, and ``FindDeadCode.run`` together against a real vulture
subprocess. Zero mocks: fixture files are written to a real temp project dir and
scanned by the real vulture binary, with cwd chdir'd into that dir (the tool
resolves paths against ``Path.cwd()``). Asserts:

a. a .py file with a genuinely dead module-level function -> result ok, body
   carries a ``path:line`` locator and ``% confidence``, structured ``count`` >= 1,
   and ``adapter == 'vulture'``;
b. a clean .py file (every symbol used) -> ok with structured ``count`` == 0;
c. a nonexistent path -> err code='not-found';
d. a directory holding only a non-Python file -> err code='no-adapter'.

If vulture is not on PATH the eval prints a clear SKIP and exits 0 (it must not
hard-fail on a machine without the engine installed).

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it) and restores
the cwd it changed.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


class _Fail(Exception):
    """Raised to abort the eval with a specific assertion message."""


# A module-level function that nothing else references — vulture flags it as an
# unused function with high confidence.
_DEAD_PY = '''"""Fixture with a genuinely unused module-level function."""


def used_helper(x):
    return x + 1


def orphaned_unused_function(y):
    return y * 2


print(used_helper(41))
'''

# Every symbol here is reached: the function is called at module scope, so
# vulture reports nothing.
_CLEAN_PY = '''"""Fixture where every symbol is used."""


def greet(name):
    return "hi " + name


print(greet("world"))
'''


def _dispatch(rel_path: str | None):
    from tools.registry import dispatch

    args: dict[str, object] = {}
    if rel_path is not None:
        args['path'] = rel_path
    return dispatch('find_dead_code', args)


def _run(project: Path) -> None:
    (project / 'dead.py').write_text(_DEAD_PY, encoding='utf-8')
    (project / 'clean.py').write_text(_CLEAN_PY, encoding='utf-8')
    txt_only = project / 'txtonly'
    txt_only.mkdir()
    (txt_only / 'notes.txt').write_text('just some prose, no code here\n', encoding='utf-8')

    # ---- a. a file with a genuinely dead function ----
    result = _dispatch('dead.py')
    if result.status != 'success':
        raise _Fail(f"dead-function file: expected ok, got {result.status!r} code={result.code!r} body={result.body!r}")
    body = result.body if isinstance(result.body, str) else str(result.body)
    if 'dead.py:' not in body:
        raise _Fail(f"dead-function file: body carries no 'path:line' locator: {body!r}")
    if '% confidence' not in body:
        raise _Fail(f"dead-function file: body carries no '% confidence': {body!r}")
    if result.meta.get('adapter') != 'vulture':
        raise _Fail(f"dead-function file: expected adapter='vulture', got {result.meta.get('adapter')!r}")
    count = result.meta.get('count')
    if not isinstance(count, int) or count < 1:
        raise _Fail(f"dead-function file: expected structured count >= 1, got {count!r}")

    # ---- b. a clean file: ok, count == 0 ----
    result = _dispatch('clean.py')
    if result.status != 'success':
        raise _Fail(f"clean file: expected ok, got {result.status!r} code={result.code!r} body={result.body!r}")
    if result.meta.get('count') != 0:
        raise _Fail(f"clean file: expected count == 0, got {result.meta.get('count')!r} (body={result.body!r})")

    # ---- c. nonexistent path -> not-found ----
    result = _dispatch('does_not_exist.py')
    if result.status != 'error':
        raise _Fail(f"nonexistent path: expected error, got {result.status!r}")
    if result.code != 'not-found':
        raise _Fail(f"nonexistent path: expected code='not-found', got {result.code!r}")

    # ---- d. directory with only a .txt file -> no-adapter ----
    result = _dispatch('txtonly')
    if result.status != 'error':
        raise _Fail(f"non-Python dir: expected error, got {result.status!r} body={result.body!r}")
    if result.code != 'no-adapter':
        raise _Fail(f"non-Python dir: expected code='no-adapter', got {result.code!r}")


def main() -> int:
    if shutil.which('vulture') is None:
        print('SKIP: vulture is not installed on PATH; find_dead_code contract not exercised')
        return 0

    from tools.registry import activate, discover

    discover()
    # dispatch() refuses tools that are not PINNED or loaded via load_tool;
    # activate() is the exact production call load_tool.run() makes.
    activate('find_dead_code')

    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix='find_dead_code_contract_')
    try:
        os.chdir(tmp)
        _run(Path(tmp))
    except _Fail as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"FAIL: unexpected error: {exc}", file=sys.stderr)
        return 1
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    print("PASS: find_dead_code flags dead Python via vulture, reports clean files, and errors on missing/non-Python targets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
