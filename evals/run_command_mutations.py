"""Dispatch-level check for run_command's snapshot-diff mutation events (no LLM).

Drives the real production hot path -- ``tools.registry.dispatch('run_command', ...)``
-- from a throwaway project dir, with a recorder subscribed to the shared mutation
bus (``tools._sandbox.subscribe_mutations``), so the foreground snapshot/diff that
publishes 'created'/'changed'/'deleted' events for files a shell command touches is
exercised end to end with zero mocks. The recorder is removed from
MUTATION_SUBSCRIBERS in the finally block so it never leaks into a later scenario
sharing the same process. Asserts:

a. a command that writes a NEW file -> result ok AND exactly one 'created' event
   whose absolute path ends with the new file's name;
b. a command that appends to an existing file -> exactly one 'changed' event;
c. a command that deletes an existing file (via python os.remove, clear of the
   deny-list) -> exactly one 'deleted' event;
d. a command that writes only into ``__pycache__/`` and a ``.hidden/`` dir ->
   ZERO events (both are pruned from the snapshot walk);
e. a pure read command -> ZERO events AND an ok result;
f. a command that writes a file then sleeps past a 1s timeout -> the timeout error
   result AND the 'created' event still fires (the diff runs on the timeout path).

Every recorded event dict is asserted to carry the expected 'kind' and an absolute
'path'. Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1
otherwise. Runs with the repo root on ``sys.path`` (evals/run.py inserts it) and
restores the cwd it changed.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


class _Fail(Exception):
    """Raised to abort the eval with a specific assertion message."""


def _dispatch(cmd: str, timeout: int | None = None):
    from tools.registry import dispatch

    args: dict[str, object] = {'cmd': cmd}
    if timeout is not None:
        args['timeout'] = timeout
    return dispatch('run_command', args)


def _assert_event_shape(event: dict, expected_kind: str) -> None:
    if event.get('kind') != expected_kind:
        raise _Fail(f"event has kind {event.get('kind')!r}, expected {expected_kind!r}")
    path = event.get('path')
    if not isinstance(path, str) or not os.path.isabs(path):
        raise _Fail(f"event path is not an absolute string: {path!r}")


def _run(project: Path, events: list[dict]) -> None:
    # ---- a. NEW file -> exactly one 'created' event ----
    events.clear()
    result = _dispatch('python3 -c "open(\'made.txt\', \'w\').write(\'x\')"')
    if result.status != 'success':
        raise _Fail(f"new-file: expected ok, got {result.status!r} code={result.code!r} body={result.body!r}")
    if len(events) != 1:
        raise _Fail(f"new-file: expected exactly one event, got {len(events)}: {events!r}")
    _assert_event_shape(events[0], 'created')
    if not events[0]['path'].endswith('made.txt'):
        raise _Fail(f"new-file: created path does not end with made.txt: {events[0]['path']!r}")

    # ---- b. append to an existing file -> one 'changed' event ----
    (project / 'appendable.txt').write_text('seed\n', encoding='utf-8')
    events.clear()
    result = _dispatch('python3 -c "open(\'appendable.txt\', \'a\').write(\'more\')"')
    if result.status != 'success':
        raise _Fail(f"append: expected ok, got {result.status!r} code={result.code!r} body={result.body!r}")
    if len(events) != 1:
        raise _Fail(f"append: expected exactly one event, got {len(events)}: {events!r}")
    _assert_event_shape(events[0], 'changed')
    if not events[0]['path'].endswith('appendable.txt'):
        raise _Fail(f"append: changed path does not end with appendable.txt: {events[0]['path']!r}")

    # ---- c. delete an existing file (os.remove, not rm) -> one 'deleted' event ----
    (project / 'todelete.txt').write_text('bye\n', encoding='utf-8')
    events.clear()
    result = _dispatch('python3 -c "import os; os.remove(\'todelete.txt\')"')
    if result.status != 'success':
        raise _Fail(f"delete: expected ok, got {result.status!r} code={result.code!r} body={result.body!r}")
    if len(events) != 1:
        raise _Fail(f"delete: expected exactly one event, got {len(events)}: {events!r}")
    _assert_event_shape(events[0], 'deleted')
    if not events[0]['path'].endswith('todelete.txt'):
        raise _Fail(f"delete: deleted path does not end with todelete.txt: {events[0]['path']!r}")

    # ---- d. writes only into pruned dirs -> ZERO events ----
    events.clear()
    result = _dispatch(
        'python3 -c "'
        'import os; '
        "os.makedirs('__pycache__', exist_ok=True); "
        "open('__pycache__/cached.txt', 'w').write('y'); "
        "os.makedirs('.hidden', exist_ok=True); "
        "open('.hidden/secret.txt', 'w').write('z')"
        '"'
    )
    if result.status != 'success':
        raise _Fail(f"pruned-dirs: expected ok, got {result.status!r} code={result.code!r} body={result.body!r}")
    if events:
        raise _Fail(f"pruned-dirs: expected zero events, got {events!r}")

    # ---- e. pure read command -> ZERO events, ok result ----
    (project / 'seed.txt').write_text('readable\n', encoding='utf-8')
    events.clear()
    result = _dispatch('python3 -c "print(open(\'seed.txt\').read())"')
    if result.status != 'success':
        raise _Fail(f"read: expected ok, got {result.status!r} code={result.code!r} body={result.body!r}")
    if events:
        raise _Fail(f"read: expected zero events, got {events!r}")

    # ---- f. timeout branch: write then sleep past the timeout -> timeout error + 'created' ----
    events.clear()
    result = _dispatch(
        'python3 -c "'
        "f = open('slowmade.txt', 'w'); f.write('x'); f.close(); "
        'import time; time.sleep(5)'
        '"',
        timeout=1,
    )
    if result.status != 'error' or result.code != 'timeout':
        raise _Fail(f"timeout: expected error code='timeout', got {result.status!r} code={result.code!r}")
    created = [e for e in events if e['kind'] == 'created' and e['path'].endswith('slowmade.txt')]
    if len(created) != 1:
        raise _Fail(f"timeout: expected one 'created' event for slowmade.txt, got {events!r}")
    _assert_event_shape(created[0], 'created')


def main() -> int:
    from tools import _sandbox
    from tools import registry

    # run_command is declared by 'code' mode (modes.py) — dispatch() refuses
    # any tool outside the active mode's fixed set.
    saved_mode = registry.current_mode()
    registry.activate_mode('code')

    events: list[dict] = []

    def _recorder(event: dict) -> None:
        events.append(event)

    _sandbox.subscribe_mutations(_recorder)

    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix='run_command_mutations_')
    try:
        os.chdir(tmp)
        _run(Path(tmp), events)
    except _Fail as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"FAIL: unexpected error: {exc}", file=sys.stderr)
        return 1
    finally:
        # Remove the recorder so it never leaks into a later scenario in the
        # same process (MUTATION_SUBSCRIBERS is a plain module-level list).
        if _recorder in _sandbox.MUTATION_SUBSCRIBERS:
            _sandbox.MUTATION_SUBSCRIBERS.remove(_recorder)
        os.chdir(prev_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    print("PASS: run_command publishes snapshot-diff mutation events for created/changed/deleted files, prunes dotdirs and caches, stays silent on reads, and fires on the timeout path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
