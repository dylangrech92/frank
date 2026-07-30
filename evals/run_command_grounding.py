"""Dispatch-level check for run_command's failure-render cwd grounding and the
mutation note it appends when a command touches project files (no LLM).

Drives the real production hot path -- ``tools.registry.dispatch('run_command', ...)``
-- from a throwaway project dir, then renders each ToolResult exactly as the model
sees it via ``agent.render_tool_result``, so both the grounding line the tool appends
on a nonzero exit and the mutation note it appends when a command writes to disk are
exercised end to end with zero mocks. Two recorded live failures motivated these
checks: a model prefixing commands with a hallucinated ``cd`` and never learning the
concrete working directory (the grounding line puts that fact at the failure moment),
and a model running a README write-command that permanently mutated a production data
file without ever noticing (the mutation note names the touched files at that moment).
Asserts:

a. a command that exits NONZERO (``exit 3``) -> ok result whose rendered text
   contains the exit code AND the absolute working directory (the real fixture
   root) AND the grounding marker;
b. a command that exits ZERO (``echo hi``) -> rendered text does NOT carry the
   grounding line (no token tax on success);
c. the tool's description states commands already run with the project root as the
   working directory and that a ``cd`` into the project is never needed;
d. the timeout path (a command killed past a 1s timeout) still renders WITHOUT the
   grounding line -- the timeout message already explains what happened;
e. a command that writes a file (``echo x > side.txt``) -> rendered text carries the
   mutation note naming side.txt as created;
f. a pure command (``echo hi``) -> rendered text carries NO mutation note;
g. a nonzero-exit command that also writes carries BOTH the mutation note (naming the
   written file) AND the unchanged cwd grounding line, with the mutation note first;
h. a deletion (``rm`` of an existing fixture file) -> rendered text names the file as
   deleted in the mutation note;
i. the tool's description states that when a command creates, changes, or deletes
   project files the result names them.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise. Runs
with the repo root on ``sys.path`` (evals/run.py inserts it) and restores the cwd it
changed.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from tools.result import ToolResult

# Marker unique to the grounding line (the description says "with the project
# root as the working directory"; the grounding line says "ran in working
# directory <abs>"), so an assertion on this phrase cannot be satisfied by the
# static description text bleeding into a render.
_GROUNDING_MARKER = 'ran in working directory'

# Marker unique to the mutation note the tool appends when a command creates,
# changes, or deletes project files.
_MUTATION_MARKER = 'this command modified project files'


class _Fail(Exception):
    """Raised to abort the eval with a specific assertion message."""


def _dispatch_render(cmd: str, timeout: int | None = None) -> tuple[ToolResult, str]:
    from agent import render_tool_result
    from tools.registry import dispatch

    args: dict[str, object] = {'cmd': cmd}
    if timeout is not None:
        args['timeout'] = timeout
    result = dispatch('run_command', args)
    return result, render_tool_result('run_command', result)


def _run(root_str: str) -> None:
    # ---- a. nonzero exit -> grounding line names exit code + absolute cwd ----
    result, rendered = _dispatch_render('exit 3')
    if result.status != 'success':
        raise _Fail(f"nonzero: expected ok result, got {result.status!r} code={result.code!r}")
    if 'exit code 3' not in rendered:
        raise _Fail(f"nonzero: rendered result is missing the exit code: {rendered!r}")
    if root_str not in rendered:
        raise _Fail(f"nonzero: rendered result is missing the absolute cwd {root_str!r}: {rendered!r}")
    if _GROUNDING_MARKER not in rendered:
        raise _Fail(f"nonzero: rendered result is missing the grounding marker: {rendered!r}")

    # ---- b. zero exit -> no grounding line (no token tax on success) ----
    result, rendered = _dispatch_render('echo hi')
    if result.status != 'success':
        raise _Fail(f"zero: expected ok result, got {result.status!r} code={result.code!r}")
    if _GROUNDING_MARKER in rendered:
        raise _Fail(f"zero: rendered result unexpectedly carries the grounding line: {rendered!r}")

    # ---- c. description states the already-runs-at-project-root fact ----
    from tools.run_command import RunCommand

    description = RunCommand.description
    if 'project root as the working directory' not in description:
        raise _Fail(f"description: missing the project-root working-directory sentence: {description!r}")
    if 'cd into the project is never needed' not in description:
        raise _Fail(f"description: missing the no-cd-needed clause: {description!r}")
    # ---- i. description states the result names files a command touches ----
    if 'creates, changes, or deletes project files' not in description:
        raise _Fail(f"description: missing the mutation-naming sentence: {description!r}")
    if 'the result names' not in description:
        raise _Fail(f"description: missing the result-names-them clause: {description!r}")

    # ---- d. timeout path -> no grounding line ----
    result, rendered = _dispatch_render(
        'python3 -c "import time; time.sleep(5)"', timeout=1
    )
    if result.status != 'error' or result.code != 'timeout':
        raise _Fail(f"timeout: expected error code='timeout', got {result.status!r} code={result.code!r}")
    if _GROUNDING_MARKER in rendered:
        raise _Fail(f"timeout: rendered result unexpectedly carries the grounding line: {rendered!r}")

    # ---- e. a side-effect write names the created file in the mutation note ----
    result, rendered = _dispatch_render('echo x > side.txt')
    if result.status != 'success':
        raise _Fail(f"write: expected ok result, got {result.status!r} code={result.code!r}")
    if _MUTATION_MARKER not in rendered:
        raise _Fail(f"write: rendered result is missing the mutation note: {rendered!r}")
    if 'created:' not in rendered or 'side.txt' not in rendered:
        raise _Fail(f"write: mutation note does not name side.txt as created: {rendered!r}")

    # ---- f. a pure command carries NO mutation note (no token tax when nothing changed) ----
    result, rendered = _dispatch_render('echo hi')
    if result.status != 'success':
        raise _Fail(f"pure: expected ok result, got {result.status!r} code={result.code!r}")
    if _MUTATION_MARKER in rendered:
        raise _Fail(f"pure: rendered result unexpectedly carries a mutation note: {rendered!r}")

    # ---- g. a nonzero-exit command that also writes carries BOTH notes, mutation first ----
    result, rendered = _dispatch_render('echo x > side_fail.txt; exit 3')
    if result.status != 'success':
        raise _Fail(f"write+fail: expected ok result, got {result.status!r} code={result.code!r}")
    if _MUTATION_MARKER not in rendered or 'side_fail.txt' not in rendered:
        raise _Fail(f"write+fail: mutation note does not name side_fail.txt: {rendered!r}")
    if _GROUNDING_MARKER not in rendered or 'exit code 3' not in rendered:
        raise _Fail(f"write+fail: rendered result is missing the cwd grounding line: {rendered!r}")
    if rendered.index(_MUTATION_MARKER) > rendered.index(_GROUNDING_MARKER):
        raise _Fail(f"write+fail: mutation note must render before the grounding line: {rendered!r}")

    # ---- h. a deletion is named as deleted in the mutation note ----
    Path('todelete.txt').write_text('bye\n', encoding='utf-8')
    result, rendered = _dispatch_render('rm todelete.txt')
    if result.status != 'success':
        raise _Fail(f"delete: expected ok result, got {result.status!r} code={result.code!r}")
    if _MUTATION_MARKER not in rendered or 'deleted:' not in rendered or 'todelete.txt' not in rendered:
        raise _Fail(f"delete: mutation note does not name todelete.txt as deleted: {rendered!r}")


def main() -> int:
    from tools import registry

    # run_command is declared by 'code' mode (modes.py) — dispatch() refuses
    # any tool outside the active mode's fixed set.
    saved_mode = registry.current_mode()
    registry.activate_mode('code')

    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix='run_command_grounding_')
    try:
        os.chdir(tmp)
        # Capture the working directory as the tool itself sees it (os.getcwd()
        # canonicalizes symlinks the way Path.cwd() does inside run()), so the
        # asserted absolute path matches the grounding line byte for byte.
        _run(str(Path.cwd()))
    except _Fail as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"FAIL: unexpected error: {exc}", file=sys.stderr)
        return 1
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    print("PASS: run_command grounds a nonzero exit with the exit code and absolute working directory, stays silent on success and timeout, names files it creates/changes/deletes in a mutation note (and stays silent when nothing changed), and its description states both facts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
