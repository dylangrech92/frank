"""Dispatch-level check for run_command's failure-render cwd grounding (no LLM).

Drives the real production hot path -- ``tools.registry.dispatch('run_command', ...)``
-- from a throwaway project dir, then renders each ToolResult exactly as the model
sees it via ``agent.render_tool_result``, so the grounding line the tool appends on a
nonzero exit is exercised end to end with zero mocks. The recorded live failure that
motivated this check was a model prefixing commands with a hallucinated ``cd`` and
never learning the concrete working directory; the grounding line puts that fact at
the failure moment. Asserts:

a. a command that exits NONZERO (``exit 3``) -> ok result whose rendered text
   contains the exit code AND the absolute working directory (the real fixture
   root) AND the grounding marker;
b. a command that exits ZERO (``echo hi``) -> rendered text does NOT carry the
   grounding line (no token tax on success);
c. the tool's description states commands already run with the project root as the
   working directory and that a ``cd`` into the project is never needed;
d. the timeout path (a command killed past a 1s timeout) still renders WITHOUT the
   grounding line -- the timeout message already explains what happened.

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

    # ---- d. timeout path -> no grounding line ----
    result, rendered = _dispatch_render(
        'python3 -c "import time; time.sleep(5)"', timeout=1
    )
    if result.status != 'error' or result.code != 'timeout':
        raise _Fail(f"timeout: expected error code='timeout', got {result.status!r} code={result.code!r}")
    if _GROUNDING_MARKER in rendered:
        raise _Fail(f"timeout: rendered result unexpectedly carries the grounding line: {rendered!r}")


def main() -> int:
    from tools.registry import activate, discover

    discover()
    # dispatch() refuses tools that are not PINNED or loaded via load_tool;
    # activate() is the exact production call load_tool.run() makes.
    activate('run_command')

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

    print("PASS: run_command grounds a nonzero exit with the exit code and absolute working directory, stays silent on success and timeout, and its description states commands already run at the project root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
