"""Dispatch-level contract eval for the four profiling tools (no LLM).

Drives the real production hot path — ``tools.registry.dispatch`` — exactly as
``handle_user_message`` does per tool call, exercising the mode gate, argument
validation, real subprocess execution, and the shared ``tools._snapshot``
mutation-truthfulness machinery together, not any tool class in isolation.
Zero mocks: every snippet below is written to a real OS temp file (or passed
inline) and executed by a real ``sys.executable`` / ``node`` / ``php``
subprocess against the real project root as cwd, exactly as a live model turn
would.

Covers, for ``profile_command``, ``profile_hotspots``, ``profile_memory``, and
``trace_execution``:

    1. the ``not-in-mode`` gate rejects a call before any mode is active
       (run before ``registry.activate_mode()`` is ever called this process);
    2. ``profile_command``'s happy path, deny-list, and timeout contracts;
    3. ``profile_hotspots`` python: exact ``ncalls`` for a hot function;
    4. ``profile_memory`` python: a ``Peak:`` summary and a per-site table;
    5. ``trace_execution``: exact call counts and max recursion stack depth;
    6. mutation truthfulness — a file a profiled command writes into the repo
       root is both named in the result body and published on the shared
       mutation bus (``tools._sandbox.subscribe_mutations`` — the module that
       actually owns the subscriber list; ``tools._snapshot`` re-exports only
       ``emit_mutation``, not the subscribe API);
    7. no ``perf_profile_<pid>_*`` artifact-dir leak into the OS temp dir —
       scoped to this process, since the temp dir is shared machine-wide;
    8. documented contract error codes: ``profile-result-missing`` for a
       python target that skips the bootstrap's ``finally`` via
       ``os._exit(0)``, ``is_module`` passthrough (module name, not a
       file-resolved target), and ``bad-arguments`` for missing/conflicting
       target+snippet;
    9. node hotspot + memory profiling (skipped visibly if ``node`` is absent);
    10. php hotspot profiling, xdebug-aware (skipped visibly if ``php`` is
        absent).

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before
exec'ing this file), and chdir's to the repo root itself before dispatching
so the cwd invariant (commands always run with the project root as cwd) is
reproduced explicitly rather than depending on whichever directory happened
to invoke this script directly.
"""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure the repo root is on sys.path so top-level imports (tools, runtime)
# resolve when this script is invoked directly. evals/run.py already sets
# PYTHONPATH for inline scenarios, but we also add it here (mirrors
# evals/activate_tools_wiring.py) so `.venv/bin/python evals/profile_tools_contract.py`
# works standalone from the repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# sys.executable, shell-quoted once up front — every fixture command below
# invokes this same interpreter.
PY = shlex.quote(sys.executable)


def _dispatch(name: str, arguments: dict):
    from tools.registry import dispatch

    return dispatch(name, arguments)


def _body_text(result) -> str:
    return result.body if isinstance(result.body, str) else str(result.body)


# ---------------------------------------------------------------------------
# Fixture sources
# ---------------------------------------------------------------------------

_PY_HOTSPOTS_SNIPPET = (
    "def hot():\n"
    "    return 1\n"
    "\n"
    "def main():\n"
    "    for _ in range(300):\n"
    "        hot()\n"
    "\n"
    "main()\n"
)

# 50000 DISTINCT strings — 'x' * 100 would be constant-folded to one shared
# object and never show up as 50000 allocations under tracemalloc.
_PY_MEMORY_SNIPPET = "data = [('x%0100d' % i) for i in range(50000)]\n"

_TRACE_SNIPPET = (
    "def hot():\n"
    "    return 1\n"
    "\n"
    "def rec(n):\n"
    "    if n <= 0:\n"
    "        return 0\n"
    "    return 1 + rec(n - 1)\n"
    "\n"
    "def main():\n"
    "    for _ in range(1234):\n"
    "        hot()\n"
    "    rec(15)\n"
    "\n"
    "main()\n"
)

_NODE_HOTSPOTS_SNIPPET = (
    "function hot() {\n"
    "  let s = 0;\n"
    "  for (let i = 0; i < 3000000; i++) { s += i; }\n"
    "  return s;\n"
    "}\n"
    "function main() {\n"
    "  for (let i = 0; i < 20; i++) { hot(); }\n"
    "}\n"
    "main();\n"
)

_NODE_MEMORY_SNIPPET = (
    "let data = [];\n"
    "for (let i = 0; i < 200000; i++) {\n"
    "  data.push({ i: i, s: 'x'.repeat(20) });\n"
    "}\n"
    "console.log(data.length);\n"
)

_PHP_HOTSPOTS_SNIPPET = (
    "<?php\n"
    "function hot() {\n"
    "    return 1;\n"
    "}\n"
    "function main() {\n"
    "    for ($i = 0; $i < 300; $i++) {\n"
    "        hot();\n"
    "    }\n"
    "}\n"
    "main();\n"
)


# ---------------------------------------------------------------------------
# 1. not-in-mode gate
# ---------------------------------------------------------------------------


def check_not_in_mode_gate() -> list[str]:
    """Before any mode is active, profile_command must be rejected with code='not-in-mode'."""
    failures: list[str] = []
    result = _dispatch('profile_command', {'cmd': f'{PY} -c "print(1)"'})
    if result.status != 'error':
        failures.append(f"not-in-mode gate: expected status='error', got {result.status!r}")
    if result.code != 'not-in-mode':
        failures.append(f"not-in-mode gate: expected code='not-in-mode', got {result.code!r}")
    return failures


# ---------------------------------------------------------------------------
# 2. profile_command
# ---------------------------------------------------------------------------


def check_profile_command_happy() -> list[str]:
    failures: list[str] = []
    result = _dispatch('profile_command', {'cmd': f'{PY} -c "print(42)"'})
    if result.status != 'success':
        failures.append(
            f"profile_command happy: expected success, got {result.status!r} "
            f"code={result.code!r} body={result.body!r}"
        )
        return failures
    body = _body_text(result).lower()
    for expected in ('wall', 'cpu', 'peak rss'):
        if expected not in body:
            failures.append(f"profile_command happy: body missing {expected!r}: {body!r}")
    return failures


def check_profile_command_deny_list() -> list[str]:
    failures: list[str] = []
    result = _dispatch('profile_command', {'cmd': 'rm -rf /'})
    if result.status != 'error':
        failures.append(f"deny-list: expected status='error', got {result.status!r}")
    if result.code != 'destructive-command-blocked':
        failures.append(f"deny-list: expected code='destructive-command-blocked', got {result.code!r}")
    return failures


def check_profile_command_timeout() -> list[str]:
    failures: list[str] = []
    result = _dispatch(
        'profile_command',
        {'cmd': f'{PY} -c "import time; time.sleep(30)"', 'timeout': 1},
    )
    if result.status != 'error':
        failures.append(f"timeout: expected status='error', got {result.status!r}")
    if result.code != 'timeout':
        failures.append(f"timeout: expected code='timeout', got {result.code!r}")
    return failures


# ---------------------------------------------------------------------------
# 3. profile_hotspots (python)
# ---------------------------------------------------------------------------


def check_profile_hotspots_python() -> list[str]:
    failures: list[str] = []
    result = _dispatch('profile_hotspots', {'language': 'python', 'snippet': _PY_HOTSPOTS_SNIPPET})
    if result.status != 'success':
        failures.append(
            f"profile_hotspots python: expected success, got {result.status!r} "
            f"code={result.code!r} body={result.body!r}"
        )
        return failures
    body = _body_text(result)
    if 'hot' not in body:
        failures.append(f"profile_hotspots python: body missing 'hot': {body!r}")
    if '300' not in body:
        failures.append(f"profile_hotspots python: body missing '300': {body!r}")
    return failures


# ---------------------------------------------------------------------------
# 4. profile_memory (python)
# ---------------------------------------------------------------------------


def check_profile_memory_python() -> list[str]:
    failures: list[str] = []
    result = _dispatch('profile_memory', {'language': 'python', 'snippet': _PY_MEMORY_SNIPPET})
    if result.status != 'success':
        failures.append(
            f"profile_memory python: expected success, got {result.status!r} "
            f"code={result.code!r} body={result.body!r}"
        )
        return failures
    body = _body_text(result)
    if 'Peak:' not in body:
        failures.append(f"profile_memory python: body missing 'Peak:': {body!r}")
    if 'site' not in body:
        failures.append(f"profile_memory python: body missing a 'site' column: {body!r}")
    return failures


# ---------------------------------------------------------------------------
# 5. trace_execution
# ---------------------------------------------------------------------------


def check_trace_execution() -> list[str]:
    failures: list[str] = []
    result = _dispatch('trace_execution', {'snippet': _TRACE_SNIPPET})
    if result.status != 'success':
        failures.append(
            f"trace_execution: expected success, got {result.status!r} "
            f"code={result.code!r} body={result.body!r}"
        )
        return failures
    body = _body_text(result)
    if '1234' not in body:
        failures.append(f"trace_execution: body missing '1234' call count: {body!r}")
    if 'max stack depth' not in body:
        failures.append(f"trace_execution: body missing 'max stack depth': {body!r}")
    return failures


# ---------------------------------------------------------------------------
# 6. mutation truthfulness
# ---------------------------------------------------------------------------


def check_mutation_truthfulness() -> list[str]:
    from tools import _sandbox

    failures: list[str] = []
    events: list[dict] = []

    def _recorder(event: dict) -> None:
        events.append(event)

    _sandbox.subscribe_mutations(_recorder)
    target = REPO_ROOT / 'perf_contract_out.txt'
    try:
        if target.exists():
            target.unlink()
        cmd = f'{PY} -c "open(\'perf_contract_out.txt\', \'w\').write(\'x\')"'
        result = _dispatch('profile_command', {'cmd': cmd})
        if result.status != 'success':
            failures.append(
                f"mutation truthfulness: expected success, got {result.status!r} "
                f"code={result.code!r} body={result.body!r}"
            )
            return failures
        body = _body_text(result)
        if 'perf_contract_out.txt' not in body:
            failures.append(
                f"mutation truthfulness: body does not name perf_contract_out.txt: {body!r}"
            )
        created = [
            e for e in events
            if e.get('kind') == 'created' and str(e.get('path', '')).endswith('perf_contract_out.txt')
        ]
        if not created:
            failures.append(
                f"mutation truthfulness: no 'created' mutation event for perf_contract_out.txt, "
                f"recorded events: {events!r}"
            )
    finally:
        if _recorder in _sandbox.MUTATION_SUBSCRIBERS:
            _sandbox.MUTATION_SUBSCRIBERS.remove(_recorder)
        if target.exists():
            target.unlink()
    return failures


# ---------------------------------------------------------------------------
# 7. tempdir-leak
# ---------------------------------------------------------------------------


def check_tempdir_leak() -> list[str]:
    """Assert this process left no artifact directory behind.

    Scoped to our own pid.  The OS temp dir is shared by every process on the
    machine, so an unscoped ``perf_profile_*`` glob also matches the live
    directory of any concurrent profiling run — which made this check fail for
    someone else's work in progress rather than for a real leak.  ``_dispatch``
    runs the tools in this process, so their artifact directories carry this
    pid and nothing else does.
    """
    failures: list[str] = []
    tmp_dir = Path(tempfile.gettempdir())
    leaked = sorted(glob.glob(str(tmp_dir / f'perf_profile_{os.getpid()}_*')))
    if leaked:
        failures.append(f"tempdir leak: perf_profile_* entries remained: {leaked}")
    return failures


# ---------------------------------------------------------------------------
# 8. contract error codes
# ---------------------------------------------------------------------------


def check_profile_memory_os_exit() -> list[str]:
    """os._exit(0) skips the bootstrap's finally -> no result JSON is ever written."""
    failures: list[str] = []
    result = _dispatch('profile_memory', {'language': 'python', 'snippet': 'import os; os._exit(0)'})
    if result.status != 'error':
        failures.append(f"profile_memory os._exit: expected status='error', got {result.status!r}")
    if result.code != 'profile-result-missing':
        failures.append(
            f"profile_memory os._exit: expected code='profile-result-missing', got {result.code!r}"
        )
    return failures


def check_is_module_passthrough() -> list[str]:
    """target='this' + is_module=True is passed through verbatim as a module name
    (never file-resolved) — runs `python -m this`."""
    failures: list[str] = []
    result = _dispatch(
        'profile_hotspots',
        {'language': 'python', 'target': 'this', 'is_module': True},
    )
    if result.status != 'success':
        failures.append(
            f"is_module passthrough: expected success, got {result.status!r} "
            f"code={result.code!r} body={result.body!r}"
        )
    return failures


def check_profile_memory_bad_arguments() -> list[str]:
    failures: list[str] = []

    neither = _dispatch('profile_memory', {'language': 'python'})
    if neither.status != 'error' or neither.code != 'bad-arguments':
        failures.append(
            f"profile_memory neither target nor snippet: expected error code='bad-arguments', "
            f"got status={neither.status!r} code={neither.code!r}"
        )

    both = _dispatch(
        'profile_memory',
        {'language': 'python', 'target': 'a.py', 'snippet': 'x'},
    )
    if both.status != 'error' or both.code != 'bad-arguments':
        failures.append(
            f"profile_memory both target and snippet: expected error code='bad-arguments', "
            f"got status={both.status!r} code={both.code!r}"
        )
    return failures


# ---------------------------------------------------------------------------
# 9. node
# ---------------------------------------------------------------------------


def check_node_profiling() -> list[str]:
    failures: list[str] = []
    if shutil.which('node') is None:
        print("SKIP node (binary absent)")
        return failures

    hotspots = _dispatch('profile_hotspots', {'language': 'node', 'snippet': _NODE_HOTSPOTS_SNIPPET})
    if hotspots.status != 'success':
        failures.append(
            f"node profile_hotspots: expected success, got {hotspots.status!r} "
            f"code={hotspots.code!r} body={hotspots.body!r}"
        )

    memory = _dispatch('profile_memory', {'language': 'node', 'snippet': _NODE_MEMORY_SNIPPET})
    if memory.status != 'success':
        failures.append(
            f"node profile_memory: expected success, got {memory.status!r} "
            f"code={memory.code!r} body={memory.body!r}"
        )
    else:
        body = _body_text(memory)
        if 'self_bytes' not in body:
            failures.append(f"node profile_memory: body missing 'self_bytes': {body!r}")
    return failures


# ---------------------------------------------------------------------------
# 10. php
# ---------------------------------------------------------------------------


def check_php_profiling() -> list[str]:
    failures: list[str] = []
    if shutil.which('php') is None:
        print("SKIP php (binary absent)")
        return failures

    from runtime.profiling import php_xdebug_status, which_interpreter

    php_bin = which_interpreter('php')
    status = php_xdebug_status(php_bin) if php_bin else 'missing'

    result = _dispatch('profile_hotspots', {'language': 'php', 'snippet': _PHP_HOTSPOTS_SNIPPET})

    if status == 'missing':
        if result.status != 'error':
            failures.append(f"php xdebug-unavailable: expected status='error', got {result.status!r}")
        if result.code != 'xdebug-unavailable':
            failures.append(
                f"php xdebug-unavailable: expected code='xdebug-unavailable', got {result.code!r}"
            )
    else:
        if result.status != 'success':
            failures.append(
                f"php profile_hotspots: expected success, got {result.status!r} "
                f"code={result.code!r} body={result.body!r}"
            )
        else:
            body = _body_text(result)
            if 'calls' not in body and 'Time' not in body:
                failures.append(
                    f"php profile_hotspots: body missing a 'calls'/'Time' cachegrind column: {body!r}"
                )
    return failures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    # main.py's real process is always launched with cwd already at the
    # project root (see evals/run.py: cwd=project_dir for live scenarios,
    # cwd=REPO_ROOT for inline ones) — every profiling tool derives its
    # "project root" from Path.cwd() at call time. Reproduce that invariant
    # explicitly here instead of depending on whatever directory happened to
    # invoke this script directly.
    os.chdir(REPO_ROOT)

    from tools import registry

    all_failures: list[str] = []

    def _run(description: str, check: Callable[[], list[str]]) -> None:
        failures = check()
        if failures:
            print(f"FAIL: {description}", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
            all_failures.extend(failures)
        else:
            print(f"PASS: {description}")

    # 1. not-in-mode gate (no mode active yet), then activate 'performance_debug'
    # — the mode that declares all four profiling tools (modes.py).
    _run("not-in-mode gate rejects profile_command before any mode is active", check_not_in_mode_gate)
    registry.activate_mode('performance_debug')

    # 2. profile_command
    _run("profile_command happy path reports wall/cpu/peak rss", check_profile_command_happy)
    _run("profile_command deny-list blocks a destructive command", check_profile_command_deny_list)
    _run("profile_command timeout reports code='timeout'", check_profile_command_timeout)

    # 3. profile_hotspots (python)
    _run("profile_hotspots python reports hot()'s exact 300 calls", check_profile_hotspots_python)

    # 4. profile_memory (python)
    _run("profile_memory python reports Peak: + a site table", check_profile_memory_python)

    # 5. trace_execution
    _run("trace_execution reports 1234 calls + max stack depth", check_trace_execution)

    # 6. mutation truthfulness
    _run(
        "mutation truthfulness: profile_command names + publishes the file it wrote",
        check_mutation_truthfulness,
    )

    # 7. tempdir-leak (checks 1-6 so far)
    _run("no perf_profile_<pid>_* tempdir leak (checks 1-6)", check_tempdir_leak)

    # 8. contract error codes
    _run("profile_memory os._exit(0) -> profile-result-missing", check_profile_memory_os_exit)
    _run("profile_hotspots is_module passthrough runs `python -m this`", check_is_module_passthrough)
    _run(
        "profile_memory bad-arguments (missing / conflicting target+snippet)",
        check_profile_memory_bad_arguments,
    )

    # 9. node
    _run("node profile_hotspots + profile_memory", check_node_profiling)

    # 10. php
    _run("php profile_hotspots (xdebug-aware)", check_php_profiling)

    # Final leak check covering every dispatch in the whole run, not just 1-6.
    _run("no perf_profile_<pid>_* tempdir leak (full run)", check_tempdir_leak)

    if all_failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
