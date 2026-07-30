"""Spawn-agents tool: fan out independent subtasks to concurrent one-shot children.

Each child is a fresh one-shot invocation of this same harness
(``main.py -p -``) running in its own process. Children have no
access to the parent's context or transcript — every spec's
``prompt`` must be fully self-contained.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult

# Recursion guard: refuse to spawn once *this* invocation is already running
# at depth >= 2 (i.e. it is itself a grandchild) — caps fan-out at two levels.
_MAX_DEPTH = 2

_DEFAULT_MAX_CONCURRENT = 4
_DEFAULT_TIMEOUT_S = 3600

# Per-child answer cap and stderr-tail line count, kept consistent with the
# truncation conventions used elsewhere (web_read's max_chars, run_command's
# oversize hint) so one runaway child can't blow out the parent's context.
_MAX_ANSWER_CHARS = 20_000
_STDERR_TAIL_LINES = 15


class SpawnAgents(Tool):
    """Fan out independent subtasks to concurrent one-shot children.

    Use this to delegate independent read/research questions (e.g. "what does
    file A do", "what does file B do") to keep your own context clean, or to
    run genuinely independent subtasks concurrently instead of doing them one
    at a time. Each spec becomes a brand-new child process of this same
    harness with **no access to this conversation's history and no shared
    context** — it only ever sees the single ``prompt`` string you give it, so
    every prompt must be fully self-contained (state the question, any needed
    background, and what form the answer should take; do not say "as discussed
    above" or reference anything the child cannot see). Children may read and
    edit files, so do not use this for two subtasks that touch the same file
    concurrently. Not for trivial single-step lookups you could just do
    yourself with a normal tool call.
    """

    name = 'spawn_agents'
    description = (
        'Fan out independent subtasks to concurrent one-shot subagent '
        'children of this same harness. Use this to delegate independent read/research '
        'questions (keeping your own context clean) or to run genuinely independent '
        'subtasks concurrently rather than sequentially. Each child is a brand-new '
        'process with no access to this conversation\'s history and no shared context — '
        'it only ever sees the exact `prompt` string in its spec, so every prompt must '
        'be fully self-contained (state the question, any needed background, and the '
        'expected answer shape). Children run up to a configured concurrency limit '
        'and each has a bounded timeout. Not `parallel_safe` — children may mutate '
        'files. Refuses to spawn when already running as a subagent two levels deep '
        '(no unbounded recursive fan-out).'
    )
    action = 'fan out subagents'
    oversize_hint = 'reduce the number of specs or ask children for shorter answers'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'specs': {
                'type': 'array',
                'description': (
                    'One entry per subagent to spawn. Each is an object with a '
                    'required `prompt` (fully self-contained task text — the child '
                    'has no other context) and an optional `cwd` (working directory '
                    'for that child; defaults to the current project root).'
                ),
                'items': {
                    'type': 'object',
                    'properties': {
                        'prompt': {
                            'type': 'string',
                            'description': (
                                'The complete, self-contained task/question for this '
                                'child. The child sees only this text.'
                            ),
                        },
                        'cwd': {
                            'type': 'string',
                            'description': (
                                'Working directory for this child. Defaults to the '
                                'current project root when omitted.'
                            ),
                        },
                    },
                    'required': ['prompt'],
                },
            },
        },
        'required': ['specs'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the spawn_agents tool.

        Args:
            specs: List of ``{prompt, cwd?}`` objects (required, non-empty).

        Returns:
            A ``ToolResult`` rendering each child's outcome (prompt summary,
            exit code, full answer, stderr tail on failure), or an error when
            the recursion guard trips or arguments are malformed.
        """
        specs_raw = kwargs.get('specs')
        if not isinstance(specs_raw, list) or not specs_raw:
            return ToolResult.err(
                "'specs' must be a non-empty array of {prompt, cwd?} objects",
                code='bad-arguments',
            )

        parsed_specs: list[dict[str, str]] = []
        for i, item in enumerate(specs_raw):
            if not isinstance(item, dict):
                return ToolResult.err(
                    f'specs[{i}] must be an object with a `prompt` string',
                    code='bad-arguments',
                )
            prompt = item.get('prompt')
            if not isinstance(prompt, str) or not prompt.strip():
                return ToolResult.err(
                    f"specs[{i}].prompt must be a non-empty string",
                    code='bad-arguments',
                )
            cwd = item.get('cwd')
            if cwd is not None and not isinstance(cwd, str):
                return ToolResult.err(
                    f'specs[{i}].cwd must be a string when present',
                    code='bad-arguments',
                )
            parsed_specs.append({'prompt': prompt.strip(), 'cwd': cwd or os.getcwd()})

        depth = _current_depth()
        if depth >= _MAX_DEPTH:
            return ToolResult.err(
                f'refusing to spawn subagents at recursion depth {depth} '
                f'(max is {_MAX_DEPTH - 1}) — this agent is itself a spawned '
                'subagent and its children would be too deep',
                code='recursion-limit',
                hint='handle this subtask directly instead of calling spawn_agents again',
            )

        from tools import registry  # pylint: disable=import-outside-toplevel

        mode = registry.current_mode()
        if mode is None:
            raise RuntimeError(
                'spawn_agents: no mode is active in this process — refusing to '
                'spawn a modeless child'
            )

        max_concurrent, timeout_s = _subagent_config()
        install_dir = Path(__file__).resolve().parent.parent
        main_py = install_dir / 'main.py'

        child_env = dict(os.environ)
        child_env['CODING_AGENT_DEPTH'] = str(depth + 1)

        results: list[dict[str, Any]] = [{} for _ in parsed_specs]

        def _run_child(idx: int, spec: dict[str, str]) -> tuple[int, dict[str, Any]]:
            return idx, _spawn_one(main_py, spec, child_env, timeout_s, mode)

        workers = max(1, min(max_concurrent, len(parsed_specs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_child, i, spec) for i, spec in enumerate(parsed_specs)]
            for future in futures:
                idx, outcome = future.result()
                results[idx] = outcome

        body = _render_results(parsed_specs, results, timeout_s)
        failures = sum(1 for r in results if r['exit_code'] != 0 or r['timed_out'])
        return ToolResult.ok(
            body,
            children=len(parsed_specs),
            failures=failures,
        )


def _current_depth() -> int:
    """Return the current subagent-recursion depth from ``CODING_AGENT_DEPTH``.

    Defaults to 0 (top-level) when unset or unparsable.
    """
    try:
        return int(os.environ.get('CODING_AGENT_DEPTH', '0'))
    except ValueError:
        return 0


def _subagent_config() -> tuple[int, float]:
    """Read ``subagents.max_concurrent`` / ``subagents.timeout_s`` from the config file.

    Reads the path from ``CODING_AGENT_CONFIG`` (set by main.py), falling back
    to ``config.json``, and re-reads on every call (mirrors the pattern used
    by ``tools/git.py``'s ``_allow_destructive``). Falls back to defaults on
    any I/O or parse error, or when the values are malformed.
    """
    max_concurrent = _DEFAULT_MAX_CONCURRENT
    timeout_s = _DEFAULT_TIMEOUT_S
    try:
        config_path = Path(os.environ.get('CODING_AGENT_CONFIG', 'config.json'))
        data = json.loads(config_path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return max_concurrent, timeout_s

    section = data.get('subagents') or {}
    if not isinstance(section, dict):
        return max_concurrent, timeout_s

    raw_max = section.get('max_concurrent', _DEFAULT_MAX_CONCURRENT)
    if isinstance(raw_max, int) and not isinstance(raw_max, bool) and raw_max >= 1:
        max_concurrent = raw_max

    raw_timeout = section.get('timeout_s', _DEFAULT_TIMEOUT_S)
    if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool) and raw_timeout > 0:
        timeout_s = raw_timeout

    return max_concurrent, timeout_s


def _spawn_one(
    main_py: Path,
    spec: dict[str, str],
    child_env: dict[str, str],
    timeout_s: float,
    mode: str,
) -> dict[str, Any]:
    """Run one child one-shot subprocess and return a normalized outcome dict.

    The child is launched with ``--mode`` set to the parent's own active mode
    (never a different one) — the child's toolset is exactly the parent's, so
    a read-only research parent cannot escalate a child into an edit-capable
    mode.

    Returns a dict with keys ``exit_code`` (int or None), ``timed_out`` (bool),
    ``spawn_error`` (bool), ``stdout`` (str), ``stderr`` (str).
    """
    try:
        proc = subprocess.run(
            ['python3', str(main_py), '-p', '-', '--mode', mode],
            cwd=spec['cwd'],
            input=spec['prompt'],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=child_env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b'').decode('utf-8', 'replace')
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b'').decode('utf-8', 'replace')
        return {
            'exit_code': None,
            'timed_out': True,
            'spawn_error': False,
            'stdout': stdout,
            'stderr': stderr,
        }
    except OSError as exc:
        return {
            'exit_code': None,
            'timed_out': False,
            'spawn_error': True,
            'stdout': '',
            'stderr': str(exc),
        }

    return {
        'exit_code': proc.returncode,
        'timed_out': False,
        'spawn_error': False,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
    }


def _render_results(
    specs: list[dict[str, str]],
    results: list[dict[str, Any]],
    timeout_s: float,
) -> str:
    """Render one block per child: first prompt line, status, full answer, stderr tail on failure."""
    blocks: list[str] = []
    for i, (spec, result) in enumerate(zip(specs, results)):
        first_line = spec['prompt'].splitlines()[0][:200]
        header = f'--- child {i + 1}: {first_line} ---'

        failed = bool(result['timed_out']) or bool(result.get('spawn_error')) or result['exit_code'] != 0

        if result['timed_out']:
            status = f'TIMED OUT after {timeout_s}s'
        elif result.get('spawn_error'):
            status = 'FAILED TO SPAWN'
        else:
            status = f"exit code: {result['exit_code']}"

        answer = result['stdout'] or ''
        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[:_MAX_ANSWER_CHARS] + f'\n[truncated at {_MAX_ANSWER_CHARS} characters]'

        block_lines = [header, status, '', answer.strip() or '(empty answer)']

        if failed:
            stderr_text = result.get('stderr') or ''
            tail_lines = stderr_text.splitlines()[-_STDERR_TAIL_LINES:]
            if tail_lines:
                block_lines += ['', '--- stderr (tail) ---', '\n'.join(tail_lines)]

        blocks.append('\n'.join(block_lines))

    return '\n\n'.join(blocks)
