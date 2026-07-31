"""Verify-scratch tool: run a throwaway snippet against the project without
ever writing the verification harness into the project tree, so the model
has a clean alternative to inlining repro/test harnesses into production
source or repurposing a module's ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

from runtime.process import run_one_shot
from tools.base import Tool
from tools.result import ToolResult

# Suffix chosen per interpreter for readability of the temp file (and so
# language-aware tools/tracebacks referencing the path make sense). Falls
# back to no suffix for interpreters not listed here.
_INTERPRETER_SUFFIXES: dict[str, str] = {
    'python': '.py',
    'python3': '.py',
    'node': '.js',
    'php': '.php',
    'bash': '.sh',
    'sh': '.sh',
}

# Running the snippet with the project root as cwd is not enough to make project
# imports work: an interpreter handed a script path puts the SCRIPT's directory on
# the module search path, and this script deliberately lives in the system temp
# dir. Observed live — a snippet opening with `from calc import add` against a
# project whose calc.py sat right there in the working directory died on
# ModuleNotFoundError, and the model, having been promised by this tool's own
# description that imports resolve, burned two more calls inventing paths that did
# not exist. The env var per interpreter puts the project root back on the search
# path so the promise holds.
_MODULE_PATH_VARS: dict[str, str] = {
    'python': 'PYTHONPATH',
    'python3': 'PYTHONPATH',
    'node': 'NODE_PATH',
}


class VerifyScratch(Tool):
    """Runs a throwaway snippet against the real project without writing it into the tree.

    The snippet itself is written to a temporary file outside the project tree
    (never under the project root) and executed with the project root as the
    working directory *and* on the interpreter's module search path (see
    ``_MODULE_PATH_VARS``), so imports and relative paths resolve exactly as they
    would for the real code. The temp file is deleted after the run regardless
    of outcome. Because the working directory is the project root, the snippet
    still has write access to project files — it just isn't one itself, so it
    should not write into project files.
    """

    name = 'verify_scratch'
    description = (
        'Verifies a change end-to-end by running a THROWAWAY snippet against the '
        'real project. The harness itself is never written into the project '
        'tree: your snippet is written to a temporary file OUTSIDE the project '
        'tree and executed with the project root as the working directory, so '
        'imports and relative paths resolve exactly as they would for the real '
        'code; the temp file is deleted afterward. Use this to verify a fix '
        'instead of adding verification/repro code to a production file or '
        'repurposing its `if __name__ == "__main__"` block — doing that '
        "silently breaks the file's real entry point. Caution: the snippet "
        'runs with write access to the project root, so it should not write '
        'into project files. Returns stdout, stderr, and exit_code; a zero '
        "exit code means the snippet's assertions passed."
    )
    action = 'run the verification snippet'
    oversize_hint = 'have the snippet print only the summary/assertions, not full data dumps'
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'snippet': {
                'type': 'string',
                'description': (
                    'The verification code to run. Written to a temporary file '
                    'outside the project tree and executed with the project root '
                    'as the working directory.'
                ),
            },
            'interpreter': {
                'type': 'string',
                'enum': sorted(_INTERPRETER_SUFFIXES),
                'description': (
                    'The command used to execute the temp file. One of: '
                    f'{", ".join(sorted(_INTERPRETER_SUFFIXES))}. Defaults to "python".'
                ),
            },
            'timeout': {
                'type': 'integer',
                'description': 'Maximum seconds to wait for the snippet to finish. Default is 60.',
            },
        },
        'required': ['snippet'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Write *snippet* to a temp file outside the project tree and execute it
        against the project root, without ever adding the harness to the project.

        Args:
            snippet: The verification code to run (required).
            interpreter: Command used to execute the temp file (optional, default
                "python"). Must be one of the keys in ``_INTERPRETER_SUFFIXES``;
                this is validated even though the schema also declares an enum,
                since direct dispatch can bypass schema validation.
            timeout: Max seconds to wait (optional, default 60).

        Returns:
            A ``ToolResult`` describing success or failure, mirroring
            ``run_command``'s foreground shape (stdout/stderr/exit_code/timed_out).
            A non-zero exit code is treated as a failed verification and reported
            as an error result so the agent's verify-gate is not cleared.
        """
        snippet = kwargs.get('snippet') if isinstance(kwargs.get('snippet'), str) else ''
        interpreter = kwargs.get('interpreter') if isinstance(kwargs.get('interpreter'), str) else ''
        interpreter = interpreter or 'python'
        timeout_raw = kwargs.get('timeout')
        timeout: int = timeout_raw if timeout_raw is not None else 60

        if not snippet:
            return ToolResult.err(
                'snippet is required and must be a non-empty string.',
                code='missing-snippet',
            )

        if interpreter not in _INTERPRETER_SUFFIXES:
            allowed = ', '.join(sorted(_INTERPRETER_SUFFIXES))
            return ToolResult.err(
                f'interpreter {interpreter!r} is not allowed.',
                code='invalid-interpreter',
                hint=f'interpreter must be one of: {allowed}.',
            )

        suffix = _INTERPRETER_SUFFIXES.get(interpreter, '')
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix='verify_scratch_')

        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(snippet)

            project_root = str(Path.cwd())
            cmd = f'{interpreter} {shlex.quote(tmp_path)}'
            path_var = _MODULE_PATH_VARS.get(interpreter)
            if path_var is not None:
                # Prepend rather than replace: an inherited value stays usable.
                quoted_root = shlex.quote(project_root)
                cmd = f'{path_var}={quoted_root}${{{path_var}:+:${path_var}}} {cmd}'
            result = run_one_shot(cmd, project_root, timeout_seconds=timeout)

            if result['timed_out']:
                partial = result['stdout']
                stderr_part = result.get('stderr', '')
                if stderr_part:
                    body = (
                        f'--- stdout ---\n{partial}\n'
                        f'--- stderr ---\n{stderr_part}'
                    )
                else:
                    body = f'--- stdout ---\n{partial}'

                return ToolResult.err(
                    f'Verification snippet was killed after {timeout} seconds.\n\n{body}',
                    code='timeout',
                    hint=f'The snippet ran longer than {timeout}s. Try a shorter timeout or a narrower snippet.',
                )

            stdout_text = result['stdout']
            stderr_text = result.get('stderr', '')
            if stderr_text:
                output = (
                    f'--- stdout ---\n{stdout_text}\n'
                    f'--- stderr ---\n{stderr_text}'
                )
            else:
                output = f'--- stdout ---\n{stdout_text}'

            exit_code = result['exit_code']
            if exit_code not in (0, None):
                return ToolResult.err(
                    f'Verification snippet failed (exit code {exit_code}).\n\n{output}',
                    code='verification-failed',
                    hint="A non-zero exit code means the snippet's assertions did not pass. "
                    'Inspect the stdout/stderr above and fix the underlying issue.',
                    exit_code=exit_code,
                    timed_out=False,
                )

            return ToolResult.ok(
                output,
                exit_code=exit_code,
                timed_out=False,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
