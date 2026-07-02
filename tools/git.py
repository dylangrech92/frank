"""Git tool: execute a restricted set of git subcommands inside the project root."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

import json

from tools.base import Tool
from tools.result import ToolResult


# ---------------------------------------------------------------------------
# Allow-list and gates
# ---------------------------------------------------------------------------

ALLOWED_SUBCOMMANDS: frozenset[str] = frozenset(
    (
        'status',
        'diff',
        'log',
        'add',
        'commit',
        'branch',
        'checkout',
        'show',
        'rev-parse',
        'reset',
        'clean',
        'restore',
    )
)

# Shapes that are always destructive (regardless of allow_destructive flag):
# any form of these subcommands is gated.
DESTRUCTIVE_SUBCOMMANDS: frozenset[str] = frozenset(('reset', 'clean', 'restore'))


def _allow_destructive() -> bool:
    """Read ``allow_destructive`` from the git section of *config.json* on every call.

    Re-reads config.json each time so a config change takes effect on REPL relaunch.

    Returns:
        True when the ``git.allow_destructive`` key in config.json is truthy;
        False when the file, the git section, or the key is missing, or when an
        I/O / parse error occurs.
    """
    try:
        config_path = Path('config.json')
        text = config_path.read_text(encoding='utf-8')
        data = json.loads(text)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False

    git_section: dict | None = data.get('git') or {}
    return bool(git_section.get('allow_destructive'))


def _is_destructive(tokens: list[str]) -> bool:
    """Return ``True`` when *tokens* contain a destructive git operation.

    Destructive shapes:

    * Any form of ``reset``, ``clean``, or ``restore``.
    * ``checkout -- <path>`` style (checkout followed by a lone ``--`` token).

    Checkout **without** a ``--`` token (e.g. ``checkout -b newbranch``) is never
    destructive and is always allowed when checkout is in the allow-list.

    Args:
        tokens: Parsed argument list (first token should be the subcommand).

    Returns:
        True if the operation is destructive and requires the gate.
    """
    if not tokens:
        return False

    cmd = tokens[0]

    # checkout -- <paths>  is destructive
    if cmd == 'checkout' and '--' in tokens:
        return True

    # any form of reset, clean, restore
    if cmd in DESTRUCTIVE_SUBCOMMANDS:
        return True

    return False


class Git(Tool):
    """Runs a restricted set of git subcommands inside the project root.

    Only whitelisted subcommands are permitted. Destructive operations (reset,
    clean, restore checkout ``--`` style) are blocked by default; set
    ``git.allow_destructive`` to **true** in *config.json* to enable them.

    Args:
        args: Everything after the word ``git``, e.g. ``"status"`` or
              ``"-a -v"`` for ``git add -a -v``.
    """

    name = 'git'
    description = (
        'Runs a restricted set of git subcommands inside the project root. '
        'Only whitelisted subcommands are allowed; destructive operations (reset, clean, '
        'restore checkout -- style) are blocked by default. Set '
        '`git.allow_destructive` to true in config.json to enable them.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'args': {
                'type': 'string',
                'description': (
                    'Arguments passed directly after the ``git`` subcommand. '
                    'Examples: ``"status"``, ``"add -u"``, ``"commit -m msg"``.'
                ),
            },
        },
        'required': ['args'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute a whitelisted git subcommand in the project root.

        Args:
            args: Gitness arguments for git (required). Must be non-empty and
                  non-whitespace-only after splitting.

        Returns:
            A ``ToolResult`` describing success or failure of the git operation.
        """
        raw = kwargs.get('args') if isinstance(kwargs.get('args'), str) else ''

        # --- Argument validation ---
        tokens = shlex.split(raw)

        if not raw.strip():
            return ToolResult.err(
                'Empty or whitespace-only arguments are not allowed.',
                code='bad-arguments',
                hint='Pass at least one argument after git, e.g. "git status".',
            )

        subcommand = tokens[0]

        # --- Subcommand allow-list gate ---
        if subcommand not in ALLOWED_SUBCOMMANDS:
            return ToolResult.err(
                f'Subcommand "{subcommand}" is not in the allow-list.',
                code='subcommand-not-allowed',
                hint=(
                    f'Allowed subcommands are: {", ".join(sorted(ALLOWED_SUBCOMMANDS))}.'
                ),
            )

        # --- Destructive gate (lazy config read) ---
        if _is_destructive(tokens):
            if _allow_destructive():
                pass  # allowed
            else:
                return ToolResult.err(
                    f'The git subcommand "{subcommand}" is destructive and is blocked. '
                    'To enable destructive operations, set `git.allow_destructive` to true '
                    'in config.json.',
                    code='destructive-git-blocked',
                    hint='Set "git": { "allow_destructive": true } in your config.json and restart this repl.',
                )

        # --- Execute the command ---
        project_root = Path.cwd()
        cmd_list = ['git'] + tokens

        try:
            result = subprocess.run(
                cmd_list,
                shell=False,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                return ToolResult.err(
                    'Timed out after 60 seconds.',
                    code='timeout',
                    hint='Git operations are limited to a 60-second timeout.',
                )
            return ToolResult.err(
                f'git command failed: {exc}',
                code='subprocess-error',
            )

        stdout = result.stdout
        stderr = result.stderr

        # --- Build output body ---
        if stderr:
            body = (
                f'--- stdout ---\n{stdout}\n'
                f'--- stderr ---\n{stderr}'
            )
        else:
            body = f'--- stdout ---\n{stdout}'

        # Non-zero exit codes are still ok results (the user may want to inspect)
        return ToolResult.ok(
            body,
            exit_code=result.returncode,
        )
