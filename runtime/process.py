"""Process runtime for executing one-shot commands in the project workspace.

The deny-list patterns are a config-free constant set defined at import time.
This design choice ensures no file I/O is needed to gate hazardous commands,
avoids YAML/JSON schema drift across environments, and keeps evaluation a
pure string-match pass with zero dependencies beyond stdlib `re` and `subprocess`.
"""

from __future__ import annotations

import os
import re
import subprocess  # noqa: S404 - shell commands are user-provided tool inputs, not untrusted data


# ---------------------------------------------------------------------------
# DENY_LIST_PATTERNS — compiled once at module load time
# ---------------------------------------------------------------------------

DENY_LIST_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # rm with recursive+force flags (bundled like -rf/-fr or separate -r -f) targeting ~, $HOME, ${HOME}, /home/... or /root/...
    (
        re.compile(r"\brm\b(?=.*\s-\w*[rR])(?=.*\s-\w*f).*\s(?:~(?:/)?|\$\{?HOME\}?|/(?:home|root)\b\S*)\s*(?:$|[;&|])"),
        "Blocked 'rm -rf' targeting home directory (~, $HOME) or critical path (/home/*, /root/*)",
    ),
    (
        re.compile(r"\brm\b(?=.*\s-\w*[rR])(?=.*\s-\w*f).*\s/\*?\s*(?:$|[;&|])"),
        "Blocked 'rm -rf' targeting filesystem root (/ or paths ending in /./ or /)",
    ),
    # mkfs with any fs type — e.g. mkfs.ext4, mkfs.vfat
    (
        re.compile(r"\bmkfs(?:\.\w+)?\b"),
        "Blocked mkfs disk-formatting command",
    ),
    # Fork bomb: :() {: :|:&  ::  {1..256} etc. — the classic colon- brace pipe form
    (
        re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
        "Blocked fork-bomb pattern (:(){ :|: style)",
    ),
    # dd writing to a device via of=/dev/*
    (
        re.compile(r"\bdd\b.*of=(/dev/\S+)"),
        "Blocked dd writing to raw device (/dev/...)",
    ),
]


def is_denied(cmd: str) -> str | None:
    """Return a human-readable deny reason if *cmd* matches any deny pattern.

    Args:
        cmd: The full command string to evaluate (e.g. "rm -rf /tmp/foo").

    Returns:
        A descriptive reason string when blocked, else ``None``.
    """
    for pattern, reason in DENY_LIST_PATTERNS:
        if pattern.search(cmd):
            return reason
    return None


def run_one_shot(
    cmd: str,
    project_root: str,
    timeout_seconds: int | None = 60,
) -> dict[str, object]:
    """Run *cmd* as a one-shot shell process inside *project_root*.

    Args:
        cmd: Shell command string to execute.
        project_root: Working directory for the subprocess (cwd).
        timeout_seconds: Max seconds before forcible termination. Defaults to 60.

    Returns:
        A plain dict with keys:

        ``stdout`` (str)
            Captured standard output text, possibly partial on timeout.
        ``stderr`` (str)
            Captured standard error text, possibly partial on timeout.
        ``exit_code`` (int | None)
            Process exit code; ``None`` when the process was killed by timeout.
        ``timed_out`` (bool)
            Whether the command exceeded *timeout_seconds*.

    Raises:
        subprocess.SubprocessError: When the child process cannot be started.
    """
    if timeout_seconds is None:
        timeout_seconds = 60

    proc = subprocess.Popen(  # noqa: S602/S603 - shell=True intentional; start_new_session for clean process-group kill
        cmd,
        shell=True,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_seconds)
        timed_out = False
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        # Kill the entire process group (grandchildren included)
        pgid = os.getpgid(proc.pid)
        try:
            os.killpg(pgid, 9)  # SIGKILL on POSIX
        except ProcessLookupError:
            pass

        stdout_bytes, stderr_bytes = proc.communicate()
        timed_out = True
        exit_code = None

    return {
        "stdout": stdout_bytes.decode("utf-8", errors="replace"),
        "stderr": stderr_bytes.decode("utf-8", errors="replace"),
        "exit_code": exit_code,
        "timed_out": timed_out,
    }
