"""Process runtime for executing one-shot commands in the project workspace.

The deny-list patterns are a config-free constant set defined at import time.
This design choice ensures no file I/O is needed to gate hazardous commands,
avoids YAML/JSON schema drift across environments, and keeps evaluation a
pure string-match pass with zero dependencies beyond stdlib ``re`` and
``subprocess``.

Background-process handle registry extends this module with long-lived subprocess
management: start, stream output from, stop, and reap background processes.
"""

from __future__ import annotations

import os
import re
import subprocess  # noqa: S404 - shell commands are user-provided tool inputs, not untrusted data
import threading
from collections import deque
from itertools import count
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subprocess import Popen


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


# ---------------------------------------------------------------------------
# Background-process handle registry
# ---------------------------------------------------------------------------

_background_handles: dict[str, "BackgroundProcess"] = {}
_handle_counter = count(1)


class BackgroundProcess:
    """A handle for a long-lived background shell process.

    Attributes:
        id: Unique string handle (e.g. ``proc-1``).
        command: The original shell command string that spawned the process.
        proc: The underlying :class:`subprocess.Popen` object.
        output_buffer: Ring buffer of combined stdout/stderr lines (maxlen 2000).
        lock: Threading lock guarding *output_buffer*.
        reader: The daemon reader thread delivering lines into *output_buffer*.
    """

    __slots__ = ("id", "command", "proc", "pgid", "output_buffer", "lock", "reader")

    id: str
    command: str
    proc: Popen[str]
    pgid: int
    output_buffer: deque[str]
    lock: threading.Lock
    reader: threading.Thread

    def __init__(
        self,
        handle_id: str,
        command: str,
        proc: Popen[str],
        buffer_maxlen: int = 2000,
    ) -> None:
        """Initialise a new *BackgroundProcess* record.

        Args:
            handle_id: Unique identifier that maps to this process in the
                registry returned by :func:`_list_handles`.
            command: The original command string used to spawn the subprocess.
            proc: Running :class:`subprocess.Popen` instance.
            buffer_maxlen: Maximum lines kept in the ring buffer.  Defaults
                to 2000 so that a short burst of output never exhausts memory.
        """
        self.id = handle_id
        self.command = command
        self.proc = proc
        # start_new_session=True makes the child its own process-group leader,
        # so pgid == proc.pid. Captured at spawn so stop_background can kill the
        # group even after this shell itself has exited — which happens when the
        # command shell-backgrounds a server with `&` and returns, orphaning the
        # server in this process group. Re-deriving via os.getpgid(proc.pid)
        # would raise ProcessLookupError on the dead shell and leak the orphan.
        self.pgid = proc.pid
        self.output_buffer = deque(maxlen=buffer_maxlen)
        self.lock = threading.Lock()
        self.reader = threading.Thread(
            target=_reader_thread_func,
            args=(proc.stdout, self.output_buffer, self.lock),
            daemon=True,
            name=f"bgread-{handle_id}",
        )


def _reader_thread_func(
    stream: object,
    buf: deque[str],
    lock: threading.Lock,
) -> None:
    """Reader thread target (module-private).

    Reads lines from *stream* until EOF and appends every line to *buf*.

    This function holds *lock* while writing each line so that callers
    can acquire the same lock safely in ``read_output_from`` / ``stop_background``.
    """
    # pyre-ignore: stream is text IO opened by Popen above
    for line in stream:  # type: ignore[union-attr]
        with lock:
            buf.append(line.rstrip("\r\n"))
    stream.close()


def start_background(
    cmd: str,
    project_root: str,
    *,
    buffer_maxlen: int = 2000,
) -> str | None:
    """Spawn *cmd* as a tracked background shell process.

    The command is started with ``shell=True``, ``start_new_session=True`` so
    the entire process tree can be killed later via process-group SIGTERM/SIGKILL.
    Standard error is merged into stdout and lines are drained into an in-memory
    ring buffer maintained by a daemon reader thread.

    Args:
        cmd: Shell command string to execute.
        project_root: Working directory for the subprocess (cwd).
        buffer_maxlen: Maximum lines kept in the ring buffer per process.

    Returns:
        A short string handle such as ``proc-1``, or ``None`` when *cmd* is
        blocked by a deny-list pattern.
    """
    denied = is_denied(cmd)
    if denied:
        return None

    next_handle_id = f"proc-{next(_handle_counter)}"

    proc = subprocess.Popen(  # noqa: S602/S603 - shell=True intentional; start_new_session for clean process-group kill
        cmd,
        shell=True,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        start_new_session=True,
    )

    handle = BackgroundProcess(next_handle_id, cmd, proc, buffer_maxlen)
    handle.reader.start()

    _background_handles[next_handle_id] = handle
    return next_handle_id


def read_output_from(handle_id: str) -> dict | None:
    """Return the current state of a tracked background process.

    The buffered output is **drained** — a subsequent call will return only lines
    produced since the previous call.

    Args:
        handle_id: Handle string returned by :func:`start_background`.

    Returns:
        A dict::

            {
                "output": str,           # All buffered lines joined on \\n with trailing \\n; "" when no pending data
                "running": bool,         # True while the subprocess is alive
                "exit_code": int | None, # exit code (``None`` while running)
            }

        Returns ``None`` when *handle_id* is not found in the registry.
    """
    handle = _background_handles.get(handle_id)
    if handle is None:
        return None

    with handle.lock:
        lines = list(handle.output_buffer)
        handle.output_buffer.clear()

    if lines:
        output = "\n".join(lines) + "\n"
    else:
        output = ""
    running = handle.proc.poll() is None
    exit_code = handle.proc.returncode  # type: ignore[union-attr]

    return {
        "output": output,
        "running": running,
        "exit_code": exit_code if not running else None,
    }


def stop_background(handle_id: str) -> dict | None:
    """Stop a tracked background process and drain its remaining output.

    Sends SIGTERM to the entire process group; if the process is still alive
    after two seconds it receives SIGKILL.  The reader thread is joined, the
    handle is removed from the registry, and the final drained buffer plus the
    exit code are returned.

    Args:
        handle_id: Handle string returned by :func:`start_background`.

    Returns:
        A dict::

            {
                "output": str,           # Final output drained after kill with trailing \\n; "" when no data
                "exit_code": int | None,# Exit code from the dead process
            }

        Returns ``None`` when *handle_id* is not found in the registry.
    """
    handle = _background_handles.pop(handle_id, None)
    if handle is None:
        return None

    # --- Kill the entire process group (grandchildren included) ---
    # Use the pgid captured at spawn rather than re-deriving it via
    # os.getpgid(handle.proc.pid): when the command shell-backgrounds a server
    # with `&` and then exits, handle.proc.pid is already dead and os.getpgid
    # raises ProcessLookupError, leaking the orphaned server. The recorded pgid
    # still identifies that (possibly orphaned) process group.
    try:
        os.killpg(handle.pgid, 15)  # SIGTERM
    except (ProcessLookupError, PermissionError):
        # ProcessLookupError: group already gone. PermissionError: the pgid was
        # recycled to an unrelated process after our member exited (common when
        # a background command failed fast and its PID was reused) — not our
        # group anymore, nothing to signal.
        pass

    # Wait up to 2 s for graceful termination before escalating. Probe the
    # whole process group (not just handle.proc, which may already be a dead
    # shell) so an orphaned `&` server that survives SIGTERM still triggers the
    # SIGKILL escalation.
    alive_before_sigkill = False
    for _ in range(20):
        if not _group_alive(handle.pgid):
            break
        import time

        time.sleep(0.1)
    else:
        alive_before_sigkill = True

    if not _group_alive(handle.pgid):
        sigkilled = False
    else:
        try:
            os.killpg(handle.pgid, 9)  # SIGKILL
        except (ProcessLookupError, PermissionError):
            pass
        sigkilled = True

        # Give the killed process a brief window to release its fd.
        import time

        time.sleep(0.1)

    # Ensure proc is fully reaped so poll/returncode settle.
    if handle.proc.poll() is None:  # type: ignore[union-attr]
        try:
            handle.proc.wait(timeout=2)  # type: ignore[union-attr]
        except subprocess.TimeoutExpired:
            pass

    exit_code = handle.proc.returncode  # type: ignore[union-attr]

    # Drain any remaining lines the reader thread may have buffered.
    with handle.lock:
        lines = list(handle.output_buffer)
        handle.output_buffer.clear()

    if lines:
        output = "\n".join(lines) + "\n"
    else:
        output = ""

    # Join the reader thread (it should already be dead after EOF; give it
    # a brief window just in case).
    handle.reader.join(timeout=2)

    return {
        "output": output,
        "exit_code": exit_code,
    }


def _group_alive(pgid: int) -> bool:
    """Return True if any process remains in process group *pgid*.

    Sends signal 0 (probe) to the whole group: it succeeds if at least one
    member is alive and raises :class:`ProcessLookupError` once the group is
    empty. Used by :func:`stop_background` to decide SIGKILL escalation against
    the *group* rather than the (possibly already-dead) shell PID — so an
    orphaned ``&`` server that ignores SIGTERM is still caught.
    """
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        # ProcessLookupError: the group is empty (all members gone).
        # PermissionError: the pgid was recycled to an unrelated process after
        # our member exited — treat as "not our group, nothing left to kill".
        return False
    return True


def reap_all() -> list[str]:
    """Stop every registered background process.

    Called on REPL / interpreter shutdown to perform a clean teardown of all
    long-lived subprocesses spawned via :func:`start_background`.

    Returns:
        A list of handle IDs that were reaped, in ascending order.  The
        registry is emptied after this call.
    """
    handled = sorted(_background_handles)
    for _handle_id in handled:
        # Never let one handle's teardown abort the rest of shutdown — a handle
        # whose process already exited (e.g. a fast-failing background command)
        # must not prevent the remaining live processes from being reaped.
        try:
            stop_background(_handle_id)
        except Exception:
            _background_handles.pop(_handle_id, None)
    return handled
