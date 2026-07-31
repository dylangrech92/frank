"""Advisory single-writer lock on a session transcript.

One transcript file, one live writer. ``acquire`` claims ``<transcript>.lock`` by
writing this process's pid into it and refuses when the pid already in there is
still running; ``release`` removes the file only when this process is the one
named inside it, so a run that lost its lock to someone else can never delete
that other run's claim on the way out.

Liveness is decided with ``os.kill(pid, 0)``, which separates the three cases
that matter: the process is gone (stale lock — take it), the process answers
(refuse), or the process exists but we are not allowed to signal it (refuse — it
is someone else's live run, not a leftover file). An unparseable lock body is no
evidence of a live holder, so it is treated as stale too.

This lives outside ``Session`` because none of it reads session state: it is
filesystem mechanics over one path, and its failure mode — a lock that quietly
stops refusing — is invisible from the class's own behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path


def acquire(lock_path: Path, session_id: str) -> None:
    """Claim an advisory lock on this session's transcript.

    Writes ``<transcript>.lock`` containing this process's pid. If a lock
    file already exists and its pid is still alive, refuse to proceed —
    two processes must never append to the same transcript concurrently.
    A lock left behind by a dead process is treated as stale and replaced.

    Raises:
        RuntimeError: If another live process already holds the lock.
    """
    if lock_path.exists():
        try:
            existing_pid_text = lock_path.read_text(encoding="utf-8").strip()
            existing_pid = int(existing_pid_text)
        except (OSError, ValueError):
            existing_pid = None

        alive = False
        if existing_pid is not None:
            try:
                os.kill(existing_pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                # Process exists but we can't signal it — treat as alive.
                alive = True
            except OSError:
                alive = False

        if alive:
            raise RuntimeError(
                f"session '{session_id}' is already active in process "
                f"{existing_pid} — resume it from that process or use a "
                f"different --session id"
            )

    lock_path.write_text(str(os.getpid()), encoding="utf-8")


def release(lock_path: Path) -> None:
    """Release this session's advisory lock file, if this process still owns it.

    Safe to call multiple times and safe to call even if the lock was
    never successfully acquired (e.g. constructor raised before writing
    it) — both cases are silently no-ops.
    """
    try:
        if lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock_path.unlink()
    except OSError:
        pass
