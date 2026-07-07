"""Code-anchoring helpers: hash + commit stamps for persisted knowledge atoms.

A knowledge atom (a row in the ``facts`` table) may be *anchored* to a file
(and optionally a symbol within it). The anchor is a content hash taken at
learn time plus the git commit it was learned against. Recall re-checks the
anchor -- **anchor-liveness**, not a clock -- to decide whether an atom is
still trustworthy: staleness is "the code under this insight changed", never
"time has passed". See MEMORY_REDESIGN.md section 5.

Both the write side (``memory.atomic.remember``) and the read side
(``memory.atomic.recall_facts``) import from this module so the two stay in
lock-step on how a hash is computed.
"""

from __future__ import annotations

import hashlib
import os
import subprocess


def _sha256_file(path: str) -> str | None:
    """Return the sha256 hex digest of *path*'s bytes, or None if unreadable."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _short_head(cwd: str | None) -> str | None:
    """Best-effort ``git rev-parse --short HEAD``; None on any failure.

    Never raises -- a subagent's temp dir, a non-repo project, or a missing
    ``git`` binary are all expected and simply yield no commit stamp.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


def anchor_for(path: str | None, symbol: str | None = None) -> tuple[str | None, str | None]:
    """Compute ``(anchor_hash, learned_commit)`` for *path* at learn time.

    ``anchor_hash`` is the sha256 of the file's current bytes (None if *path*
    is falsy, missing, or unreadable -- i.e. a repo-wide/no-anchor atom).
    ``learned_commit`` is the short git HEAD of the repo containing *path*
    (None if not inside a git repo or ``git`` is unavailable). *symbol* is
    accepted for API symmetry with the stored schema (a future revision may
    hash just the symbol body) but does not currently affect the hash.
    Never raises.
    """
    del symbol  # not yet used to scope the hash -- accepted for schema symmetry.
    if not path:
        return None, None

    anchor_hash = _sha256_file(path)
    learned_commit = _short_head(os.path.dirname(path) or ".")
    return anchor_hash, learned_commit


def current_hash(path: str) -> str | None:
    """Recompute the sha256 hash of *path*'s CURRENT bytes (None if unreadable).

    A thin, cacheable primitive: callers checking many atoms anchored to the
    same file (e.g. ``recall_facts``) should memoise this per call rather than
    re-reading the file for every candidate.
    """
    return _sha256_file(path)


def is_stale(
    path: str | None,
    anchor_hash: str | None,
    cache: dict[str, str | None] | None = None,
) -> bool:
    """Return True if the atom anchored to *path*/*anchor_hash* has gone STALE.

    STALE means the anchor file is now missing/unreadable, or its content no
    longer hashes to *anchor_hash* (the code moved under the insight). An
    atom with no *path* (repo-wide) or no recorded *anchor_hash* (never
    verifiable) has nothing to compare against, so it is never stale by this
    check -- it is trusted, not penalised, for lack of evidence.

    Pass a shared *cache* dict when checking many candidates in one recall
    pass -- it is read/populated by file path so the same file is only ever
    hashed once per call. Never raises.
    """
    if not path or not anchor_hash:
        return False

    if cache is not None and path in cache:
        current = cache[path]
    else:
        current = current_hash(path)
        if cache is not None:
            cache[path] = current

    return current is None or current != anchor_hash
