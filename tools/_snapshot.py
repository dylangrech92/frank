"""Shared before/after tree-snapshot diff for tools that execute commands.

Keeps ``files_changed`` truthful for any tool that runs a shell command by
snapshotting the project tree before and after, diffing the two snapshots, and
publishing one mutation event per created/deleted/changed file.  This module is
the single source of truth for that pattern — profiling tools that need to
snapshot around a child run simply import it rather than reimplementing the
walk-and-diff logic.
"""

from __future__ import annotations

import os
from pathlib import Path

from tools._sandbox import IGNORED_DIRS, emit_mutation

# Upper bound on files walked when snapshotting the tree around a foreground
# command. A very large tree must not pay a per-command walk tax, so detection
# is abandoned (no snapshot, no diff) once the walk crosses this many files.
MAX_SNAPSHOT_FILES = 20000

# Upper bound on paths named per kind in the mutation note appended to a
# foreground result. A command that writes hundreds of files (e.g. a generator
# script) must not bloat the render, so extra paths collapse to a "+N more"
# tail.
MAX_RENDERED_PATHS = 5


def snapshot_tree(root: Path) -> dict[str, tuple[int, int]] | None:
    """Record ``{abs_path: (st_mtime_ns, st_size)}`` for every file under *root*.

    Prunes any directory whose name starts with ``.`` plus the shared
    :data:`IGNORED_DIRS`, and skips any file whose name starts with ``.``. Walk
    or stat problems degrade to a partial/absent snapshot rather than raising —
    a detection failure must never break the command result.

    Returns:
        The snapshot mapping, or ``None`` when the file cap is exceeded or the
        walk cannot be completed (so the caller skips the diff entirely).
    """
    snapshot: dict[str, tuple[int, int]] = {}
    count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith('.') and d not in IGNORED_DIRS
            ]
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                count += 1
                if count > MAX_SNAPSHOT_FILES:
                    return None
                full = os.path.join(dirpath, filename)
                try:
                    stat = os.stat(full)
                except OSError:
                    # A file that vanished mid-walk (a race) or is unreadable is
                    # simply absent from this snapshot, not a fatal error.
                    continue
                snapshot[full] = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    return snapshot


def diff_snapshots(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> tuple[list[str], list[str], list[str]]:
    """Classify the difference between two tree snapshots.

    A path present only in *after* is created, present only in *before* is
    deleted, and present in both with a different ``(mtime_ns, size)`` is
    changed. Each list is sorted so the classification is deterministic
    regardless of dict iteration order.

    Returns:
        ``(created, deleted, changed)`` — three sorted lists of absolute paths.
    """
    before_paths = set(before)
    after_paths = set(after)
    created = sorted(after_paths - before_paths)
    deleted = sorted(before_paths - after_paths)
    changed = sorted(p for p in before_paths & after_paths if before[p] != after[p])
    return created, deleted, changed


def publish_snapshot_diff(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> tuple[list[str], list[str], list[str]]:
    """Emit a mutation event for every file that a command created/changed/deleted.

    Classification is delegated to :func:`diff_snapshots`; this publishes one
    event per path in the order class created, deleted, changed. Paths are
    already absolute (from :func:`os.walk`), matching how the edit tools publish
    resolved paths. ``emit_mutation`` is called directly (not wrapped) so
    subscriber errors surface exactly as they do for the other publishers.

    Returns:
        The ``(created, deleted, changed)`` classification, so the caller can
        render the same fact it just published without recomputing it.
    """
    created, deleted, changed = diff_snapshots(before, after)
    for path in created:
        emit_mutation('created', path)
    for path in deleted:
        emit_mutation('deleted', path)
    for path in changed:
        emit_mutation('changed', path)
    return created, deleted, changed


def render_mutation_line(
    root: Path,
    created: list[str],
    deleted: list[str],
    changed: list[str],
) -> str:
    """Build the one-line world-fact note naming files a command touched.

    Paths arrive absolute (from the snapshot walk) and are rendered relative to
    *root* so the note reads in project terms. Each kind names at most
    :data:`MAX_RENDERED_PATHS` paths, then a ``+N more`` tail, so a script that
    writes many files cannot bloat the result. Empty kinds are omitted, and an
    all-empty diff yields the empty string so the caller can test truthiness.

    The note states the world fact only — which files changed — with no
    directive or conditional advice attached.
    """
    segments = [
        seg for seg in (
            _format_kind('created', _relativize(created, root)),
            _format_kind('changed', _relativize(changed, root)),
            _format_kind('deleted', _relativize(deleted, root)),
        )
        if seg is not None
    ]
    if not segments:
        return ''
    return f"(this command modified project files — {'; '.join(segments)})"


def _relativize(paths: list[str], root: Path) -> list[str]:
    """Render each absolute path in *paths* relative to *root*.

    A path that cannot be expressed relative to *root* (e.g. a different drive)
    is kept as-is rather than dropped, so nothing silently vanishes from the note.
    """
    rel: list[str] = []
    for path in paths:
        try:
            rel.append(os.path.relpath(path, root))
        except ValueError:
            rel.append(path)
    return rel


def _format_kind(label: str, paths: list[str]) -> str | None:
    """Format one ``label: p1, p2, +N more`` segment, or ``None`` when empty."""
    if not paths:
        return None
    shown = paths[:MAX_RENDERED_PATHS]
    joined = ', '.join(shown)
    extra = len(paths) - len(shown)
    if extra > 0:
        joined += f', +{extra} more'
    return f'{label}: {joined}'
