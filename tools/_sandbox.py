"""Path resolution guard and mutation event bus for tool safety.

The leading underscore keeps this module out of tool discovery by the registry.

Exports
-------
resolve_in_root : verify a candidate path stays under *root*
MUTATION_SUBSCRIBERS, subscribe_mutations, emit_mutation, log_mutation_to_stderr
"""

from __future__ import annotations

import sys
from pathlib import Path

# Directory names never worth walking when a tool sweeps the project tree:
# version-control internals, the harness's own state dir, Python bytecode
# caches, and vendored/virtual-env trees. Shared here so every tool that walks
# the tree agrees on the same skip set instead of keeping private copies that
# can drift apart. Tools with a deliberately different skip policy (e.g.
# gitignore-driven listing) keep their own filter.
IGNORED_DIRS = frozenset({'.git', '.coding_agent', '__pycache__', 'node_modules', '.venv', 'venv'})


def resolve_in_root(root: str | Path, candidate: str | Path) -> Path:
    """Resolve *candidate* as a path under *root*.

    The candidate must be relative; absolute candidates are rejected.
    Symlinks are followed via :meth:`Path.resolve`. After joining the
    absolute *root* and *candidate* the resolved real path must not land
    outside the resolved real root.  This covers dot-dot traversal as well
    as symlinks pointing outside the project tree.

    Args:
        root: The project root directory.
        candidate: A relative path to resolve under *root*.

    Returns:
        The resolved absolute ``Path`` when it is safe (equal to or inside *root*).

    Raises:
        ValueError: When *candidate* is absolute, or the resolved path escapes *root*.
    """
    root_path = Path(root).resolve()
    cand_path = Path(candidate)

    if cand_path.is_absolute():
        raise ValueError(
            f"absolute paths are not allowed; paths must be given relative "
            f"to the project root. Got: {candidate!r}"
        )

    resolved = (root_path / cand_path).resolve()

    if resolved == root_path or resolved.is_relative_to(root_path):
        return resolved

    raise ValueError(
        f"path escapes the project root: {candidate!r} "
        f"resolves to {resolved}, which is not under {root_path}"
    )


# ---------------------------------------------------------------------------
# Mutation event bus
# ---------------------------------------------------------------------------

MUTATION_SUBSCRIBERS: list[callable] = []  # type: ignore[assignment]


def subscribe_mutations(callback: callable) -> None:
    """Register *callback* to receive every mutation event dict.

    Args:
        callback: A function accepting a single ``dict`` argument.
    """
    MUTATION_SUBSCRIBERS.append(callback)


def log_mutation_to_stderr(event: dict) -> None:
    """Default subscriber that prints a mutation line to stderr.

    Prints one line in the form ``mutation: KIND PATH`` per event.

    Args:
        event: The mutation event dict with at least ``kind`` and ``path`` keys.
    """
    print(f"mutation: {event['kind']} {event['path']}", file=sys.stderr)


# Always-visible by default.
subscribe_mutations(log_mutation_to_stderr)

KIND_TYPES = ('created', 'changed', 'deleted', 'renamed')


def emit_mutation(kind: str, path: str | Path, *, extra: dict | None = None) -> None:
    """Notify every registered subscriber that a file was mutated.

    The event dict carries ``kind``, ``path`` (stringified), and the optional
    ``extra`` dict (e.g. ``{'old_path': ...}`` for renames).

    Args:
        kind: One of ``'created'``, ``'changed'``, ``'deleted'``, ``'renamed'``.
        path: The file path that was mutated.
        extra: Optional mapping of additional detail (unused by the default subscriber).
    """
    event: dict = {
        'kind': kind,
        'path': str(path),
        'extra': extra or {},
    }

    if kind not in KIND_TYPES:
        raise ValueError(
            f"invalid kind {kind!r}; must be one of {', '.join(map(repr, KIND_TYPES))}"
        )

    for subscriber in MUTATION_SUBSCRIBERS:
        try:
            subscriber(event)
        except Exception as exc:
            print(f'subscriber error on mutation event: {subscriber}: {type(exc).__name__}({exc!r})', file=sys.stderr)

