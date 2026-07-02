"""Path resolution guard: resolves a candidate relative to a root and ensures it stays inside.

The leading underscore keeps this module out of tool discovery by the registry.
"""

from pathlib import Path


def resolve_in_root(root: str | Path, candidate: str | Path) -> Path:
    """Resolve *candidate* as a path under *root*.

    The candidate is interpreted as relative to *root*. Symlinks are followed
    via :meth:`Path.resolve`.  The resolved absolute path must be equal to *root*
    or live strictly underneath it.

    Args:
        root: The project root directory.
        candidate: A relative path to resolve under *root*.

    Returns:
        The resolved absolute ``Path`` when it is safe (equal to or inside *root*).

    Raises:
        ValueError: When the resolved path escapes *root*.
    """
    root_path = Path(root).resolve()
    try:
        # Make candidate relative first, then join so we don't accidentally resolve
        # through a symlinked parent directory.
        resolved = (root_path / Path(candidate)).resolve()
    except (TypeError, ValueError):
        raise ValueError(f"candidate {candidate!r} could not be resolved against root {root!r}")

    if resolved == root_path or (resolved.parent == root_path or resolved.is_relative_to(root_path)):
        return resolved

    raise ValueError(
        f"path escapes the project root: {candidate!r} "
        f"resolves to {resolved}, which is not under {root_path}"
    )
