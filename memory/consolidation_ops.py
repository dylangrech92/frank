from __future__ import annotations

import difflib
import os
import sys

KEY_REUSE_SIMILARITY = 0.62    # min (key+value) similarity to reuse a live atom's key (see _best_existing_key_match)


def _log(msg: str) -> None:
    print(f"consolidation: {msg}", file=sys.stderr, flush=True)


def _coerce_kind(raw) -> str:
    """Clamp a mined atom's kind to a durable, non-TTL kind (never discovery/misc)."""
    if isinstance(raw, str) and raw.strip().lower() in ("project", "convention"):
        return raw.strip().lower()
    return "project"


def _resolve_anchor_path(project_root: str, raw_path) -> str | None:
    """Resolve a model-supplied path against *project_root* to an absolute path.

    Anchors must be absolute: ``memory.anchor.is_stale`` re-opens the path
    directly at recall time, regardless of the process's current working
    directory. Returns None (repo-wide) for anything not a non-empty string.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = raw_path.strip()
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(project_root, path))


def _normalize_for_match(text: str) -> str:
    """Lowercase + collapse whitespace, for similarity comparison only."""
    return " ".join(text.lower().split())


def _best_existing_key_match(key: str, value: str, existing_atoms: list[dict]) -> str | None:
    """Best-effort reuse of a live atom's key for a same-subject fact whose
    key the model chose differently this time.

    Two independent weak-LLM key choices for the same underlying fact need
    not match (e.g. a correction turn inventing ``abilities_total`` where the
    original wrong atom lives under ``app.abilities.count``). Compares the
    proposed ``key + value`` text against each existing atom's ``key + value``
    text via ``difflib.SequenceMatcher`` on lowercased, whitespace-normalized
    strings, and returns the single best match's key if its ratio clears
    ``KEY_REUSE_SIMILARITY``. Conservative by design -- distinct facts about a
    similar broad topic (e.g. a count vs. a registry path vs. an unrelated
    module's purpose) must stay separate, not merge. Returns None if nothing
    clears the bar, ``existing_atoms`` is empty, or on any failure.
    """
    if not existing_atoms:
        return None
    try:
        proposed = _normalize_for_match(f"{key} {value}")
        best_key: str | None = None
        best_ratio = 0.0
        for atom in existing_atoms:
            existing_key = atom.get("key")
            if not isinstance(existing_key, str) or not existing_key:
                continue
            candidate = _normalize_for_match(f"{existing_key} {atom.get('value') or ''}")
            ratio = difflib.SequenceMatcher(None, proposed, candidate).ratio()
            if ratio > best_ratio:
                best_ratio, best_key = ratio, existing_key
        if best_key is not None and best_ratio >= KEY_REUSE_SIMILARITY:
            _log(f"key-reuse: model_key={key!r} -> existing_key={best_key!r} (ratio={best_ratio:.2f})")
            return best_key
        return None
    except Exception:
        return None
