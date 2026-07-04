"""recall.py - Memory Recall Coordinator + Layer Registry

Registry seam for the coding agent memory subsystem. Layers register a recall
handler AND a forget handler; later phases extend recall+forget by registration
only.
"""

import os
from dataclasses import dataclass

from memory.store import open_store
from memory.embedding import EmbeddingService


@dataclass
class MemoryContext:
    store: object
    embedder: object
    project_root: str


_LAYERS: list[dict] = []


def register_layer(name: str, recall_fn, forget_fn) -> None:
    """Register or replace a named recall/forget layer (idempotent by name)."""
    for i, layer in enumerate(_LAYERS):
        if layer["name"] == name:
            _LAYERS[i] = {"name": name, "recall": recall_fn, "forget": forget_fn}
            return
    _LAYERS.append({"name": name, "recall": recall_fn, "forget": forget_fn})


def registered_layers() -> list[str]:
    """Return the list of registered layer names (handy for tests)."""
    return [layer["name"] for layer in _LAYERS]


_CONTEXTS: dict[str, MemoryContext] = {}


def get_memory(project_root: str | None = None) -> MemoryContext:
    """Return (lazily building + caching) the MemoryContext for project_root."""
    root = os.path.abspath(project_root or os.getcwd())
    cached = _CONTEXTS.get(root)
    if cached is not None:
        return cached
    store = open_store(root)
    model_path = os.environ.get("CODING_AGENT_EMBED_MODEL") or None
    embedder = EmbeddingService(model_path=model_path)
    ctx = MemoryContext(store=store, embedder=embedder, project_root=root)
    _ensure_layers_registered()
    _CONTEXTS[root] = ctx
    return ctx


def _ensure_layers_registered() -> None:
    """Import built-in layers and let them self-register (lazy, to break the cycle)."""
    try:
        from memory.atomic import register_facts_layer
        register_facts_layer()
    except Exception:
        pass

    try:
        from memory.episodic import register_episodic_layer
        register_episodic_layer()
    except Exception:
        pass

    try:
        from memory.graph import register_graph_layer
        register_graph_layer()
    except Exception:
        pass


def recall(query: str, project_root: str | None = None, limit: int = 10) -> list[dict]:
    """Hybrid recall across all registered layers; top `limit` by score desc."""
    ctx = get_memory(project_root)
    results: list[dict] = []
    for layer in _LAYERS:
        try:
            hits = layer["recall"](ctx, query, limit)
        except Exception:
            continue
        if hits:
            results.extend(hits)
    if not results:
        return []
    results.sort(key=lambda d: d["score"], reverse=True)
    return results[:limit]


def forget(key: str, kind: str | None = None, project_root: str | None = None) -> int:
    """Invalidate across all registered layers; returns total invalidated."""
    ctx = get_memory(project_root)
    total = 0
    for layer in _LAYERS:
        try:
            count = layer["forget"](ctx, key, kind)
        except Exception:
            continue
        total += count
    return total


def remember(kind: str, key: str, value: str, project_root: str | None = None) -> int:
    """Convenience passthrough used by the remember tool (facts layer only)."""
    ctx = get_memory(project_root)
    from memory.atomic import remember as _remember
    return _remember(ctx, kind, key, value)
