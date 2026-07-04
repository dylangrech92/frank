"""graph.py - Data-graph memory: typed nodes (rule/decision/pivot/spec) + edges.

Write path over the P12-created graph_nodes / graph_edges tables. Every write
maintains graph_nodes_fts (external-content FTS5) and graph_nodes_vec (vec0
float[768]). Pivot supersession is transactional (all-or-nothing): if any target
node is missing or any write fails, the whole operation is rolled back so no
pivot node and no supersedes edge are left behind.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.recall import MemoryContext

NODE_TYPES = ("rule", "decision", "pivot", "spec")
EDGE_TYPES = ("supersedes", "implements", "constrains", "refines", "relates_to")


def _now_iso() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _embed(ctx, text: str):
    """Best-effort 768-d embedding of *text*; None when the embedder is unavailable."""
    emb = getattr(ctx, "embedder", None)
    if emb is None or not getattr(emb, "available", False):
        return None
    try:
        return emb.embed(text)
    except Exception:
        return None


def _insert_fts(store, rowid: int, title: str, body: str) -> None:
    """Insert the node's title/body into the external-content FTS index."""
    store.conn.execute(
        "INSERT INTO graph_nodes_fts(rowid, title, body) VALUES(?, ?, ?)",
        (rowid, title, body),
    )


def _insert_vec(store, rowid: int, emb) -> None:
    """Insert the node's embedding into the vec0 index when available."""
    if not (store.vec_enabled and emb is not None):
        return
    try:
        import sqlite_vec  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return
    store.conn.execute(
        "INSERT INTO graph_nodes_vec(rowid, embedding) VALUES(?, ?)",
        (rowid, sqlite_vec.serialize_float32(emb)),
    )


def _node_exists(store, node_id: int) -> bool:
    """Return True when a graph node with *node_id* exists (regardless of active)."""
    return (
        store.conn.execute(
            "SELECT 1 FROM graph_nodes WHERE id=?", (node_id,)
        ).fetchone()
        is not None
    )


def create_node(ctx, node_type: str, title: str, body: str, extra: dict | None = None) -> int:
    """Insert a rule/decision/spec node plus its FTS + vec entries.

    Pivots must go through :func:`record_pivot` (which also wires supersedes edges).
    Returns the new node id.
    """
    if node_type not in NODE_TYPES:
        raise ValueError(f"node_type must be one of {NODE_TYPES}, got {node_type!r}")
    if node_type == "pivot":
        raise ValueError("use record_pivot() to create pivot nodes")
    if not (title and title.strip()):
        raise ValueError("title is required")
    if body is None:
        body = ""

    store = ctx.store
    now = _now_iso()
    extra_json = json.dumps(extra) if extra else None

    cur = store.conn.execute(
        "INSERT INTO graph_nodes(type, title, body, extra, created_at, superseded_at, active) "
        "VALUES(?, ?, ?, ?, ?, NULL, 1)",
        (node_type, title, body, extra_json, now),
    )
    rowid = cur.lastrowid
    _insert_fts(store, rowid, title, body)
    _insert_vec(store, rowid, _embed(ctx, f"{title}\n{body}"))
    store.conn.commit()
    return rowid


def link_nodes(ctx, from_id: int, to_id: int, edge_type: str) -> int:
    """Insert a typed edge between two existing nodes; returns the edge id.

    Idempotent on (from_id, to_id, edge_type): a duplicate returns the existing id.
    """
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"edge_type must be one of {EDGE_TYPES}, got {edge_type!r}")

    store = ctx.store
    if not _node_exists(store, from_id):
        raise ValueError(f"from_id {from_id} does not exist")
    if not _node_exists(store, to_id):
        raise ValueError(f"to_id {to_id} does not exist")

    existing = store.conn.execute(
        "SELECT id FROM graph_edges WHERE from_id=? AND to_id=? AND edge_type=?",
        (from_id, to_id, edge_type),
    ).fetchone()
    if existing is not None:
        return existing["id"]

    cur = store.conn.execute(
        "INSERT INTO graph_edges(from_id, to_id, edge_type, created_at) VALUES(?, ?, ?, ?)",
        (from_id, to_id, edge_type, _now_iso()),
    )
    store.conn.commit()
    return cur.lastrowid


def record_pivot(ctx, title: str, why: str, supersedes: list[int]) -> int:
    """Transactionally record a pivot node that supersedes one or more nodes.

    Within a single transaction: insert the pivot node (+ FTS + vec), create a
    ``supersedes`` edge from the pivot to each target, and stamp ``superseded_at``
    plus ``active=0`` on each target. If any target id does not exist (or any write
    fails), the entire operation is rolled back -- no pivot node and no edges are
    left behind. Returns the new pivot node id.
    """
    if not (title and title.strip()):
        raise ValueError("title is required")
    targets = list(supersedes or [])
    if not targets:
        raise ValueError("a pivot must supersede at least one node")

    store = ctx.store
    now = _now_iso()
    try:
        cur = store.conn.execute(
            "INSERT INTO graph_nodes(type, title, body, extra, created_at, superseded_at, active) "
            "VALUES('pivot', ?, ?, NULL, ?, NULL, 1)",
            (title, why or "", now),
        )
        pivot_id = cur.lastrowid
        _insert_fts(store, pivot_id, title, why or "")
        _insert_vec(store, pivot_id, _embed(ctx, f"{title}\n{why or ''}"))

        for target in targets:
            if not _node_exists(store, target):
                raise ValueError(f"supersedes target {target} does not exist")
            store.conn.execute(
                "INSERT INTO graph_edges(from_id, to_id, edge_type, created_at) "
                "VALUES(?, ?, 'supersedes', ?)",
                (pivot_id, target, now),
            )
            store.conn.execute(
                "UPDATE graph_nodes SET superseded_at=?, active=0 WHERE id=?",
                (now, target),
            )

        store.conn.commit()
        return pivot_id
    except Exception:
        store.conn.rollback()
        raise
