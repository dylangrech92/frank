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


# ---------------------------------------------------------------------------
# Read path: similarity recall of decision/pivot/spec nodes + 1-hop expansion
# ---------------------------------------------------------------------------

_RECALL_TYPES = ("decision", "pivot", "spec")  # rules are always-injected, never similarity-recalled
_EXPANSION_SCORE_FACTOR = 0.5
_REL_FLOOR = 0.15

_graph_registered = False


def _fts_or_query(query: str) -> str:
    """Build an FTS5 MATCH expression that ORs each quoted token (broad recall)."""
    tokens = [t for t in query.replace('"', " ").split() if t]
    if not tokens:
        return '""'
    return " OR ".join('"' + t + '"' for t in tokens)


def _node_result(row, score: float, via: str) -> dict:
    """Shape a graph_nodes row into the standard recall result dict."""
    ntype = row["type"]
    title = row["title"]
    body = row["body"] or ""
    active = row["active"]
    superseded = row["superseded_at"] is not None
    tag = " [superseded]" if superseded else ""
    text = f"[{ntype}]{tag} {title}: {body}".rstrip()
    return {
        "layer": "graph",
        "id": row["id"],
        "kind": ntype,
        "key": str(row["id"]),
        "value": title,
        "text": text,
        "active": active,
        "superseded": superseded,
        "via": via,
        "score": round(score, 6),
    }


def recall_graph(ctx, query: str, limit: int) -> list[dict]:
    """Recall active decision/pivot/spec nodes matching *query*, then 1-hop expand.

    Hybrid FTS(bm25) + vector(distance) scoring over active recall-type nodes; a
    relative floor prunes weak hits. Each surviving hit is then expanded one hop
    along graph_edges (either direction): the connected nodes -- INCLUDING
    superseded ones -- are returned as history context at a reduced score. Rules
    are never similarity-recalled (they are always injected via the context
    provider). Returns primary hits followed by their 1-hop neighbours.
    """
    conn = ctx.store.conn

    s_fts_map: dict[int, float] = {}
    s_vec_map: dict[int, float] = {}

    # --- FTS leg ---------------------------------------------------------
    try:
        fts_results = conn.execute(
            "SELECT rowid, bm25(graph_nodes_fts) AS b FROM graph_nodes_fts "
            "WHERE graph_nodes_fts MATCH ?",
            (_fts_or_query(query),),
        ).fetchall()
        bm25_values = [r["b"] for r in fts_results]
        if bm25_values:
            b_min = min(bm25_values)
            b_max = max(bm25_values)
            for r in fts_results:
                raw = r["b"]
                s_fts = 1.0 if b_max == b_min else (b_max - raw) / (b_max - b_min)
                s_fts_map[r["rowid"]] = max(0.0, min(1.0, float(s_fts)))
    except Exception:
        pass

    # --- Vector leg ------------------------------------------------------
    if ctx.store.vec_enabled and ctx.embedder.available:
        qv = ctx.embedder.embed(query)
        if qv is not None:
            try:
                import sqlite_vec

                vec_bytes = sqlite_vec.serialize_float32(qv)
                vec_results = conn.execute(
                    "SELECT rowid, distance FROM graph_nodes_vec "
                    "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                    (vec_bytes, limit * 4),
                ).fetchall()
                dist_values = [r["distance"] for r in vec_results]
                if dist_values:
                    d_min = min(dist_values)
                    d_max = max(dist_values)
                    for r in vec_results:
                        dist = r["distance"]
                        if d_min == d_max:
                            s_vec = 1.0 / (1.0 + abs(dist))
                        else:
                            normalised = (dist - d_min) / (d_max - d_min)
                            s_vec = 1.0 - float(normalised) * 0.999
                        s_vec_map[r["rowid"]] = max(0.0, min(1.0, s_vec))
            except (ImportError, ModuleNotFoundError):
                pass

    if not s_fts_map and not s_vec_map:
        return []

    all_ids = set(s_fts_map) | set(s_vec_map)
    placeholders = ",".join(str(i) for i in all_ids)
    active_types = ",".join("'" + t + "'" for t in _RECALL_TYPES)
    node_map = {
        r["id"]: r
        for r in conn.execute(
            f"SELECT id, type, title, body, extra, active, superseded_at "
            f"FROM graph_nodes WHERE id IN ({placeholders}) "
            f"AND active = 1 AND type IN ({active_types})"
        ).fetchall()
    }

    relevance: dict[int, float] = {}
    for nid in node_map:
        s_val = max(s_fts_map.get(nid, 0.0), s_vec_map.get(nid, 0.0))
        if s_val > 0:
            relevance[nid] = s_val
    if not relevance:
        return []

    max_rel = max(relevance.values())
    threshold = _REL_FLOOR * max_rel if max_rel > 0 else 0.0
    relevance = {nid: s for nid, s in relevance.items() if s >= threshold}

    primary = sorted(relevance.items(), key=lambda x: x[1], reverse=True)[:limit]
    primary_ids = {nid for nid, _ in primary}

    results: list[dict] = [_node_result(node_map[nid], score, "direct") for nid, score in primary]

    # --- 1-hop expansion (includes superseded nodes as history) ----------
    expansion: dict[int, float] = {}
    for nid, score in primary:
        neighbours = conn.execute(
            "SELECT CASE WHEN from_id=? THEN to_id ELSE from_id END AS other "
            "FROM graph_edges WHERE from_id=? OR to_id=?",
            (nid, nid, nid),
        ).fetchall()
        for nb in neighbours:
            other = nb["other"]
            if other in primary_ids:
                continue
            exp_score = score * _EXPANSION_SCORE_FACTOR
            if exp_score > expansion.get(other, 0.0):
                expansion[other] = exp_score

    if expansion:
        exp_ph = ",".join(str(i) for i in expansion)
        exp_rows = {
            r["id"]: r
            for r in conn.execute(
                f"SELECT id, type, title, body, extra, active, superseded_at "
                f"FROM graph_nodes WHERE id IN ({exp_ph})"
            ).fetchall()
        }
        for oid, exp_score in sorted(expansion.items(), key=lambda x: x[1], reverse=True):
            row = exp_rows.get(oid)
            if row is not None:
                results.append(_node_result(row, exp_score, "1-hop"))

    return results


def forget_graph(ctx, key, kind: str | None = None) -> int:
    """Deactivate (forget) a graph node by id. Returns the count invalidated.

    *key* is the node id (int or its string form). When *kind* is supplied and the
    node's type does not match it, nothing is invalidated (returns 0) so a forget
    aimed at another layer's key is a no-op here.
    """
    try:
        node_id = int(key)
    except (TypeError, ValueError):
        return 0

    conn = ctx.store.conn
    row = conn.execute(
        "SELECT id, type, active FROM graph_nodes WHERE id=?", (node_id,)
    ).fetchone()
    if row is None or row["active"] == 0:
        return 0
    if kind is not None and row["type"] != kind:
        return 0

    conn.execute("UPDATE graph_nodes SET active=0 WHERE id=?", (node_id,))
    conn.commit()
    return 1


def register_graph_layer() -> None:
    """Register the graph recall/forget layer once (idempotent)."""
    global _graph_registered
    if _graph_registered:
        return
    try:
        from memory.recall import register_layer  # type: ignore[attr-defined]

        register_layer("graph", recall_graph, forget_graph)
    except Exception:
        return
    _graph_registered = True
