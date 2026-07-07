"""Atomic-facts memory layer: CRUD for the facts table.

Provides a structured ``remember / forget / recall`` API over the ``facts``
table (bi-temporal keyed supersession). Staleness is anchor-liveness, not a
clock (MEMORY_REDESIGN.md section 5) -- ``recall_facts`` checks each atom's
code anchor and down-weights/labels drift; there is no TTL or time-decay in
this layer (retired M7 -- see git history for the prior clock-decay scorer).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from memory import anchor as _anchor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KINDS = ("project", "convention", "discovery", "misc")

# ---------------------------------------------------------------------------
# Lazy idempotent registration
# ---------------------------------------------------------------------------

_registered = False


def register_facts_layer() -> None:
    """Register the facts recall/forget layer with memory.recall (if available).

    Importantly this does NOT crash if ``memory.recall`` is not yet importable;
    callers that *do* have a clean ``memory/recall.py`` will call the global
    ``get_context()`` path which invokes ``register_facts_layer()`` itself.
    """
    global _registered

    if _registered:
        return

    try:
        from memory.recall import register_layer  # type: ignore[attr-defined]

        register_layer("facts", recall_facts, forget)
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass  # recall.py not yet available; will be picked up later.

    _registered = True


# Also attempt registration at import time (best-effort).
try:
    register_facts_layer()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# FTS query escaping helper
# ---------------------------------------------------------------------------


def _escape_fts_query(query: str) -> str:
    """Escape *query* for use in an FTS5 ``MATCH`` expression.

    ORs each quoted token so a multi-word natural-language question matches
    rows containing ANY term (broad lexical recall), mirroring
    ``graph.py:_fts_or_query``. Quoting each token individually keeps FTS5
    from choking on punctuation while avoiding the exact-phrase trap of
    wrapping the whole query in one quote pair (which matches nothing for
    any realistic question). Returns ``'""'`` for an empty query.
    """
    tokens = [t for t in query.replace('"', " ").split() if t]
    if not tokens:
        return '""'
    return " OR ".join('"' + t + '"' for t in tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from memory.recall import MemoryContext  # type: ignore[attr-defined]


def remember(
    ctx: "MemoryContext",
    kind: str,
    key: str,
    value: str,
    *,
    salience_floor: float = 0.1,
    d_base: float = 0.3,
    now: str | None = None,
    anchor_path: str | None = None,
    anchor_symbol: str | None = None,
    anchor_hash: str | None = None,
    learned_commit: str | None = None,
    confidence: float = 0.5,
    source: str = "legacy",
) -> int:
    """Insert a new atom, superseding any current one with the same (kind, key).

    Optional code-anchor fields (MEMORY_REDESIGN.md §5): *anchor_path*/
    *anchor_symbol* name what the insight is about (NULL/None = repo-wide);
    *anchor_hash*/*learned_commit* stamp what the code looked like when
    learned -- callers compute these with ``memory.anchor.anchor_for(path)``
    before calling ``remember`` (this function only persists what it is
    given, it does not hash files itself). *confidence* and *source*
    ('consolidation'|'explorer'|'explicit-tool'|'legacy') record how much to
    trust the atom and where it came from.

    Returns the new fact id.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown facts kind {kind!r}; allowed: {KINDS}")

    now = now or _now_iso()
    conn = ctx.store.conn

    # 1. Close every live row with the same (kind, key).
    conn.execute(
        """
        UPDATE facts
        SET valid_to = ?, active = 0
        WHERE kind = ? AND key = ? AND valid_to IS NULL AND active = 1 AND deleted_at IS NULL
        """,
        (now, kind, key),
    )

    # 2. Insert the new row.
    cur = conn.execute(
        """
        INSERT INTO facts(
            kind, key, value,
            salience_floor, d_base, retrieval_weight,
            first_seen_at, last_confirmed_at, last_accessed_at,
            valid_from, valid_to, active, deleted_at,
            anchor_path, anchor_symbol, anchor_hash, learned_commit, confidence, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1, NULL, ?, ?, ?, ?, ?, ?)
        """,
        (
            kind,
            key,
            value,
            salience_floor,
            d_base,
            1.0,
            now,
            now,
            None,
            now,
            anchor_path,
            anchor_symbol,
            anchor_hash,
            learned_commit,
            confidence,
            source,
        ),
    )
    new_id = cur.lastrowid

    # 3. Sync FTS external-content table.
    conn.execute(
        "INSERT INTO facts_fts(rowid, key, value, kind) VALUES (?, ?, ?, ?)",
        (new_id, key, value, kind),
    )

    # 4. Embed key + value if available; insert into vec0 tables on success.
    if ctx.store.vec_enabled and ctx.embedder.available:
        try:
            import sqlite_vec  # noqa: F811

            _vec_ok = True
        except (ImportError, ModuleNotFoundError):
            _vec_ok = False

        key_emb = ctx.embedder.embed(key)
        val_emb = ctx.embedder.embed(value)

        if key_emb is not None and _vec_ok:
            conn.execute(
                "INSERT INTO facts_key_vec(rowid, embedding) VALUES (?, ?)",
                (new_id, sqlite_vec.serialize_float32(key_emb)),
            )

        if val_emb is not None and _vec_ok:
            conn.execute(
                "INSERT INTO facts_value_vec(rowid, embedding) VALUES (?, ?)",
                (new_id, sqlite_vec.serialize_float32(val_emb)),
            )

    conn.commit()
    return new_id


def forget(
    ctx: "MemoryContext",
    key: str,
    kind: str | None = None,
    *,
    now: str | None = None,
) -> int:
    """Soft-delete all LIVE atoms matching *key* (and optional *kind*).

    Returns the count of rows soft-deleted.
    """
    now = now or _now_iso()
    conn = ctx.store.conn

    if kind is not None:
        live_rows = conn.execute(
            "SELECT id, key, value, kind FROM facts WHERE key = ? AND kind = ? AND valid_to IS NULL AND active = 1 AND deleted_at IS NULL",
            (key, kind),
        ).fetchall()
    else:
        live_rows = conn.execute(
            "SELECT id, key, value, kind FROM facts WHERE key = ? AND valid_to IS NULL AND active = 1 AND deleted_at IS NULL",
            (key,),
        ).fetchall()

    if not live_rows:
        return 0

    ids = [r["id"] for r in live_rows]
    placeholders = ",".join(str(i) for i in ids)

    # Soft-delete from facts.
    conn.execute(
        f"UPDATE facts SET deleted_at = ?, active = 0, valid_to = COALESCE(valid_to, ?) WHERE id IN ({placeholders})",
        (now, now),
    )

    # Remove FTS entries.
    for r in live_rows:
        rid, rkey, rval = r["id"], r["key"], r["value"]
        fts_kind = r["kind"]
        conn.execute(
            "INSERT INTO facts_fts(facts_fts, rowid, key, value, kind) VALUES('delete', ?, ?, ?, ?)",
            (rid, rkey, rval, fts_kind),
        )

    # Remove from vec tables.
    if ctx.store.vec_enabled:
        conn.execute(f"DELETE FROM facts_key_vec WHERE rowid IN ({placeholders})")
        conn.execute(f"DELETE FROM facts_value_vec WHERE rowid IN ({placeholders})")

    conn.commit()
    return len(ids)


def recall_facts(ctx: "MemoryContext", query: str, limit: int) -> list[dict]:
    """Return up to *limit* live facts most relevant to *query*.

    Each item is a dict with keys: ``layer``, ``id``, ``kind``, ``key``,
    ``value``, ``valid_from``, ``score``, ``text``, ``anchor_path``,
    ``anchor_symbol``, ``anchor_hash``, ``learned_commit``, ``confidence``,
    ``source``, ``stale``. Hybrid FTS + vector relevance, scored with
    confidence + anchor-liveness (see ``memory.anchor.is_stale``) -- a
    stale/drifted anchor is down-weighted and its ``text`` is labelled, never
    silently trusted.
    """
    now = _now_iso()
    conn = ctx.store.conn

    s_fts_rows: list[tuple[int, float]] = []
    s_vec_rows: dict[int, float] = {}

    # --- FTS leg --------------------------------------------------------
    fts_query = _escape_fts_query(query)
    try:
        fts_results = conn.execute(
            f"SELECT rowid, bm25(facts_fts) AS b FROM facts_fts WHERE facts_fts MATCH ?",
            (fts_query,),
        ).fetchall()

        bm25_values = [r["b"] for r in fts_results]
        if bm25_values:
            b_min = min(bm25_values)  # most negative == best match
            b_max = max(bm25_values)  # least negative == worst match
            for r in fts_results:
                raw = r["b"]
                if b_max == b_min:
                    s_fts = 1.0
                else:
                    # best (raw == b_min) -> 1.0, worst (raw == b_max) -> 0.0
                    s_fts = (b_max - raw) / (b_max - b_min)
                s_fts = max(0.0, min(1.0, float(s_fts)))
                s_fts_rows.append((r["rowid"], s_fts))
    except Exception:
        pass  # FTS leg silently degraded to vector-only on any error.

    # --- Vector leg -----------------------------------------------------
    if ctx.store.vec_enabled and ctx.embedder.available:
        qv = ctx.embedder.embed(query)
        if qv is not None:
            try:
                import sqlite_vec

                vec_bytes = sqlite_vec.serialize_float32(qv)

                vec_results = conn.execute(
                    """
                    SELECT rowid, distance FROM facts_value_vec
                    WHERE embedding MATCH ? ORDER BY distance LIMIT ?
                    """,
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
                        s_vec = max(0.0, min(1.0, s_vec))
                        s_vec_rows[r["rowid"]] = round(s_vec, 6)
            except (ImportError, ModuleNotFoundError):
                pass  # sqlite_vec not available at this moment; skip vector leg.

    # --- Merge & score --------------------------------------------------
    if not s_fts_rows and not s_vec_rows:
        return []

    # Build a full facts map for the live filter + column reads.
    all_ids = set()
    if s_fts_rows:
        all_ids.update(rowid for rowid, _ in s_fts_rows)
    all_ids.update(s_vec_rows.keys())

    placeholders = ",".join(str(i) for i in all_ids)
    facts_map: dict[int, sqlite3.Row] = {
        r["id"]: r
        for r in conn.execute(
            f"""
            SELECT id, kind, key, value, valid_from, salience_floor, d_base,
                   anchor_path, anchor_symbol, anchor_hash, learned_commit,
                   confidence, source
            FROM facts
            WHERE id IN ({placeholders})
              AND valid_to IS NULL AND deleted_at IS NULL AND active = 1
            """,
        ).fetchall()
    }

    s_fts_map = dict(s_fts_rows)
    relevance: dict[int, float] = {}
    for rid in facts_map:
        s_val = max(
            s_fts_map.get(rid, 0.0),
            s_vec_rows.get(rid, 0.0),
        )
        if s_val > 0:
            relevance[rid] = s_val

    if not relevance:
        return []

    # Relative floor.
    max_rel = max(relevance.values())
    if max_rel > 0:
        rel_threshold = 0.15 * max_rel
    else:
        rel_threshold = 0.0
    relevance = {rid: s for rid, s in relevance.items() if s >= rel_threshold}

    # Final score: relevance + confidence + anchor-freshness (§7 rank formula).
    # Anchor-liveness replaces the clock-decay nudge -- staleness is "the code
    # under this insight changed", never "time has passed" (MEMORY_REDESIGN.md
    # §5/§10). The old clock-decay/TTL apparatus (`decay_weight`, `KIND_TTL_DAYS`,
    # `purge_expired`) was removed in M7 -- this scorer never called it anyway.
    # Candidate set only (bounded by the FTS/vec legs above, well under the
    # whole table) and hashes are cached per file so a file anchoring several
    # candidate atoms is only ever read+hashed once for this call.
    hash_cache: dict[str, str | None] = {}
    stale_map: dict[int, bool] = {}
    scored = []
    for rid, rel in relevance.items():
        row = facts_map[rid]
        anchor_path = row["anchor_path"]
        stale = _anchor.is_stale(anchor_path, row["anchor_hash"], cache=hash_cache)
        stale_map[rid] = stale
        fresh = 0.0 if stale else 1.0
        confidence = row["confidence"] if row["confidence"] is not None else 0.5
        final_score = rel + 0.15 * confidence + 0.15 * fresh
        if stale:
            final_score *= 0.4
        scored.append((rid, final_score))

    # Sort desc by score, apply limit.
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:limit]

    # Build result dicts.
    results: list[dict] = []
    ret_ids = []
    for rid, final_score in scored:
        row = facts_map[rid]
        kind_val = row["kind"]
        key_val = row["key"]
        value_val = row["value"]
        valid_from = row["valid_from"] or now
        anchor_path = row["anchor_path"]
        stale = stale_map.get(rid, False)
        text = f"[{kind_val}] {key_val}: {value_val}  (recorded {valid_from[:10]})"
        if stale:
            commit_val = row["learned_commit"]
            when = f" @{commit_val}" if commit_val else ""
            text += f"  ⚠ may be stale (code at {anchor_path} changed since learned{when})"
        results.append({
            "layer": "facts",
            "id": rid,
            "kind": kind_val,
            "key": key_val,
            "value": value_val,
            "valid_from": valid_from,
            "score": round(final_score, 6),
            "text": text,
            "anchor_path": anchor_path,
            "anchor_symbol": row["anchor_symbol"],
            "anchor_hash": row["anchor_hash"],
            "learned_commit": row["learned_commit"],
            "confidence": row["confidence"],
            "source": row["source"],
            "stale": stale,
        })
        ret_ids.append(rid)

    # Best-effort last-accessed-at update.
    if ret_ids:
        ids_pl = ",".join(str(i) for i in ret_ids)
        try:
            conn.execute(
                f"UPDATE facts SET last_accessed_at = ? WHERE id IN ({ids_pl})",
                (now,),
            )
            conn.commit()
        except Exception:
            pass  # Non-critical; ignore failures.

    return results

