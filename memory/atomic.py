"""Atomic-facts memory layer: CRUD + decay for the facts table.

Provides a structured ``remember / forget / recall`` API over the
``facts`` table (bi-temporal keyed supersession), plus TTL-based
expiration and power-law retrieval-weight decay.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KINDS = ("project", "convention", "discovery", "misc")

KIND_TTL_DAYS: dict[str, int | None] = {
    "project": None,
    "convention": None,
    "discovery": 14,
    "misc": 3,
}

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


def _parse_iso(s: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string to a ``datetime``."""
    # Handle 'Z' suffix and optional microseconds.
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _age_days(valid_from: str, now: str) -> float:
    """Return the age in days (float) between *valid_from* and *now*."""
    delta = _parse_iso(now) - _parse_iso(valid_from)
    return delta.total_seconds() / 86400


# ---------------------------------------------------------------------------
# FTS query escaping helper
# ---------------------------------------------------------------------------


def _escape_fts_query(query: str) -> str:
    """Escape *query* for use in an FTS5 ``MATCH`` expression.

    Wraps the entire query in double-quotes so terms are matched literally,
    and any embedded double-quotes inside the query are escaped as ``""``.
    Falls back to raw term OR-matching if quoting fails.
    """
    try:
        escaped = query.replace('"', '""')
        return f'"{escaped}"'
    except Exception:
        # Fallback: join individual words with OR (least precise but safe).
        terms = [t.strip() for t in query.split() if t.strip()]
        return " ".join(f'"{t}"' for t in terms)


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
) -> int:
    """Insert a new atom, superseding any current one with the same (kind, key).

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
            valid_from, valid_to, active, deleted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1, NULL)
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


def purge_expired(ctx: "MemoryContext", *, now: str | None = None) -> int:
    """Hard-delete atoms whose kind TTL has elapsed from ``valid_from``.

    Returns the count of rows purged.
    """
    now = now or _now_iso()
    conn = ctx.store.conn

    to_purge: list[int] = []

    for kind, ttl in KIND_TTL_DAYS.items():
        if ttl is None:
            continue
        expired = conn.execute(
            """
            SELECT id FROM facts
            WHERE kind = ? AND valid_from IS NOT NULL
              AND CAST((julianday(?) - julianday(valid_from)) AS INTEGER) > ?
            """,
            (kind, now, ttl),
        ).fetchall()
        to_purge.extend(r["id"] for r in expired)

    if not to_purge:
        return 0

    ids = to_purge
    placeholders = ",".join(str(i) for i in ids)

    # Gather facts rows before deleting (need key/value/kind for FTS cleanup).
    facts_rows = conn.execute(
        f"SELECT id, key, value, kind FROM facts WHERE id IN ({placeholders})",
    ).fetchall()

    # Hard-delete from facts.
    conn.execute(f"DELETE FROM facts WHERE id IN ({placeholders})")

    # Clean up FTS.
    for r in facts_rows:
        rid, rkey, rval, rkind = r["id"], r["key"], r["value"], r["kind"]
        conn.execute(
            "INSERT INTO facts_fts(facts_fts, rowid, key, value, kind) VALUES('delete', ?, ?, ?, ?)",
            (rid, rkey, rval, rkind),
        )

    # Clean up vec tables.
    if ctx.store.vec_enabled:
        conn.execute(f"DELETE FROM facts_key_vec WHERE rowid IN ({placeholders})")
        conn.execute(f"DELETE FROM facts_value_vec WHERE rowid IN ({placeholders})")

    conn.commit()
    return len(ids)


def decay_weight(
    salience_floor: float,
    d_base: float,
    valid_from: str,
    now: str | None = None,
) -> float:
    """Power-law retrieval-weight decay read-time helper.

    ``rw = max(salience_floor, max(1, age_days) ** (-d_base))``.
    Never raises on age=0.
    """
    now = now or _now_iso()
    age = _age_days(valid_from, now)
    if age < 0:
        age = 0.0
    return max(salience_floor, max(1, age) ** (-d_base))


def recall_facts(ctx: "MemoryContext", query: str, limit: int) -> list[dict]:
    """Return up to *limit* live facts most relevant to *query*.

    Each item is a dict with keys: ``layer``, ``id``, ``kind``, ``key``,
    ``value``, ``valid_from``, ``score``, ``text``.
    Hybrid FTS + vector scoring with power-law recency nudge.
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
            SELECT id, kind, key, value, valid_from, salience_floor, d_base
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

    # Final score with decay nudge.
    scored = []
    for rid, rel in relevance.items():
        row = facts_map[rid]
        dw = decay_weight(row["salience_floor"], row["d_base"], row["valid_from"], now)
        final_score = rel + 0.25 * dw
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
        text = f"[{kind_val}] {key_val}: {value_val}  (recorded {valid_from[:10]})"
        results.append({
            "layer": "facts",
            "id": rid,
            "kind": kind_val,
            "key": key_val,
            "value": value_val,
            "valid_from": valid_from,
            "score": round(final_score, 6),
            "text": text,
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


# ---------------------------------------------------------------------------
# Mem0-style auto-extractor: mine episode gists into durable facts
# ---------------------------------------------------------------------------

_EXTRACTOR_SYSTEM_PROMPT = """You maintain a project's long-term FACTS from what happened in a work session.

You are given ONE episode gist (a short summary of a slice of work) and the existing facts most similar to it. Decide what durable, reusable facts about THIS project the gist establishes, and return operations against the facts store.

Return ONLY a JSON array of operation objects (no prose, no markdown fences). Each object:
{
  "op": "ADD" | "UPDATE" | "DELETE" | "NOOP",
  "kind": "project" | "convention",
  "key": "short-stable-identifier",
  "value": "the fact, stated atomically"
}

Rules:
- ADD: a new durable fact not already present (e.g. "tests run via pytest", "web layer uses FastAPI").
- UPDATE: the gist refines/corrects an existing fact -- reuse that fact's EXACT key so it supersedes the old value.
- DELETE: an existing fact is now wrong/abandoned -- give its key (value may be empty).
- NOOP: nothing durable to record (transient chatter, already-known facts). Return [{"op":"NOOP"}] or [].
- kind is ALWAYS "project" (facts about this codebase) or "convention" (observed coding conventions). NEVER anything else.
- Keys are short, stable, kebab-or-snake identifiers you can reuse later (e.g. "test-runner", "api-framework"). Reuse an existing key when refining it.
- Prefer FEW high-value facts over many trivial ones. Do not restate facts already shown as existing unless correcting them."""


_EXTRACTOR_KINDS = ("project", "convention")


def _coerce_extractor_kind(raw) -> str:
    """Clamp an auto-mined fact kind to a durable kind (never discovery/misc TTLs)."""
    if isinstance(raw, str) and raw.strip().lower() in _EXTRACTOR_KINDS:
        return raw.strip().lower()
    return "convention"


def _unmined_episodes(conn, limit: int) -> list:
    """Return up to *limit* non-deleted episodes with no facts_extracted_at stamp."""
    return conn.execute(
        """
        SELECT id, gist FROM episodes
        WHERE facts_extracted_at IS NULL AND deleted_at IS NULL
        ORDER BY created_at
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def extract_facts(ctx: "MemoryContext", client, *, limit: int = 25, now: str | None = None) -> dict:
    """Mine unmined episode gists into durable facts via the LLM (Mem0-style).

    For each episode with ``facts_extracted_at IS NULL``: show the gist plus the
    most-similar existing facts, ask the model for ADD/UPDATE/DELETE/NOOP ops
    (kind clamped to project/convention -- never a TTL kind), apply them through
    the keyed bi-temporal ``remember``/``forget`` path, then stamp
    ``facts_extracted_at`` so the episode is never re-mined. An LLM/parse failure
    on one episode leaves it UNSTAMPED for a later retry and never writes.

    Returns ``{"processed", "added", "updated", "deleted", "noop"}``.
    """
    import sys as _sys

    now = now or _now_iso()
    conn = ctx.store.conn
    from memory.episodic import _safe_json_array

    episodes = _unmined_episodes(conn, limit)
    stats = {"processed": 0, "added": 0, "updated": 0, "deleted": 0, "noop": 0}

    for ep in episodes:
        eid = ep["id"]
        gist = ep["gist"] or ""
        if not gist.strip():
            # Nothing to mine, but stamp so we skip it next time.
            conn.execute("UPDATE episodes SET facts_extracted_at = ? WHERE id = ?", (now, eid))
            conn.commit()
            continue

        # Show the model the most-similar existing facts (for UPDATE/DELETE/dedupe).
        try:
            existing = recall_facts(ctx, gist, 8)
        except Exception:
            existing = []
        existing_block = "\n".join(
            f'- kind={f.get("kind")} key={f.get("key")!r}: {f.get("value")}' for f in existing
        ) or "(none)"

        user_msg = (
            f"Episode gist:\n{gist}\n\n"
            f"Existing facts most similar to this gist:\n{existing_block}"
        )
        messages = [
            {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # LLM call -- on failure, leave the episode UNSTAMPED and move on.
        try:
            resp = client.chat(messages, tools=None)
            raw = resp.text
        except Exception as exc:
            print(f"extractor: episode {str(eid)[:8]} llm-error: {exc}", file=_sys.stderr, flush=True)
            continue

        ops = _safe_json_array(raw)  # unparseable -> [] -> counted as NOOP below

        applied_any = False
        for op in ops:
            if not isinstance(op, dict):
                continue
            action = str(op.get("op", "")).strip().upper()
            key = op.get("key")
            key = key.strip() if isinstance(key, str) else ""
            value = op.get("value")
            value = value if isinstance(value, str) else ""
            kind = _coerce_extractor_kind(op.get("kind"))

            if action in ("ADD", "UPDATE") and key and value.strip():
                # Distinguish add vs update by whether a live row already holds this key.
                pre = conn.execute(
                    "SELECT 1 FROM facts WHERE kind=? AND key=? AND valid_to IS NULL AND active=1 AND deleted_at IS NULL",
                    (kind, key),
                ).fetchone()
                try:
                    remember(ctx, kind, key, value, now=now)
                except Exception as exc:
                    print(f"extractor: episode {str(eid)[:8]} write-error op={action} key={key!r}: {exc}", file=_sys.stderr, flush=True)
                    continue
                if pre is not None:
                    stats["updated"] += 1
                    print(f"extractor: episode {str(eid)[:8]} UPDATE kind={kind} key={key!r}", file=_sys.stderr, flush=True)
                else:
                    stats["added"] += 1
                    print(f"extractor: episode {str(eid)[:8]} ADD kind={kind} key={key!r}", file=_sys.stderr, flush=True)
                applied_any = True
            elif action == "DELETE" and key:
                try:
                    n = forget(ctx, key, kind, now=now)
                except Exception as exc:
                    print(f"extractor: episode {str(eid)[:8]} delete-error key={key!r}: {exc}", file=_sys.stderr, flush=True)
                    continue
                if n:
                    stats["deleted"] += n
                    print(f"extractor: episode {str(eid)[:8]} DELETE kind={kind} key={key!r} ({n})", file=_sys.stderr, flush=True)
                applied_any = True
            else:
                # NOOP or malformed -> no write.
                print(f"extractor: episode {str(eid)[:8]} NOOP", file=_sys.stderr, flush=True)

        if not applied_any:
            stats["noop"] += 1

        # Stamp the episode as mined (success path, even when the decision was NOOP).
        conn.execute("UPDATE episodes SET facts_extracted_at = ? WHERE id = ?", (now, eid))
        conn.commit()
        stats["processed"] += 1

    return stats
