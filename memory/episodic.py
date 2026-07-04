"""Episodic memory WRITE path: turn-end gist encoder run off-thread on a single-writer queue,
plus the DB write helpers (store / reconsolidate / soft-delete) it calls.

Read/recall, erosion and eviction live in a later addition to this module.
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GATE_N = 12  # extract only after >= this many new transcript rows past the watermark
EXTRACTION_WINDOW = 40  # max most-recent transcript rows to feed the encoder
TOP_N_CANDIDATES = 10  # most-similar existing episodes shown to the encoder for update/delete
NOVELTY_RECENT_LIMIT = 50  # recent episodes compared against for novelty
SALIENCE_NOVELTY_WEIGHT = 0.6
SALIENCE_OPEN_LOOP_WEIGHT = 0.4

# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Encoder system prompt
# ---------------------------------------------------------------------------

_ENCODER_SYSTEM_PROMPT = """You are an episodic memory encoder. You read a recent transcript window plus any existing memory episodes similar to it, and return a JSON array of snapshots.

Each snapshot summarises a coherent moment in the transcript. One snapshot may span multiple transcript entries.

Shape of each array element:
{
  "gist": "2-4 sentence summary of what happened in this slice",
  "transcript_ids": [id, id, ...],
  "has_open_loop": false,
  "update_id": null,
  "delete_id": null
}

Field rules:
- transcript_ids: the integer ids (shown in brackets in the window) this snapshot draws from.
- has_open_loop: true if this snapshot ends with an unresolved thread -- a commitment to future action, an unanswered question, a task paused mid-flight.

Reconsolidation:
- If a snapshot UPDATES an existing episode you were shown (refines, corrects, or extends it), set "update_id" to that episode's id. Your snapshot replaces it.
- If the transcript makes an existing episode OBSOLETE (the user said it was wrong or abandoned), emit an object with ONLY "delete_id" set to that episode's id and every other field null/empty.
- Otherwise leave both ids null (a new episode).

Return ONLY a JSON array. No preamble, no markdown fences. If nothing meaningful happened, return []."""


# ---------------------------------------------------------------------------
# Salience & novelty (pure code)
# ---------------------------------------------------------------------------


def compute_salience(has_open_loop: bool, novelty: float) -> int:
    """Compute a 1-10 salience score from open-loop status and novelty."""
    open_boost = 1.0 if has_open_loop else 0.0
    raw = SALIENCE_NOVELTY_WEIGHT * float(novelty) + SALIENCE_OPEN_LOOP_WEIGHT * open_boost
    return max(1, min(10, int(round(raw * 10))))


def compute_novelty(gist_emb: list[float] | None, prior_embs: list[list[float]]) -> float:
    """Compute novelty as 1 - max cosine similarity against prior embeddings.

    Embeddings are L2-normalized so cosine == dot product. Returns 1.0 when
    gist_emb is None or prior_embs is empty. Clamped to [0.0, 1.0].
    """
    if gist_emb is None or not prior_embs:
        return 1.0

    max_sim = -1.0
    for emb in prior_embs:
        sim = sum(a * b for a, b in zip(gist_emb, emb))
        if sim > max_sim:
            max_sim = sim

    novelty = 1.0 - max_sim
    return max(0.0, min(1.0, novelty))


# ---------------------------------------------------------------------------
# Transcript window formatting
# ---------------------------------------------------------------------------


def _format_window(window: list[tuple[int, dict]]) -> str:
    """Render a transcript window as labelled lines for the encoder prompt."""
    parts: list[str] = []
    for idx, msg in window:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "assistant" and not content:
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                names = ", ".join(tc.get("name", "?") for tc in tool_calls)
                content = f"(tool calls: {names})"
        parts.append(f"[{idx}] {role}: {content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Candidate lookup (runs on the worker connection)
# ---------------------------------------------------------------------------

def _fetch_candidates(store, embedder, window_text: str) -> list[dict]:
    """Return up to TOP_N_CANDIDATES non-deleted episodes most similar to *window_text*.

    Falls back to an empty list when vec or the embedder is unavailable.
    """
    if not store.vec_enabled or not embedder.available:
        return []

    gist_emb = embedder.embed(window_text)
    if gist_emb is None:
        return []

    try:
        import sqlite_vec  # noqa: F811
    except (ImportError, ModuleNotFoundError):
        return []

    vec_bytes = sqlite_vec.serialize_float32(gist_emb)
    rows = store.conn.execute(
        "SELECT rowid FROM episodes_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (vec_bytes, TOP_N_CANDIDATES),
    ).fetchall()

    if not rows:
        return []

    rowids = [r["rowid"] for r in rows]
    placeholders = ",".join(str(r) for r in rowids)
    candidate_rows = store.conn.execute(
        f"SELECT id, gist FROM episodes WHERE rowid IN ({placeholders}) AND deleted_at IS NULL",
    ).fetchall()

    return [{"id": r["id"], "gist": r["gist"]} for r in candidate_rows]


# ---------------------------------------------------------------------------
# Recent-episode gists for novelty
# ---------------------------------------------------------------------------


def _recent_gists(store, limit: int) -> list[str]:
    """Return the *limit* most recent non-deleted episode gist strings."""
    rows = store.conn.execute(
        "SELECT gist FROM episodes WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r["gist"] for r in rows]


# ---------------------------------------------------------------------------
# DB write helpers
# ---------------------------------------------------------------------------


def _store_episode(store, embedder, gist: str, transcript_ids: list[int], has_open_loop: bool, salience: int, gist_emb: list[float] | None) -> str:  # noqa: E501
    """Insert a new episode and its FTS + vec entries. Returns the new episode id."""
    eid = uuid.uuid4().hex
    now = _now_iso()

    store.conn.execute(
        """INSERT INTO episodes(
            id, gist, salience, created_at, last_relevant_at, last_accessed_at,
            transcript_ids, has_open_loop, facts_extracted_at, deleted_at
        ) VALUES(?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL)""",
        (
            eid,
            gist,
            salience,
            now,
            now,
            json.dumps(transcript_ids),
            1 if has_open_loop else 0,
        ),
    )
    rowid = store.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # FTS external-content insert (episodes_fts always exists with the base migration).
    store.conn.execute(
        "INSERT INTO episodes_fts(rowid, gist) VALUES(?, ?)",
        (rowid, gist),
    )

    # Vector insert when available.
    if store.vec_enabled and gist_emb is not None:
        try:
            import sqlite_vec  # noqa: F811

            _vec_ok = True
        except (ImportError, ModuleNotFoundError):
            _vec_ok = False

        if _vec_ok:
            store.conn.execute(
                "INSERT INTO episodes_vec(rowid, embedding) VALUES(?, ?)",
                (rowid, sqlite_vec.serialize_float32(gist_emb)),
            )

    store.conn.commit()
    return eid


def _update_episode(store, embedder, episode_id: str, gist: str, transcript_ids: list[int], has_open_loop: bool, salience: int, gist_emb: list[float] | None) -> bool:  # noqa: E501
    """Reconsolidate an existing episode in place. Returns True on success."""

    row = store.conn.execute(
        "SELECT rowid, gist, transcript_ids FROM episodes WHERE id=? AND deleted_at IS NULL",
        (episode_id,),
    ).fetchone()

    if row is None:
        return False  # caller falls back to _store_episode as a new episode.

    old_rowid = row["rowid"]
    old_gist = row["gist"]
    old_ids_raw = row["transcript_ids"] or "[]"
    old_ids: list[int] = json.loads(old_ids_raw) if old_ids_raw else []
    merged_ids = sorted(set(old_ids) | set(transcript_ids))

    now = _now_iso()

    store.conn.execute(
        """UPDATE episodes SET gist=?, salience=?, last_relevant_at=?, last_accessed_at=?, transcript_ids=?, has_open_loop=? WHERE id=?""",
        (gist, salience, now, now, json.dumps(merged_ids), 1 if has_open_loop else 0, episode_id),
    )

    # FTS: always delete old then insert new.
    store.conn.execute(
        "INSERT INTO episodes_fts(episodes_fts, rowid, gist) VALUES('delete', ?, ?)",
        (old_rowid, old_gist),
    )
    store.conn.execute(
        "INSERT INTO episodes_fts(rowid, gist) VALUES(?, ?)",
        (old_rowid, gist),
    )

    # Vector: replace the stored embedding with the caller-supplied gist_emb.
    if store.vec_enabled and gist_emb is not None:
        try:
            import sqlite_vec  # noqa: F811

            _vec_ok = True
        except (ImportError, ModuleNotFoundError):
            _vec_ok = False

        if _vec_ok:
            store.conn.execute(
                "DELETE FROM episodes_vec WHERE rowid=?",
                (old_rowid,),
            )
            store.conn.execute(
                "INSERT INTO episodes_vec(rowid, embedding) VALUES(?, ?)",
                (old_rowid, sqlite_vec.serialize_float32(gist_emb)),
            )

    store.conn.commit()
    return True


def _soft_delete(store, episode_id: str) -> bool:
    """Soft-delete an existing episode. Returns True on success."""

    row = store.conn.execute(
        "SELECT rowid, gist FROM episodes WHERE id=? AND deleted_at IS NULL",
        (episode_id,),
    ).fetchone()

    if row is None:
        return False

    now = _now_iso()
    old_rowid = row["rowid"]
    old_gist = row["gist"]

    store.conn.execute(
        "UPDATE episodes SET deleted_at=? WHERE id=?",
        (now, episode_id),
    )

    # FTS delete.
    store.conn.execute(
        "INSERT INTO episodes_fts(episodes_fts, rowid, gist) VALUES('delete', ?, ?)",
        (old_rowid, old_gist),
    )

    # Vector cleanup when available (a plain DELETE needs no sqlite_vec module).
    if store.vec_enabled:
        store.conn.execute(
            "DELETE FROM episodes_vec WHERE rowid=?",
            (old_rowid,),
        )

    store.conn.commit()
    return True


# ---------------------------------------------------------------------------
# Safety-net JSON parser for LLM output
# ---------------------------------------------------------------------------


def _repair_json(text: str) -> str | None:
    """Repair a truncated JSON array/object by closing brackets opened outside of
    strings (after dropping a dangling trailing comma). Returns None when nothing
    needs closing."""
    stack: list[str] = []
    in_string = False
    escape = False
    backslash = chr(92)
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == backslash:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "[{":
            stack.append(ch)
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
    if not stack:
        return None
    repaired = text.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1].rstrip()
    closers = {"[": "]", "{": "}"}
    for opener in reversed(stack):
        repaired += closers[opener]
    return repaired


def _safe_json_array(raw: str) -> list:
    """Best-effort parse of an LLM JSON array response.

    Tolerant of reasoning-model output: drops a leading think block, strips prose
    and markdown fences around the JSON, isolates the first array/object, and
    repairs a truncated (unclosed) array before parsing.
    """
    text = raw.strip()

    # Drop a reasoning-model think block: keep only what follows the closing tag.
    close_tag = chr(60)+"/think"+chr(62)
    if close_tag in text:
        text = text.rsplit(close_tag, 1)[-1].strip()

    # Strip ```json / ``` fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        newline = chr(10)
        for i in range(1, len(lines)):
            if lines[i].strip().startswith("```"):
                text = newline.join(lines[1:i])
                break
        else:
            text = newline.join(lines[1:])
        text = text.strip()

    # Isolate the JSON payload from the first '[' or '{' onward (also drops any
    # unclosed think block or prose preamble).
    start = -1
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start == -1:
        return []
    text = text[start:].strip()

    # Try a strict parse first, then a bracket-repaired variant.
    for candidate in (text, _repair_json(text)):
        if candidate is None:
            continue
        try:
            result = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return result
        return []
    return []


# ---------------------------------------------------------------------------
# Core encode routine (runs ON the worker thread)
# ---------------------------------------------------------------------------


def _encode_and_store(project_root: str, window: list[tuple[int, dict]], client) -> dict:
    """Run the episodic encoder on *window* and persist results.

    Returns a small observability dict:
    ``{"ran": bool, "stored": int, "updated": int, "deleted": int, "reason": str}``.
    """
    # 1. Open a FRESH store on THIS thread (never share the main-thread conn).
    from memory.embedding import EmbeddingService
    from memory.store import open_store

    worker_key = (threading.get_ident(), os.path.abspath(project_root))
    if worker_key in _cache:
        store, embedder = _cache[worker_key]
    else:
        store = open_store(project_root)
        model_path = os.environ.get("CODING_AGENT_EMBED_MODEL")
        embedder = EmbeddingService(model_path=model_path)
        _cache[worker_key] = (store, embedder)

    # 2. Format window + collect transcript ids.
    window_text = _format_window(window)
    window_ids = {idx for idx, _ in window}

    # 3. Candidate lookup.
    candidates = _fetch_candidates(store, embedder, window_text)

    # 4. Build the user message.
    user_msg = f"Transcript window:\n{window_text}"
    if candidates:
        cand_lines = "\n".join(f"[{c['id']}] {c['gist']}" for c in candidates)
        user_msg += f"\n\nExisting episodes similar to this window (candidates for update / delete):\n{cand_lines}"

    messages = [
        {"role": "system", "content": _ENCODER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # 5. Call the LLM.
    try:
        resp = client.chat(messages, tools=None)
        raw = resp.text
    except Exception as exc:
        return {"ran": False, "stored": 0, "updated": 0, "deleted": 0, "reason": f"llm-error: {exc}"}

    # 6. Parse snapshots.
    snapshots = _safe_json_array(raw)
    if not snapshots:
        return {"ran": True, "stored": 0, "updated": 0, "deleted": 0, "reason": "empty"}

    # 7. Gather prior embeddings for novelty computation.
    prior_embs = [
        e for e in (embedder.embed(g) for g in _recent_gists(store, NOVELTY_RECENT_LIMIT))
        if e is not None
    ]

    # 8. Process each snapshot.
    stored = updated = deleted = 0
    for snap in snapshots:
        # Delete-only check.
        delete_id = snap.get("delete_id")
        if delete_id and not snap.get("gist") and not snap.get("transcript_ids") and not snap.get("update_id"):
            if _soft_delete(store, str(delete_id)):
                deleted += 1
            continue

        gist = str(snap.get("gist", "")).strip()
        if not gist:
            continue

        valid_ids = [i for i in (snap.get("transcript_ids") or []) if isinstance(i, int) and i in window_ids]
        if not valid_ids:
            continue

        has_open_loop = bool(snap.get("has_open_loop"))
        gist_emb = embedder.embed(gist) if embedder.available else None
        novelty = compute_novelty(gist_emb, prior_embs)
        salience = compute_salience(has_open_loop, novelty)
        update_id = snap.get("update_id")

        if update_id and _update_episode(store, embedder, str(update_id), gist, valid_ids, has_open_loop, salience, gist_emb):  # noqa: E501
            updated += 1
        else:
            _store_episode(store, embedder, gist, valid_ids, has_open_loop, salience, gist_emb)
            stored += 1

        if gist_emb is not None:
            prior_embs.append(gist_emb)

    # 9. Return observability (the cached store/embedder persist on this worker thread).
    return {"ran": True, "stored": stored, "updated": updated, "deleted": deleted, "reason": "ok"}


# ---------------------------------------------------------------------------
# Per-worker connection/embedder cache
# ---------------------------------------------------------------------------

_cache: dict[tuple[int, str], tuple] = {}


# ---------------------------------------------------------------------------
# Single-writer queue
# ---------------------------------------------------------------------------

_queue: "queue.Queue" = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_last_run: dict | None = None
_last_run_lock = threading.Lock()


def _ensure_worker() -> None:
    """Start the episodic writer worker if not already running."""
    global _worker

    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_drain, name="episodic-writer", daemon=True)
            _worker.start()


def _drain() -> None:
    """Drain items from the queue and run the encoder on each."""
    while True:
        item = _queue.get()
        try:
            project_root, window, client = item
            result = _encode_and_store(project_root, window, client)
            with _last_run_lock:
                global _last_run
                _last_run = result
        except Exception:
            pass  # never die from the worker thread.
        finally:
            _queue.task_done()


def enqueue(project_root: str, window: list[tuple[int, dict]], client) -> None:
    """Enqueue a turn-end transcript window for episodic encoding."""
    _ensure_worker()
    _queue.put((project_root, window, client))


def drain_and_join(timeout: float | None = None) -> None:
    """Block until the queue is empty and all tasks are done."""
    try:
        _queue.join()
    except Exception:
        pass


def pop_last_run() -> dict | None:
    """Return (and clear) the most recent observability dict from the worker."""
    global _last_run
    with _last_run_lock:
        result = _last_run
        _last_run = None
        return result

# ---------------------------------------------------------------------------
# Erosion / eviction constants
# ---------------------------------------------------------------------------

TAU_LEAF_HOURS = 14 * 24        # erosion time-constant: 14 days in hours
EVICT_WEIGHT_MAX = 0.05          # eviction threshold on eroded weight
EVICT_SALIENCE_MAX = 3           # eviction threshold on salience
EVICT_AGE_DAYS_MIN = 90          # eviction minimum age in days

# ---------------------------------------------------------------------------
# Time helpers (mirror atomic.py _parse_iso / _age_days)
# ---------------------------------------------------------------------------


def _parse_iso(s: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string to a ``datetime``."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _hours_between(earlier: str, now: str) -> float:
    """Return the floating-point hours between *earlier* and *now*. Clamps negative deltas to 0.0."""
    delta = _parse_iso(now) - _parse_iso(earlier)
    total = delta.total_seconds() / 3600
    return max(0.0, total)


def erosion_weight(salience: int, last_relevant_at: str, now: str | None = None) -> float:
    """weight = (salience/10) * exp(-delta_hours / TAU_LEAF_HOURS), anchored on last_relevant_at."""
    import math

    if now is None:
        now = _now_iso()
    dt_hours = _hours_between(last_relevant_at, now)
    return (salience / 10.0) * math.exp(-dt_hours / TAU_LEAF_HOURS)


def evict_episodes(store, now: str | None = None) -> int:
    """Hard-delete episodes with erosion_weight < EVICT_WEIGHT_MAX AND salience <= EVICT_SALIENCE_MAX
    AND age (created_at) > EVICT_AGE_DAYS_MIN days. Cleans FTS + vec rows too. Returns count evicted.

    Only considers non-deleted episodes for the weight/salience test, but should also be able to purge
    already-soft-deleted ancient rows — KEEP IT SIMPLE: operate on all rows where deleted_at IS NULL,
    matching the three thresholds.
    """
    if now is None:
        now = _now_iso()

    conn = store.conn

    evicted = 0
    while True:
        # Fetch candidates where deleted_at IS NULL (simplest: batch).
        rows = conn.execute(
            "SELECT rowid, id, salience, last_relevant_at, created_at, gist FROM episodes WHERE deleted_at IS NULL",
        ).fetchall()

        if not rows:
            break

        to_delete_rowids: list[int] = []

        for r in rows:
            rid = r["rowid"]
            salience = r["salience"] or 0
            anchor = r["last_relevant_at"] or r["created_at"]
            weight = erosion_weight(salience, anchor, now)

            created = r["created_at"]
            age_days = _hours_between(created, now) / 24.0 if created else 0.0

            if weight < EVICT_WEIGHT_MAX and salience <= EVICT_SALIENCE_MAX and age_days > EVICT_AGE_DAYS_MIN:
                to_delete_rowids.append(rid)

        if not to_delete_rowids:
            break

        rowid_placeholders = ",".join(str(r) for r in to_delete_rowids)

        # Need gists for FTS delete markers.
        details = conn.execute(
            f"SELECT rowid, id, gist FROM episodes WHERE rowid IN ({rowid_placeholders})",
        ).fetchall()

        gist_map: dict[int, str] = {}
        for dr in details:
            gist_map[dr["rowid"]] = dr["gist"]

        # Hard-delete from episodes.
        conn.execute(f"DELETE FROM episodes WHERE rowid IN ({rowid_placeholders})")

        # Remove from FTS.
        for rid, gist in gist_map.items():
            conn.execute(
                "INSERT INTO episodes_fts(episodes_fts, rowid, gist) VALUES('delete', ?, ?)",
                (rid, gist),
            )

        # Remove from vec when available.
        if store.vec_enabled:
            conn.execute(f"DELETE FROM episodes_vec WHERE rowid IN ({rowid_placeholders})")

        evicted += len(to_delete_rowids)
        conn.commit()

    return evicted


# ---------------------------------------------------------------------------
# Recall / forget layer functions
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
        terms = [t.strip() for t in query.split() if t.strip()]
        return " ".join(f'"{t}"' for t in terms)


def recall_episodes(ctx, query: str, limit: int) -> list[dict]:
    """Return up to *limit* non-deleted episodes most relevant to *query*, hybrid FTS + vector,
    with an erosion nudge on ordering (fresher/higher-salience ranks higher).

    Mirrors atomic.recall_facts structure:
      - FTS leg: SELECT rowid, bm25(episodes_fts) AS b FROM episodes_fts WHERE episodes_fts MATCH ?
        (escape the query like atomic._escape_fts_query).
        Normalize bm25 best(most negative)->1.0, worst->0.0.
      - Vector leg (if store.vec_enabled and embedder.available): embed(query); MATCH episodes_vec ORDER BY
        distance LIMIT limit*4; normalize distance to [0,1] like recall_facts.
      - Merge: relevance[rowid] = max(s_fts, s_vec). Build a map of non-deleted episodes for the union of
        rowids: SELECT rowid, id, gist, salience, created_at, last_relevant_at FROM episodes
        WHERE rowid IN (...) AND deleted_at IS NULL. Apply relative floor (0.15 * max) like recall_facts.
      - Final score: final = relevance + 0.25 * erosion_weight(salience, last_relevant_at or created_at, now).
        This makes decay affect ordering.
      - Sort desc, take top ``limit``. Best-effort UPDATE episodes SET last_accessed_at = now WHERE rowid IN (...).
    Each result dict (uniform with facts layer):
        {"layer": "episodic", "id": <episode.id TEXT>, "kind": "episode", "key": <episode.id>,
         "value": gist, "valid_from": created_at, "score": round(final,6),
         "text": f"[episode] {gist}  (recorded {created_at[:10]})"}
    Return [] when nothing matches. Swallow FTS/vec leg errors like recall_facts does.
    """
    now = _now_iso()
    conn = ctx.store.conn

    s_fts_rows: list[tuple[int, float]] = []
    s_vec_rows: dict[int, float] = {}

    # --- FTS leg --------------------------------------------------------
    fts_query = _escape_fts_query(query)
    try:
        fts_results = conn.execute(
            "SELECT rowid, bm25(episodes_fts) AS b FROM episodes_fts WHERE episodes_fts MATCH ?",
            (fts_query,),
        ).fetchall()

        bm25_values = [r["b"] for r in fts_results]
        if bm25_values:
            b_min = min(bm25_values)
            b_max = max(bm25_values)
            for r in fts_results:
                raw = r["b"]
                if b_max == b_min:
                    s_fts_val = 1.0
                else:
                    s_fts_val = (b_max - raw) / (b_max - b_min)
                s_fts_val = max(0.0, min(1.0, float(s_fts_val)))
                s_fts_rows.append((r["rowid"], s_fts_val))
    except Exception:
        pass

    # --- Vector leg -----------------------------------------------------
    if ctx.store.vec_enabled and ctx.embedder.available:
        qv = ctx.embedder.embed(query)
        if qv is not None:
            try:
                import sqlite_vec

                vec_bytes = sqlite_vec.serialize_float32(qv)

                vec_results = conn.execute(
                    """
                    SELECT rowid, distance FROM episodes_vec
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
                            s_vec_val = 1.0 / (1.0 + abs(dist))
                        else:
                            normalised = (dist - d_min) / (d_max - d_min)
                            s_vec_val = 1.0 - float(normalised) * 0.999
                        s_vec_val = max(0.0, min(1.0, s_vec_val))
                        s_vec_rows[r["rowid"]] = round(s_vec_val, 6)
            except (ImportError, ModuleNotFoundError):
                pass

    # --- Merge & score --------------------------------------------------
    if not s_fts_rows and not s_vec_rows:
        return []

    all_rowids = set()
    if s_fts_rows:
        all_rowids.update(rowid for rowid, _ in s_fts_rows)
    all_rowids.update(s_vec_rows.keys())

    placeholders = ",".join(str(i) for i in all_rowids)
    episode_rows = conn.execute(
        f"""
        SELECT rowid, id, gist, salience, created_at, last_relevant_at
        FROM episodes
        WHERE rowid IN ({placeholders})
          AND deleted_at IS NULL
        """,
    ).fetchall()

    ep_map: dict[int, sqlite3.Row] = {r["rowid"]: r for r in episode_rows}

    s_fts_map = dict(s_fts_rows)
    relevance: dict[int, float] = {}
    for rowid in ep_map:
        s_val = max(
            s_fts_map.get(rowid, 0.0),
            s_vec_rows.get(rowid, 0.0),
        )
        if s_val > 0:
            relevance[rowid] = s_val

    if not relevance:
        return []

    # Relative floor.
    max_rel = max(relevance.values())
    if max_rel > 0:
        rel_threshold = 0.15 * max_rel
    else:
        rel_threshold = 0.0
    relevance = {rid: s for rid, s in relevance.items() if s >= rel_threshold}

    # Final score with erosion nudge.
    scored: list[tuple[int, int, float, str, str | None]] = []
    for rowid, rel in relevance.items():
        row = ep_map[rowid]
        rid = row["rowid"]
        eid = row["id"]
        gist_val = row["gist"] or ""
        salience = row["salience"] or 0
        created_at = row["created_at"] or now
        anchor = row["last_relevant_at"] or created_at
        erosion = erosion_weight(salience, anchor, now)
        final = rel + 0.25 * erosion
        scored.append((rid, eid, final, gist_val, created_at))

    scored.sort(key=lambda x: x[2], reverse=True)
    scored = scored[:limit]

    # Build result dicts.
    results: list[dict] = []
    ret_ids: list[str] = []
    for rid, eid, final_score, gist_val, created_at in scored:
        text = f"[episode] {gist_val}  (recorded {created_at[:10]})"
        results.append({
            "layer": "episodic",
            "id": eid,
            "kind": "episode",
            "key": eid,
            "value": gist_val,
            "valid_from": created_at,
            "score": round(final_score, 6),
            "text": text,
        })
        ret_ids.append(eid)

    # Best-effort last_accessed_at update.
    if ret_ids:
        ep_placeholders = ",".join("?" for _ in ret_ids)
        try:
            conn.execute(
                f"UPDATE episodes SET last_accessed_at = ? WHERE id IN ({ep_placeholders})",
                [now] + ret_ids,
            )
            conn.commit()
        except Exception:
            pass

    return results


def forget_episodes(ctx, key: str, kind: str | None) -> int:
    """Soft-delete episode(s) by id. The registry's forget passes the user's `key` as the episode id
    (episodes have no separate key). If `kind` is provided and is not "episode", return 0 (this layer
    only owns kind 'episode'; a None kind means 'any layer' and DOES apply here).
    Look up a non-deleted episode WHERE id = key; if found, call the existing _soft_delete(store, key)
    and return 1; else return 0. (Reuse the module's _soft_delete.)"""
    if kind is not None and kind != "episode":
        return 0
    conn = ctx.store.conn
    row = conn.execute(
        "SELECT id FROM episodes WHERE id = ? AND deleted_at IS NULL",
        (key,),
    ).fetchone()
    if row is None:
        return 0
    _soft_delete(ctx.store, key)
    return 1


# ---------------------------------------------------------------------------
# Lazy idempotent registration
# ---------------------------------------------------------------------------

_registered = False


def register_episodic_layer() -> None:
    """Register the episodic recall/forget layer with memory.recall (if available)."""
    global _registered

    if _registered:
        return

    try:
        from memory.recall import register_layer  # type: ignore[attr-defined]

        register_layer("episodic", recall_episodes, forget_episodes)
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    _registered = True


# Also attempt registration at import time (best-effort).
try:
    register_episodic_layer()
except Exception:
    pass
