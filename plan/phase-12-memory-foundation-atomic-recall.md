## Phase 12 — Memory foundation + atomic recall

**Goal:** The agent remembers discrete facts across sessions: per-project SQLite with vector + lexical search, local ONNX embeddings with a working FTS-only fallback, and `remember`/`recall`/`forget` live.

**Scope (atomic chunks):**
- `memory/store.py`: open/migrate `<project>/.coding_agent/memory.db`; sqlite-vec `vec0 float[768]` + FTS5 wiring; WAL, busy_timeout, extension-load pattern per Chalie `database_service.py`; **one migration creates all layer tables now** — `episodes` (**including the `facts_extracted_at` mined-state column** P15's extractor keys on), `facts`, `graph_nodes`, `graph_edges` + FTS + vec companions (§7.1–7.3 schemas) — so P13/P14/P15 add code, not migrations.
- `embedding.py`: **lift Chalie's `embedding_service.py` + `onnx_session.py`** — gte-modernbert-base via onnxruntime (768-d, mean-pooled, offline, first-run auto-download of the ~300 MB model); strip queue/cache singletons; on load failure: one startup note + FTS-only mode flag respected by recall.
- `memory/atomic.py`: `(kind, key, value)` atoms — kinds `project`/`convention`/`discovery`(14d TTL)/`misc`(short TTL); **supersession trigger pinned:** `remember(kind, key, value)` closes (`valid_to = now`) any live row with the same `(kind, key)` before inserting — the remember tool's LLM-facing description instructs key reuse for updates; bi-temporal `valid_from`/`valid_to`; lazy read-time power-law decay `rw = max(salience_floor, max(1, age_days)^(−d_base))`; per-kind TTL hard-purge routine (wired to session boundaries in P15; manually invokable now).
- `memory/recall.py`: **layer registry** where each layer registers a **recall handler and a forget/invalidate handler** (seam #8 — P13/P14 extend both `recall` and `forget` by registration only); hybrid vector+FTS composite scoring (per-lane min-max normalise; relevance = max(vec, fts); + recency + importance; relative floor — adapted from Chalie's episodic retrieval); facts register today.
- `tools/remember.py` (`kind`, `key`, `value`), `tools/recall.py` (`query`), `tools/forget.py` (dispatches through the registry).

**Out of scope:** episodic (P13), graph (P14), flashback/auto-extraction (P15).

**Dependencies:** P1 only (deliberately independent of P11).

**Live test:**
1. `remember that this project uses tabs, not spaces` → `remember(kind=convention)`; row + embedding visible via sqlite3.
2. Quit, relaunch (fresh transcript): `what do you recall about formatting conventions here?` → atom returns with kind and date.
3. Paraphrase with zero keyword overlap (`how do we indent here?`) → still recalled (vector leg proven).
4. `actually we switched to spaces — remember that` (same key) → recall returns the new atom; sqlite3 shows `valid_to` set on the old row (keyed supersession proven).
5. `forget the indentation fact` → gone from recall.
6. Backdate a `discovery` fact 20 days via sqlite3, invoke the TTL purge routine manually → row hard-purged (formula + routine proven now, boundary wiring in P15).
7. Move the ONNX model aside, relaunch → one FTS-only startup note; exact-term recall still works; paraphrase recall degrades (expected).

**Acceptance criteria:**
- [ ] memory.db created with all layer tables (incl. `episodes.facts_extracted_at`) + vec + FTS in one migration.
- [ ] Hybrid recall works (vector + lexical proven separately); FTS-only fallback proven live.
- [ ] Keyed bi-temporal supersession and forget work; TTL purge proven by manual invocation; decay implemented.
- [ ] All three tools usable purely through conversation.

**Definition of done:** standard DoD.
