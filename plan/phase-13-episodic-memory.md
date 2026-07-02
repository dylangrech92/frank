## Phase 13 — Episodic memory

**Goal:** The "what happened" layer: turn-end gists auto-extracted off-thread and count-gated, reconsolidated in place as work evolves, decaying and evicting over time — recallable across sessions.

**Scope (atomic chunks):**
- `memory/episodic.py`: turn-end encoder — count-gated (**N pinned as a constant, default 12 new transcript rows past a watermark** — lower than Chalie's 20 so live tests reach it fast; tool-heavy coding turns accumulate rows quickly), run off-thread (single-writer queue; **drained/joined on REPL exit** so a quit right after a turn cannot lose the in-flight write); encoder returns `{gist, transcript_ids, has_open_loop, update_id, delete_id}` (prompt adapted from Chalie's `EpisodeEncoderSystemPrompt`); **candidate episodes for update/delete decisions pinned:** the encoder prompt is shown the top-N most-similar non-deleted episodes to the transcript window (embedding via P12) — same top-N pattern as P15's fact extractor; `update_id` → reconsolidation in place; `delete_id` → obsolescence.
- **Salience pinned:** lift Chalie's `salience_service.compute_salience(has_open_loop, novelty)` with novelty from embedding comparison against recent episodes (lift-map row added).
- Populate the P12-created `episodes` table (salience 1–10, timestamps, `transcript_ids` JSON, `has_open_loop`, `deleted_at`); maintain `episodes_fts` + `episodes_vec` on write.
- Erosion: `weight = (salience/10)·exp(−Δt_hours/τ)`, `τ_leaf = 14d`, anchored on `last_relevant_at`, computed lazily at read; eviction rule `weight < 0.05 AND salience ≤ 3 AND age > 90d` (routine callable; wired to session start in P15).
- Register episodes into the P12 registry (recall **and** forget handlers) — `recall` now returns dated gists alongside atoms; `forget` covers episodes with zero edits to P12 code.
- Fill the P1 `episodic.maybe_extract` hook; observability log (ran / reconsolidated / skipped + gate reason).

**Out of scope:** fact mining from gists (P15), flashback (P15). UMAP/HDBSCAN clustering stays dropped (§7.1).

**Dependencies:** P12.

**Live test:**
1. Do a real multi-turn task (`add a /health endpoint and test it`). After the closing answer: extraction log fires once the row-count gate passes; REPL never blocks; sqlite3 shows the episode row with sane gist, salience, `transcript_ids`.
2. Send two one-line turns → gate skips, logged with reason.
3. Keep amending the same feature → encoder emits `update_id`; the existing gist row mutates (reconsolidation), not duplicates.
4. Obsolescence: `we ripped out the health endpoint — that approach is abandoned`, drive enough turns to re-trigger extraction → encoder emits `delete_id`; sqlite3 shows `deleted_at` set on the old episode.
5. Relaunch fresh: `what do you recall about the health endpoint work?` → dated gist returns via `recall` (or its deletion is honestly reflected, per step 4's ordering).
6. `forget` an episode → gone via the registry handler.
7. Decay ordering: create two similar episodes, backdate one heavily via sqlite3 → recall returns the fresh one ranked first (decay affects ordering); backdate one to weight < 0.05 with salience ≤ 3 and age > 90d → run the eviction routine manually → hard-deleted.

**Acceptance criteria:**
- [ ] Extraction is count-gated (N observable), off-thread (observed non-blocking), drained on exit, and writes complete rows with embeddings + FTS.
- [ ] Reconsolidation **and** obsolescence both observed live (steps 3–4).
- [ ] Decay formula affects recall ordering (step 7); eviction routine works.
- [ ] Disk transcript untouched by all of this.

**Definition of done:** standard DoD.
