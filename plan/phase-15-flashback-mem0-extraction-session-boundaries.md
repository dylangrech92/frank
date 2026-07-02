## Phase 15 — Flashback, Mem0 extraction & session boundaries

**Goal:** Memory becomes ambient: gated turn-0 flashback seeds each new topic with decisions, gists, and atoms **on top of the always-present rules**; a Mem0-style extractor mines episode gists into facts at session boundaries; the full §7.4 CLI-adapted "subconscious" is complete.

**Scope (atomic chunks):**
- `memory/flashback.py`: turn-0 seed filling the last P1 agent-loop hook, with Chalie's two zero-LLM gates (`turn_zero_flashback.py`): terse gate (< 8 whitespace tokens, **checked before** the continuation gate) and continuation gate (message embedding cosine ≥ 0.55 vs centroid of the last 6 conversation messages — starting threshold valid because we use the same embedder; fails open to seeding); FTS-only mode degrades the continuation gate to terse-only.
- Flashback block renderer — renders the **gated bundle only**: top similarity-recalled Decisions/Specs (1-hop expanded) + ≤3 dated episode gists + ≤5 atoms. **Rules are NOT part of the gated bundle:** the P14 rules provider stays unconditional in every assembled context (P14's acceptance criterion keeps holding); the renderer *calls* P14's standalone rules-render function only to deduplicate — a gate-skip means "no recalled-memory bundle", never "no rules".
- Mem0-style auto-extractor in `memory/atomic.py`: mines episodes where `facts_extracted_at IS NULL` (the P12 column) into facts — shown top-N most-similar existing facts, decides ADD / UPDATE / DELETE / NOOP (unparseable output → counted NOOP, never a write); **op envelope extended with a `kind` field** (the four §7.2 kinds described in the prompt; default `project`/`convention` for durable auto-mined facts — never `discovery`, whose 14d TTL would silently expire them); UPDATE/DELETE ride P12's keyed bi-temporal supersession; stamps `facts_extracted_at` on processed episodes.
- Session-boundary jobs filling P1's `session_start_jobs`/`session_end_jobs` hooks (§7.4, no daemon/no cron): start — episodic eviction + per-kind TTL purge; start/end — gist-to-fact extraction sweep for unmined episodes.
- Observability: gate outcomes logged (seeded / terse-skip / continuation-skip); every extractor op decision logged.

**Out of scope:** §15 list (super-episode roll-up, auto-mined graph suggestions, cross-project memory, session resume, non-OpenAI providers). Whole-system sweep → P16.

**Dependencies:** P13, P14.

**Live test:**
1. Do meaningful work in a session (establish visible conventions), quit → boundary sweep logs extractor decisions; facts appear that were never explicitly `remember`-ed (e.g. "tests run via pytest"), with durable kinds; sqlite3 shows `facts_extracted_at` stamped.
2. Relaunch; type `hi` → gate log shows terse-skip; `--verbose` shows **no recalled-memory bundle but the rules block still present**.
3. `let's keep hardening the fib module against bad input` → flashback bundle appears exactly once: a decision + ≤3 dated gists + ≤5 atoms (rules present as always, not duplicated); the reply visibly uses them with no recall tool call.
4. Immediate follow-up, ≥8 tokens so the terse gate passes and the **continuation gate** is what fires: `also make sure the fib module rejects negative numbers and floats` → no re-injection; gate log shows continuation-skip.
5. Tell it a changed preference, end the session → extractor logs an UPDATE; old fact's `valid_to` set; recall returns only the new one.
6. Backdate a `discovery` fact 20 days and a low-salience episode past the eviction thresholds via sqlite3, relaunch → session-start jobs purge/evict both, logged.

**Acceptance criteria:**
- [ ] Both gates observed firing and skipping correctly (terse in step 2, continuation in step 4); bundle respects the ≤3/≤5 caps; rules remain in every context regardless of gates.
- [ ] Extractor produces ADD/UPDATE/NOOP decisions live with correct kinds; never writes on unparseable output; DELETE invalidates bi-temporally; mined episodes stamped.
- [ ] Boundary jobs run at start/end via the P1 hooks with visible logs; no daemon, no cron.
- [ ] The §9 loop is now filled verbatim — zero remaining no-op hooks.

**Definition of done:** standard DoD.
