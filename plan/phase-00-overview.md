# coding_agent — Implementation Plan (v3, adversarially verified)

**Source spec:** `DESIGN.md` (repo root, 2026-07-01 draft)
**Home:** `plan/` at the repo root — one file per phase plus this overview; `IMPLEMENTATION_PLAN.md` at the root is the index.
**Shape:** 16 additive phases. Each phase = 1 working day for one competent dev (environment/toolchain setup time counts inside the day), self-encapsulated, validated exclusively by **live-driving the running REPL harness** — no unit tests, no feature tests, no scripted suites anywhere. The seeded sample project's pytest/jest/phpunit suites are *fixtures for the agent to run*, not tests of the harness.
**Scope bullets are deliberately atomic** — each is a candidate "chunk" for the later 1-day-breakup pass.
**Verification:** this plan was adversarially reviewed by 17 independent checkers (14 per-phase day-size skeptics + spec-coverage, additivity, and live-test-realism audits); all 4 blockers, 11 majors, and 55 minors found were fixed in v2. v3 (2026-07-02) resolves the previously-open spec questions as approved decisions (§0.5) — most visibly the `read_file` addition in P2 (catalog → 44 tools).

---

## 0. Ordering rationale & additivity seams

Value-first with risk pulled forward: the agent talks (P1), edits (P2), runs commands (P3), sees compiler truth (P4), navigates (P5), refactors (P6), verifies (P7), debugs (P8–P10), survives long sessions (P11), then remembers (P12–P15), and the whole system is hardened in one final sweep (P16).

Additivity is engineered via seams created once and **filled later, never rewritten**:

1. **Agent-loop hooks (P1):** `agent.py` is written to the exact §9 loop shape from day 1, with four named hook points — `flashback.maybe_seed` (filled P15), `except OverCapError` (P1 behavior: surface a clear over-cap error and end the turn, no retry; P11 replaces the body with compact-then-retry inside the same seam), `diagnostics.inject_summary` (filled P4), `episodic.maybe_extract` (filled P13).
2. **Session-boundary hooks (P1):** `main.py` declares named no-op `session_start_jobs` / `session_end_jobs` hooks, filled by P15 (eviction, TTL purge, fact-extraction sweep).
3. **`assemble_context()` + context-provider registry (P1):** `assemble_context()` exists from day 1 as a pass-through that first consults an (initially empty) **context-provider registry**; providers render blocks appended to the system message, outside the pruning/compaction window. P11 implements pruning inside the same interface and preserves provider output; P14 registers the rules provider.
4. **Shared fs-write helper (P2):** sandbox resolution (`resolve_in_root`) + mutation-event emission (`created/changed/deleted/renamed`) live in **one shared write helper** used by every mutating tool; P4's LSP sync and diagnostics subscribe to the bus; P6's WorkspaceEdit applier and move-hook write through the same helper — no P2 code touched.
5. **`move_file` pre-move hook (P2):** no-op seam filled by P6's `workspace/willRenameFiles`.
6. **Shared JSON-RPC framing module (P4):** Content-Length framing + threaded reader written once for LSP, imported unchanged by DAP in P8.
7. **DAP client/manager seams (P8):** transport abstraction (stdio **and** TCP socket), reverse-request handling (`startDebugging`, `runInTerminal`) with a child-session registry, launch/attach duality, and a **per-language launch-config synthesizer registry** (python registered in P8) — so P9 (Xdebug) and P10 (js-debug) only *register adapters*, touching zero P8 code and zero tools.
8. **`memory/recall.py` layer registry (P12):** each memory layer registers a **recall handler and a forget/invalidate handler**; episodes (P13) and graph nodes (P14) extend `recall`/`forget` by registration only — `tools/forget.py` is never edited after P12.
9. **All memory tables in P12's single migration** — `episodes` (including a `facts_extracted_at` mined-state column consumed by P15's extractor), `facts`, `graph_nodes`, `graph_edges` + FTS + vec companions — so P13/P14/P15 add code, not migrations.
10. **Tool auto-registration (P1):** `registry.py` discovers `Tool` subclasses, so every later phase only *adds files*.
11. **Observability from day 1 (P1):** `--verbose` dumps the exact assembled context per LLM call and every tool call/result is echoed to stderr — this is how every later phase's live test observes injection, pruning, compaction, and flashback.

**File-layout note:** the plan extends the spec's §3 layout with three shared internal helpers — the fs-write/paths helper (P2), the JSON-RPC framing module (P4, directly serving §11's "same framing style as LSP"), and `lsp/edits.py` (P6, required by rename/code_actions/move_file). All three *reduce* total LOC versus duplicating the logic per consumer; they are deliberate, not scope creep.

### 0.1 Live-test conventions (apply to every phase)

- **Scratch-copy rule:** each phase's live test runs the harness against a **fresh scratch copy** of `samples/playground`, `git init`-ed first. Live-test mutations never touch the committed fixture or the harness repo — this is also what makes destructive-git validation safe and gives every phase a deterministic starting state.
- **Guard-verification phrasing:** steps that probe safety rails (deny-list, sandbox, SSRF) instruct the model explicitly — e.g. *"call run_command with cmd `rm -rf /` — I am verifying the deny-list blocks it before execution."* The pass signal is harness-side: the tool-call echo followed by the `ToolResult.err`. If the model refuses to place the call, retry with the explicit instruction; a conversational refusal is **not** a pass.
- **Fixture manifest is pinned** (see §0.4): later phases' live tests reference only files/symbols the manifest guarantees.

### 0.2 Verified Chalie lift map

Reference source: the Chalie backend (`backend/` tree of github.com/chalie-ai/chalie); paths below are repo-relative to `backend/`. All verified on disk 2026-07-02.

| Artifact | Verified source | Used in phase |
|---|---|---|
| `ToolResult.ok/err` frozen contract (status, kebab-case error codes, hint) | `abilities/_result.py` (ToolResult L76, ok L124, err L129) | P1 |
| Tool ABC shape (name/description/parameters/run) | `abilities/_ability.py` (lift the *shape only* — strip the `self.mp` MessageProcessor seam and framework params) | P1 |
| Compaction prompt | `services/system_message_prompt.py` `ChatHistoryCompactionSystemPrompt` L188-214, re-themed to Task/State/Files-touched/Open/Decisions/Last | P11 |
| Over-cap trigger discipline (cap = window − max(0.10·window, 8000); catch size errors → compact → retry) | `services/providers.py` L76-97; `services/message_processor.py` `_step()` L553-583. **Note:** Chalie duplicates the cap formula in multiple places — coding_agent defines it once. | P11 |
| Token estimation | `services/llm_clients/openai.py` L261-287 (tiktoken; counts tool schemas too). **Correction:** Chalie's fallback is words×1.3, *not* chars/4 — the spec's chars/4 heuristic is new code. | P11 |
| ddgs search | `tools/search/fetcher.py` `fetch_ddg_fallback` L298-319 + cooldown/backoff/transform L270-295 (~50 self-contained lines). **Correction:** Chalie's `web_search` *tool* is a delegate sub-agent; lift the fetcher, not the ability. | P7 |
| SSRF guard + fetch + extraction | `services/ssrf.py` (BLOCKED_NETS, resolve_and_check, fail-closed), `services/web_fetch.py`, `services/text_extractor.py` `extract_html`. **Fix on lift:** Chalie defaults `verify=False` — coding_agent enables TLS verification. | P7 |
| Local embeddings | `services/embedding_service.py` + `services/onnx_session.py` — gte-modernbert-base ONNX, mean pooling (`_mean_pool` L179-230), first-run auto-download of the ~300 MB model (L86). Strip the queue/cache singletons. | P12 |
| Turn-0 flashback gates | `services/turn_zero_flashback.py` (terse gate <8 whitespace tokens L86/137, checked **before** the continuation gate L121-128; continuation threshold 0.55 L68 — calibrated to gte-modernbert-base, the same embedder we use, so 0.55 is a valid starting point; centroid over last 6 messages L194-219; block render L232-316; fails open) | P15 |
| Episodic encoder contract | `services/system_message_prompt.py` `EpisodeEncoderSystemPrompt` L113-139 (exact `{gist, transcript_ids, has_open_loop, update_id, delete_id}` shape); apply logic `services/episodic_service.py` L300-445; decay `services/decay_engine_service.py` L148-162 (formula matches spec) | P13 |
| Episode salience | `services/salience_service.py` `compute_salience(has_open_loop, novelty)` — novelty from embedding comparison against recent episodes | P13 |
| Mem0-style fact extractor | `configs/channels/fact_extraction.py` (ADD/UPDATE/DELETE/NOOP ops, top-N similar facts in prompt, unparseable → counted NOOP, never a write; backlog keyed on `episodes.facts_extracted_at IS NULL`); apply loop `services/subconscious_worker.py` L450-547; power-law decay `services/data_graph_service.py` L1530-1543 (formula matches spec char-for-char) | P12 (decay), P15 (extractor) |
| sqlite-vec + FTS5 wiring & hybrid recall scoring | `schema.sql` (vec0 float[768] + fts5 companions), `services/database_service.py` L211-254 (WAL, busy_timeout, extension load), `services/episodic_retrieval_service.py` L436-490 (per-lane min-max normalise; relevance = max(vec, fts); score = relevance + recency + importance; relative floor) | P12 |

**No-lift zones (all-new code, sized accordingly):** the LSP/DAP protocol layers (P4-P6, P8-P10 — Chalie has no LSP/JSON-RPC/DAP code; verified by grep) and the test-runner parsers (P7).

### 0.3 Coverage matrix (spec → phase)

| Spec item | Phase |
|---|---|
| §3 file layout (all modules) | P1–P15 (each module lands exactly once; +3 shared helpers, see note above) |
| §4.1 lifecycle | P1 (LSP pre-warm step lands in P4) |
| §4.2 message format, §4.3 transcript | P1 |
| §4.4 pruned context assembly | P11 |
| §5 Explorer tools + sandbox (incl. the approved `read_file` addition, §0.5.1) | P2 |
| §5 Find/replace tools | P2 |
| §5 Code-intel tools | P4 (hover) + P5 (rest) |
| §5 Refactor/fix tools | P6 |
| §5 `get_diagnostics` + §6 hybrid diagnostics flow | P4 |
| §5 git tool | P3 |
| §5 Terminal tools | P3 |
| §5 `run_tests` | P7 |
| §5 Debug tools | P8 (+P9/P10 adapters, zero tool changes) |
| §5 Web tools + §12 SSRF | P7 |
| §5 Memory tools (`remember/recall/forget`) | P12 (`forget` gains episodes in P13 by registration) |
| §5 Memory tools (`record_*`, `link_nodes`) | P14 |
| §2.1 engine matrix + degradation | P4 (pyright + missing-engine report), P5 (ts-ls, intelephense, lazy start), P6 (html/css servers), P7 (pytest/jest/phpunit + test refusals; vitest = config-only, honestly unvalidated), P8 (debugpy), P9 (Xdebug + debug refusals), P10 (js-debug) |
| §7.1 episodic layer | P13 |
| §7.2 atomic layer + decay | P12 |
| §7.3 data-graph layer | P14 |
| §7.4 CLI integration | P12 (lazy decay), P13 (turn-end off-thread extraction), P15 (boundary jobs + flashback) |
| §8 compaction | P11 |
| §9 agent loop | P1 (shape) → P4/P11/P13/P15 (hooks filled) |
| §10 config.json | grows per phase (llm P1; git P3; language_servers P4; test_runners P7; debug_adapters P8-P10; compaction P11) |
| §12 safety (sandbox, deny-list, destructive-git gate, SSRF) | P2, P3, P3, P7 |
| §15 out-of-scope list | stays out of every phase |

Tool-count check: P1:1, P2:9, P3:4, P4:2, P5:8, P6:3, P7:3, P8:6, P12:3, P14:5 = **44 tools** — the spec's full §5 catalog, including the `read_file` addition approved and folded into the spec 2026-07-02 (§0.5.1). P9/P10/P11/P13/P15/P16 add zero tools by design.

### 0.4 Pinned playground manifest

`samples/playground` is the committed live-test fixture. It grows in three installments (each phase only **adds** files); live tests always run against a scratch copy (§0.1).

- **P1 (Python + web slice):** `app.py` (Flask, exposes `create_app`, carries one deliberately seeded unused import), `lib/fib.py` (`fibonacci()`), `tests/test_fib.py` (imports `lib.fib`; one deliberately failing test), `bin/run_fib.py` (plain runnable driver that imports and calls `fibonacci` — the P8 debug target), `index.html`, `styles.scss`. Python has a small class hierarchy (base class + override) so implementation/type-definition/call-hierarchy queries have targets.
- **P5 (JS + PHP source slice):** `js/order.js` (exports `parseOrder`; small class hierarchy), `js/index.js` (runnable driver — the P10 debug target), `php/Cart.php` (`Cart::total`), `php/bin/run.php` (runnable driver — the P9 debug target), `php/index.php`.
- **P7 (test-suite slice):** `js/` jest config + one deliberately failing jest test; `php/` phpunit.xml + one deliberately failing phpunit test (composer/phpunit + node/jest toolchain setup happens here, their first consumer).

### 0.5 Spec deviations — settled decisions (approved 2026-07-02)

All four were flagged as open questions in v2 and are now approved decisions. The three that required a `DESIGN.md` change were applied to the spec on 2026-07-02 — spec and plan now agree.

1. **`read_file` joins the catalog (P2).** The spec's §5 catalog had no way to read a file's contents — the agent could list, search, and *overwrite*, but `update_file` is a full overwrite of content the model may never have seen, and nothing readable arrives until `run_command cat` in P3. Decided: `tools/read_file.py` lands in P2; the catalog is now **44 tools**. *Spec amended: `read_file` added to §5's explorer group.*
2. **`superseded_by` = reverse traversal.** §7.3's prose says `record_pivot` "auto-creates supersedes/superseded_by edges", but the spec's own `graph_edges` CHECK constraint excludes `superseded_by`. Decided: `supersedes` edge + `superseded_at` stamp; "superseded_by" is answered by traversing the edge in reverse (P14). *Spec amended: §7.3 prose now matches the schema.*
3. **chars/4 token fallback is new code.** §8's tiktoken fallback ("chars/4") is not a Chalie lift — Chalie's actual fallback is words×1.3. Decided: implement chars/4 exactly as specified, written fresh (P11). No spec change needed; the lift map (§0.2) records the provenance.
4. **`episodes.facts_extracted_at` mined-state column.** §7.4's gist-to-fact sweep needs a marker for which episodes have been mined; without one, P15 would force a second schema migration. Decided: the column ships in P12's single migration; P15's extractor keys on `IS NULL`. *Spec amended: column added to §7.1's episodes schema.*

### 0.6 Standard Definition of Done (applies to every phase)

1. Every step of the phase's **Live test** performed against the real running harness (scratch-copy convention, §0.1) and observed to behave as written.
2. All acceptance criteria checked off with observed evidence (transcript files, `--verbose` dumps, sqlite3 output, `git log`, process listings — as applicable).
3. No unit/feature/scripted tests added.
4. Code follows the project's coding standards; a critical self-review pass completed before commit.
5. Work committed with a descriptive message; the harness still passes the *previous* phases' live-test happy paths (spot-check: one representative command per prior phase).

Per-phase DoD sections list only phase-specific additions.
