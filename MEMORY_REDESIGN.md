# Memory Redesign — codebase knowledge for a coding agent (M-series)

> **Status:** proposal / brainstorm exit artifact. Design-only; no code written yet.
> **Frame:** the current memory subsystem was ported from *Chalie* (a conversational
> agent) — see `DESIGN.md:201-271`. This redesign re-aims it at a *coding* agent:
> memory should model **the codebase**, not **the conversation**. Owner (Dylan)
> has granted license to drop / replace / augment existing layers.

---

## 1. Thesis

A coding agent's long-term memory should answer one question:

> *"What does a senior who already knows this codebase carry in their head that a
> narrow task can't tell you?"*

Not "what did we chat about," not "what happened 14 days ago." The value is a
persistent, queryable **model of the codebase** — vocabulary, concepts, how
subsystems fit together, invariants, conventions, gotchas, and the *why* behind
decisions — so that:

1. the agent is **smarter about the codebase** it works in, and
2. an **orchestrator can hand it a narrow slice** and the agent still sees the
   bigger picture, without hand-holding.

This directly serves the project thesis (coding_agent = an executor driven by an
orchestrator): a self-orienting executor lets the orchestrator delegate small,
underspecified slices safely.

Dylan's two hooks — a **gather** pass at task-start and a **persist** pass at
task-end — become the read/write loop around that model.

---

## 2. The single most important finding

From the landscape survey (Aider, Cody, Continue, Blar, mem0, Letta, A-MEM,
GraphRAG, DeepWiki — full citations in §11):

> **Structural code knowledge must be *derived*, not *remembered*.**

Every serious code tool rebuilds "where things live / what calls what" cheaply
from source (tree-sitter or LSP, mtime-diffed) instead of persisting it — because
code changes and stored structure goes stale, silently and constantly. Sourcegraph
Cody went further and **removed embeddings** (v5.3) in favour of keyword/BM25 +
structural search, for staleness, scale, and privacy reasons.

**Consequence — an important refinement of Dylan's idea #1 ("where things live"):**
"where things live" should be answered by a **fresh, derived project skeleton**,
*not* by persisted memory. What we *do* persist is the **semantic layer a skeleton
can't give you**: concepts, vocabulary, cross-cutting flows, conventions, gotchas,
and the *why*. Persisted facts are where all the poisoning/staleness risk
concentrates, so that store must be **small, code-anchored, reconciled, and
bounded** — never an append log.

---

## 3. Keep / Drop / Replace (grounded in current code)

| Current mechanism | Verdict | Rationale |
|---|---|---|
| SQLite + FTS5 + `sqlite-vec` substrate (`memory/store.py`) | **Keep** | Solid, local, per-project. Reuse. |
| Mem0-style `extract_facts` ADD/UPDATE/DELETE/NOOP (`atomic.py:543`) | **Keep + re-point** | This is the canonical reconciliation primitive — and coding_agent is *ahead of shipping mem0*, which quietly collapsed to ADD-only+MD5 dedup (V3, Apr 2026). Re-point its source from episode gists → the code diff. |
| Graph layer: rule/decision/pivot/spec + typed edges (`graph.py`) | **Keep + extend** | Genuinely coding-useful "why." Rules always-on is exactly right. |
| Bi-temporal supersession `valid_from/valid_to` (`atomic.py:135`) | **Keep** | "was true / is true" is the right audit model (mem0g's *mark-invalid-don't-delete*). |
| `CONTEXT_PROVIDERS` injection seam (`session.py:23,469`) | **Keep** | The only path memory reaches the model. Build on it. |
| **Episodic layer**: gists + `salience` 1-10 + exp **time-decay** (τ=14d) + eviction at 90d (`episodic.py`, `DESIGN.md:207-220`) | **Drop as a store** | "What happened this session," eroded by a *clock*, is conversational continuity. A coding agent cares what the code *is*, not what a chat did last fortnight. Repurpose only the write *mechanism* (window → LLM summarise) to feed diff→facts. |
| **Flashback continuation-gate** (message-embedding vs recent-turn centroid, `flashback.py:205`) | **Replace** | Topic-shift detection for a chat. Replace with *task-relative* retrieval keyed to the files/symbols the task names. |
| **Facts TTL** (`discovery`=14d, `misc`=3d, `atomic.py:20`) + power-law decay nudge (`atomic.py:313`) | **Replace** | Time is the wrong staleness axis. A fact doesn't expire on a clock — it goes false **when its code anchor changes**. Replace time-decay with **anchor-liveness** (§5). |

**Net effect:** this is largely *subtraction* — dropping the conversational
scaffolding is net-negative LOC (a success signal per the project Laws), while the
durable, coding-useful mechanisms (reconciliation, graph, substrate) stay and get
re-aimed.

---

## 4. Target architecture — three stores, two hooks

Split by **truth-lifetime** (the key design axis for code):

```
┌─ S-DERIVED ─ Project skeleton ────────────────────────────────┐
│  Where things live: file tree + ranked symbols per file.      │
│  EPHEMERAL — rebuilt from source each task, mtime-cached.      │
│  NEVER persisted as memory. LLM-free. (Aider repo-map model.)  │
└───────────────────────────────────────────────────────────────┘
┌─ S-CURATED ─ Rules & conventions ─────────────────────────────┐
│  Standing constraints (graph `rule` nodes, always injected)    │
│  + a small human/agent-owned conventions set.                  │
│  DURABLE, small, always/selectively injected.                  │
└───────────────────────────────────────────────────────────────┘
┌─ S-LEARNED ─ Anchored knowledge ──────────────────────────────┐
│  Concepts, vocabulary, cross-cutting flows, gotchas, WHY.      │
│  DURABLE, RECONCILED, BOUNDED, each atom CODE-ANCHORED.        │
│  (facts layer re-aimed + graph decisions/specs.)               │
└───────────────────────────────────────────────────────────────┘

        ORIENTATION (task-start)          CONSOLIDATION (task-end)
        agent.py:616                       agent.py:894
        ┌──────────────────┐               ┌──────────────────┐
task ──▶│ 1 refresh S-DERIVED│  answer ◀──│ 1 extract from diff│◀── turn_report
        │ 2 recall S-LEARNED │             │ 2 reconcile        │    .files_changed
        │   (staleness-check)│             │   ADD/UPD/DEL/NOOP │
        │ 3 gap-fill (lazy)  │             │ 3 anchor + record  │
        │ 4 inject brief     │             │   (background)     │
        └──────────────────┘               └──────────────────┘
```

---

## 5. Core innovation — code-anchoring & anchor-liveness

Every persisted knowledge atom carries a **code anchor**:

```
anchor_path    TEXT     -- file the insight is about (nullable for repo-wide)
anchor_symbol  TEXT     -- optional symbol (function/class) it concerns
anchor_hash    TEXT     -- content hash of the anchor AT LEARN TIME
learned_commit TEXT     -- git commit HEAD when learned
confidence     REAL     -- importance/confidence at write (Generative-Agents 1-10 → 0..1)
source         TEXT     -- 'consolidation' | 'explorer' | 'explicit-tool'
```

**Staleness = anchor drift, not elapsed time.** On recall, batch-check anchors
cheaply (Continue's hash-catalog pattern):

- anchor file missing → atom **flagged stale**, down-weighted, queued for re-verify.
- current content-hash ≠ `anchor_hash` → the code moved under the insight →
  **flag "may be stale (code changed since learned at <commit>)"** in the injected
  text so the model treats it as a hint, not ground truth.
- anchor unchanged → atom is **fresh by construction**, full weight.

This kills the two dominant failure modes at once: **staleness** (time-decay can't
know the code moved; hash-check does) and **poisoning** (a stale insight is
labelled, not silently trusted). It replaces the entire `salience/d_base/TTL/decay`
apparatus with one deterministic check.

> **What we deliberately do NOT anchor-persist:** pure structural facts ("`login`
> is in `auth/session.py`"). Those are S-DERIVED — regenerated fresh. Anchors exist
> so a *semantic* insight ("auth uses a rotating-token scheme, tokens minted in
> `auth/session.py`") knows *when to doubt itself*, not to restate structure.

---

## 6. Orientation pass — task-start (`agent.py:616`)

Replaces `flashback_maybe_seed`. Produces an **orientation brief**, stashed on the
session and injected via a `CONTEXT_PROVIDERS` entry (never persisted to transcript).

Steps, cheapest-first (only step 3 may call an LLM, and only rarely):

1. **Refresh S-DERIVED skeleton** (LLM-free, mtime-cached): gitignore-aware file
   tree (`list_files`) + ranked top symbols per relevant file (`document_symbols` /
   `find_symbol`), personalised to the files/identifiers the task names (Aider's
   personalized-PageRank idea; v1 can be a simpler relevance sort — see M3), fit to
   a token budget (~1k, Aider-style).
2. **Recall S-LEARNED** for the task area (§7 retrieval), **anchor-liveness-checked**
   (§5). Stale hits are kept but labelled.
3. **Lazy gap-fill** (rare, gated): if coverage for the task area is thin (few/no
   fresh anchored atoms *and* the skeleton is insufficient), spawn a **memory-less
   explorer subagent** (`spawn_agents` path) with a self-contained prompt to explore
   that area and return a structured brief (where things live, vocabulary, concepts,
   key flows). The **parent persists** the brief as anchored atoms — so the *next*
   task in this area is a cheap recall, not a re-exploration (DeepWiki "generate
   once, cache" + Letta amortisation).
4. **Inject** rules (S-CURATED, always-on) + skeleton + top-N anchored atoms as one
   compact brief.

**Gating (latency):** steps 1–2 are cheap and run every task. Step 3 blocks the
first token, so it is **gated to cold/thin areas only** and, where the mode allows,
run async (inject next turn). In one-shot `-p` mode a bounded sync explore is
acceptable *only* on a truly cold area, because it amortises across the whole task.

---

## 7. Retrieval & injection

- **Primary signal = keyword/FTS + structural**, vector as a *booster* (Cody's
  evidence: hybrid keyword/BM25 + structure beats pure vector on code for
  precision, cost, and staleness). Keep `sqlite-vec` — it's already built and
  working — but let FTS lead and let exact symbol/path matches win.
- **Task-relative personalisation** (Aider): the retrieval query is the task text
  *plus* the file paths / symbols it names; atoms anchored to those paths are
  boosted. This is what makes a *narrow* task pull the *right* bigger-picture context.
- **Two-stage budget contract** (Continue's `nRetrieve → rerank → nFinal`):
  pull ~25 candidates, re-rank, inject a small `nFinal` (e.g. ≤8) within a hard
  token budget.
- **Rank = relevance + confidence + anchor-freshness** (Generative-Agents
  recency+importance+relevance, with *anchor-freshness* substituted for the
  conversational *recency* term — code freshness, not clock freshness).

---

## 8. Consolidation pass — task-end (`agent.py:894`, background)

Re-points `extract_facts`. Fires on the **accepted** final answer only (mind the
three exit paths: normal `871-895`, over-cap give-ups `723/771`, and the S3
verify-gate *bounce* that can reach the final block twice). Runs **off the hot
path** (Letta sleep-time compute) — the user never waits on it; in one-shot it runs
at `session_end_jobs` (already the pattern, `main.py:199`).

1. **Extract** candidate atoms + decisions in one LLM call from `turn_report`'s
   `files_changed` diff + the transcript tail (mem0 extraction phase). Score
   **importance** so only durable, reusable insights persist (poisoning defence #1).
2. **Reconcile, don't append** (the crux for a codebase): for each candidate,
   retrieve top-k similar existing atoms and emit **ADD / UPDATE(same-key) /
   DELETE(same-key) / NOOP** — this is exactly today's `extract_facts` mechanism,
   which already does keyed bi-temporal supersession. Prefer UPDATE/DELETE on
   contradiction over blind ADD (bounded-growth defence).
3. **Anchor + record** (§5): attach path/symbol/hash/commit to each atom; route
   genuine "why"/reversals to graph `record_decision`/`record_pivot`.

Constraint resolved: a spawned subagent runs `--no-memory`, but `--no-memory`
disables only the *automatic* providers/sweeps — **the `recall`/`remember`/
`record_*` tools still write `memory.db`** (fresh store per call against cwd,
`tools/recall.py:65`, `tools/record_decision.py:58`). So consolidation may run
either **in-process** (like `extract_facts` today — preferred, simplest) *or* as a
memory-writing child. The explorer subagent (§6.3) likewise persists via these tools.

---

## 9. Sliced plan (M-series) — additive, ~1 day each, live-test-only

Ordered so each slice is independently demonstrable on the real hot path.

| Slice | Done-condition (one sentence) |
|---|---|
| **M1 — Anchored-knowledge schema** | `facts` (or a new `knowledge` table) gains anchor columns (path/symbol/hash/commit/confidence/source) via an additive v2 migration; `remember`/recall read+write them; a real atom round-trips with its anchor. |
| **M2 — Anchor-liveness staleness** | Recall batch-checks anchors (file-exists + content-hash) and labels/​down-weights drifted atoms; the decay/TTL nudge is removed from the scorer; demonstrated by moving a file and seeing its atom flagged stale on the next recall. |
| **M3 — Derived project skeleton** | A cheap, mtime-cached, token-budgeted skeleton (tree + ranked symbols, task-personalised) renders for this repo in well under the token budget with zero LLM calls. |
| **M4 — Orientation hook** | At `agent.py:616`, flashback is replaced by an orientation brief (skeleton + staleness-checked anchored recall + rules) injected via a provider; a narrow one-shot task visibly receives bigger-picture context it wasn't told. |
| **M6 — Consolidation hook** | At `agent.py:894`/`session_end_jobs`, extraction is re-pointed from gists → the diff; a real edit produces a reconciled, anchored atom (ADD/UPDATE/DELETE/NOOP) off the hot path. |
| **M5 — Lazy gap-fill explorer** | On a cold area, orientation spawns a memory-less explorer whose brief the parent persists as anchored atoms; a second task in that area recalls them instead of re-exploring (verified by log/DB). |
| **M7 — Retire conversational scaffolding** | Episodic decay/eviction/salience, the flashback continuation-gate, and facts TTL are removed; suite still green; net LOC negative. |
| **M8 — Reflection/compaction** (stretch) | At session boundaries, related atoms consolidate into higher-level notes and low-confidence/stale atoms evict, keeping the injected budget bounded (Generative-Agents reflection / Claude-Code capped-index pattern). |
| **M9 — Architecture overview doc** (stretch) | A cached, LLM-generated architecture brief regenerates on drift and feeds repo-level "how does this work" orientation (DeepWiki pattern; treated as cache, not truth). |

**Sequencing:** M1 → M2 (store + staleness) → M3 (skeleton) → M4 (read loop) →
M6 (write loop closes it) → M5 (the expensive, cached exploration) → M7 (subtract
the old) → M8/M9 (polish/stretch).

---

## 10. Pitfalls & mitigations (call these out; all evidenced in §11)

- **Memory poisoning** — a wrong atom, retrieved forever, self-reinforces.
  *Mitigate:* importance threshold to write at all; attribution + anchor + commit on
  every atom; UPDATE/DELETE reconciliation so contradictions overwrite; atoms are
  plain-text and human-inspectable; **never let the agent trust its own memory over
  live code** (anchor-liveness enforces this).
- **Staleness after code changes** — the dominant coding-agent failure. *Mitigate:*
  don't persist structural facts (derive them); anchor-liveness flags drift on recall.
- **Retrieval precision** — pure vector over-retrieves near-dupes, misses exact
  symbols. *Mitigate:* keyword/FTS-led hybrid + re-rank + task personalisation.
- **Unbounded growth** — append-only dilutes retrieval. *Mitigate:* reconcile on
  write, hard injection budget, periodic compaction/eviction (M8).
- **Cost per turn** — LLM-heavy write/read paths tax latency. *Mitigate:* orientation
  is LLM-free except the gated, cached explorer; consolidation is background /
  session-boundary; cache aggressively by mtime/hash.

---

## 11. Evidence appendix (primary sources)

- **mem0** — reconciliation primitive (ADD/UPDATE/DELETE/NOOP): paper
  arXiv:2504.19413. Note the *shipping* OSS default collapsed to ADD-only+MD5
  (docs.mem0.ai migration v2→v3) — coding_agent already implements the richer paper
  mechanism (`atomic.py:498-517`).
- **Aider repo map** — tree-sitter symbol graph + personalized PageRank + token
  budget + mtime cache: aider.chat/docs/repomap.html, repomap.py source.
- **Sourcegraph Cody** — removed embeddings for keyword/structural search:
  sourcegraph.com/blog/how-cody-understands-your-codebase.
- **Continue.dev** — hash+timestamp incremental index; `nRetrieve→rerank→nFinal`:
  docs.continue.dev.
- **Letta / MemGPT** — core-vs-archival split; sleep-time compute (background
  reconciliation off the latency path): arXiv:2310.08560, letta.com/blog/sleep-time-compute.
- **Generative Agents** — recency+importance+relevance ranking; write-time
  importance; reflection/compaction: arXiv:2304.03442.
- **A-MEM** — new memory revises linked neighbours (staleness reconciliation):
  arXiv:2502.12110.
- **DeepWiki (Cognition/Devin)** — LLM writes a durable architecture doc, RAG over
  repo+doc, treat as cache: cognition.com/blog/deepwiki.
- **Cline Memory Bank / Cursor rules / Claude Code memory** — file-based, injection-
  retrieved (not embedding-retrieved) markdown; Cline's six-slot schema as a
  *what-to-capture* checklist; Claude-Code capped-index pattern.

---

## 12. Decisions (locked by owner)

- **A — Structure → derived skeleton + cached explorer.** ✅ Deterministic
  LLM-free skeleton answers "where" (S-DERIVED / M3); a gated LLM explorer subagent
  answers "why/concepts" on cold areas only and **caches** its brief as anchored
  atoms (M5). Best freshness + amortised cost.
- **B — Episodic → drop the store, keep the mechanism.** ✅ Remove episodic-as-
  narrative + salience/decay/eviction; repurpose only the write mechanism
  (window → LLM-summarise) to feed diff → facts. Net-negative LOC (M7).
- **C — Retrieval → keep `sqlite-vec` as a booster.** ✅ FTS/keyword leads (best for
  exact symbols/paths); vector similarity boosts semantic matches. Reuse working
  infra (§7).
- **D — Skeleton ranking depth (still open, low stakes).** *Recommend:* ship the
  simple task-personalised relevance sort in M3; add Aider-style personalized
  PageRank (needs LSP `find_references` over the repo) later *only if* the simple
  sort under-selects. Not a blocker for M3.
