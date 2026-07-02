## Phase 14 — Data-graph memory

**Goal:** The project's "why" with real typed edges: Rules that are always enforced in context, plus Decisions/Pivots/Specs that are similarity-recalled with 1-hop graph expansion and honest supersession history.

**Scope (atomic chunks):**
- `memory/graph.py` over the P12-created tables: `graph_nodes` (type CHECK rule/decision/pivot/spec, title, body, extra JSON, `superseded_at`, active) + `graph_edges` (edge_type CHECK supersedes/implements/constrains/refines/relates_to, UNIQUE(from,to,type), ON DELETE CASCADE); maintain `graph_nodes_fts` + `graph_nodes_vec` on write. **Spec-contradiction resolution (deliberate):** §7.3 prose mentions `superseded_by` edges, but the schema CHECK excludes them — implemented as `supersedes`-only + `superseded_at` stamp; "superseded_by" = reverse traversal of the edge.
- `tools/record_rule.py` (title, constraint), `tools/record_decision.py` (title, rationale, alternatives?, implements?), `tools/record_spec.py` (title, body, acceptance?, status?).
- `tools/record_pivot.py` (title, why, supersedes) — transactionally: insert node, create `supersedes` edges, stamp `superseded_at` on targets (all-or-nothing).
- `tools/link_nodes.py` (from_id, to_id, edge_type).
- Injection: **active Rules rendered into every assembled context by registering a provider into P1's context-provider registry** (seam #3 — pure registration; P11's pruning already preserves provider output; the rules render is exposed as a standalone function P15's flashback renderer will *call*, unchanged). Decisions/Specs/Pivots similarity-recalled + 1-hop edge expansion, registered into the recall registry; superseded nodes excluded from active recall but reachable as history.

**Out of scope:** the gated turn-0 seed bundle (P15 — which *adds* decisions/gists/atoms on top of the always-present rules; rules injection built here survives P15 unchanged).

**Dependencies:** P12 (P11's context log makes rule-render cost visible).

**Live test:**
1. `record a rule: all API responses use camelCase keys` (counter-conventional on purpose — snake_case is what models emit by default, so compliance is attributable to the injected rule). Relaunch → `--verbose` shows the rules block inside the first assembled context; `add a /users endpoint returning the current user` → emitted JSON keys are camelCase **without being asked**.
2. `record a decision: store sessions in SQLite; alternatives: redis, flat file` → node with extra JSON.
3. `record a pivot: session storage moves to Postgres — supersedes that decision` → sqlite3 shows pivot node, `supersedes` edge, `superseded_at` stamped.
4. `what's our session-storage approach and how did we get here?` → recall surfaces the pivot; 1-hop expansion drags in the superseded decision *as history*; it no longer appears as active guidance.
5. `link the camelCase rule as constraining the users spec` → `link_nodes` edge visible.
6. Transactionality via data-forced failure: `record a pivot superseding decision id 99999` (nonexistent) → constraint/FK error mid-transaction; sqlite3 confirms **no** pivot node and **no** supersedes edge were written (full rollback), then a valid re-run succeeds.

**Acceptance criteria:**
- [ ] Four node types + five tools work by conversation; extra JSON round-trips.
- [ ] Pivot supersession is transactional (proven by the forced-failure step) and auto-edges correctly.
- [ ] Active rules appear in **every** assembled context via provider registration (zero edits to session/compaction code); superseded nodes excluded from active recall, reachable via history.
- [ ] 1-hop expansion works on recall.

**Definition of done:** standard DoD.
