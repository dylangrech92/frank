# Design — `coding_agent`: an LLM-driven headless VS Code

- **Status:** Draft for review
- **Date:** 2026-07-01
- **Author:** Dylan (with Claude)
- **Scope:** A single, self-contained implementation plan's worth of work.

---

## 1. Thesis

Rebuild everything VS Code ships with — but **headless, CLI, and driven entirely by an LLM**. Every command-palette action, gutter affordance, and panel operation becomes an LLM-driveable **tool**. A human converses with the LLM; the LLM responds either by **talking back** (discussion) or by **operating the IDE** (tool calls that do real work). Nothing in the product is human-clickable — the LLM is the sole operator of the editor. The conversation transcript *is* the session.

## 2. Architectural spine

The harness is a **protocol multiplexer + tool registry** — precisely what VS Code's extension host is, minus the GUI. The harness itself contains almost no "intelligence"; it delegates to the **same external engines VS Code uses**. This is what keeps the codebase at bare-minimum LOC even at full feature parity: our code is three thin protocol clients plus a fleet of thin tools.

- **Protocol 1 — LSP** → language servers. Powers navigation, refactor, and diagnostics. The servers *are* the background linter: they stream `publishDiagnostics` continuously, so a single subsystem provides both code intelligence and the "red squiggles." No separate linter process exists.
- **Protocol 2 — DAP** → debug adapters. Powers breakpoints, stepping, and inspection. The **only stateful subsystem**: a debug session persists across multiple tool calls (launch → stopped-at-breakpoint → step → inspect → continue → terminate).
- **Protocol 3 — Process/PTY** → child processes. Powers the integrated terminal, tasks, and test runners.

Everything else in the harness — the agent loop, transcript, memory, compaction — is glue around these three clients and the LLM.

### 2.1 Per-language engine matrix

| Language | LSP (intelligence + diagnostics) | DAP (debug) | Test runner |
|---|---|---|---|
| Python | pyright | debugpy | pytest |
| PHP | intelephense | Xdebug (vscode-php-debug) | phpunit |
| JavaScript | typescript-language-server | js-debug (`node --inspect`) | jest / vitest |
| HTML | vscode-html-language-server | — | — |
| CSS / SCSS | vscode-css-language-server | — | — |

HTML/CSS/SCSS are not executable, so they receive **intelligence + formatting only** — no debug or test tools apply, mirroring VS Code's own behavior. Missing engines are reported once at startup and degrade gracefully (code-intel tools for that language return a clear "server not installed" error; text/file tools still work).

## 3. File layout

```
coding_agent/
  main.py            # CLI entry: load config → open new session → REPL
  config.py          # Config dataclass + config.json loader
  session.py         # Transcript (full-fidelity flat array) + context assembler (pruned view)
  agent.py           # Agent loop: LLM call → dispatch → compaction trigger → diagnostics inject
  llm.py             # OpenAI-compatible client (/v1/chat/completions) + tool-schema conversion
  embedding.py       # Local gte-modernbert-base (ONNX runtime); FTS-only fallback
  compaction.py      # Reactive over-cap compaction (in-flight-turn folding + usage-driven trigger)
  diagnostics.py     # Store fed by LSP publishDiagnostics; one-line summary + full dump
  lsp/
    client.py        # Generic LSP JSON-RPC-over-stdio client + server lifecycle
    manager.py       # language → server map, lazy start, didOpen/didChange sync, diagnostics fan-in
  dap/
    client.py        # Generic DAP JSON-RPC client + session lifecycle
    manager.py       # language → adapter map; holds LIVE debug-session state between tool calls
  runtime/
    process.py       # Terminal: one-shot run_command + background process handles
    tests.py         # Test-framework detection + structured pass/fail parsing
  memory/
    store.py         # SQLite open/migrate; sqlite-vec + FTS5 wiring
    anchor.py        # Code-anchoring: content-hash + git-commit stamp; anchor-liveness check
    atomic.py        # (kind,key,value) knowledge atoms; Mem0-style ADD/UPDATE/DELETE/NOOP; anchor-liveness scoring
    skeleton.py      # LLM-free derived project skeleton, task-personalised (mtime-cached, never persisted)
    orientation.py   # Task-start hook: zero-LLM code-anchored brief (skeleton + anchored recall + graph hits)
    consolidation.py # Task-end hook: mines the turn diff + transcript tail into durable anchored atoms, off-thread
    explorer.py      # Lazy gap-fill: bounded memory-less explorer for cold task areas; persists its brief as atoms
    graph.py         # Rule/Decision/Pivot/Spec nodes + typed edges
    recall.py        # Cross-layer hybrid (vector + FTS) recall + composite scoring
    episodic.py      # Episodic store retired (M7); now only `_safe_json_array`, a shared LLM-JSON-array parser
  tools/
    base.py          # Tool ABC: name/description/parameters(JSON-Schema)/run() -> ToolResult
    result.py        # ToolResult.ok/err contract (lifted from Chalie)
    registry.py      # Auto-discovers Tool subclasses → OpenAI tools array + dispatch
    <one thin class per tool — see §5>
  config.json        # user-provided connection + engine config
```

Per-project runtime artifacts live under the **project root** (the launch CWD), not the harness:

```
<project>/.coding_agent/
  sessions/2026-07-01T20-45-03-48213.json  # id is <timestamp>-<pid>; --session <id> resumes it
  memory.db                             # SQLite: facts, graph_nodes/edges (+ vec + fts)
```

## 4. Session, transcript & context assembly

### 4.1 Lifecycle
On launch the harness: (1) loads `config.json` from the agent's own install directory by default (`--config` overrides); (2) captures the launch **CWD as the project root** (all file operations are sandboxed to it); (3) creates a **new session transcript** with a `<timestamp>-<pid>` id (unique across concurrent launches), or resumes an existing one when `--session <id>` is passed (`--list-sessions` lists available ids); an advisory lock file guards against two processes appending to the same transcript at once; (4) pre-warms LSP servers for languages detected in the tree **on a background thread** (spawns overlap the first LLM round-trip; `get_client` is lock-guarded against the race); (5) enters a REPL: read user message → run the agent loop → print the assistant's text → persist the turn. **One-shot mode** (`-p/--prompt "task"`, or `-p -` to read the task from stdin) replaces the REPL for orchestrator/subagent use: it runs a single task, prints only the final answer to stdout (all telemetry is on stderr), and exits 0 on success / 1 on error. `--no-memory` skips every long-term-memory job (orientation, rules provider, end-of-session consolidation) for the fastest possible ephemeral run. Concurrent instances on the same project are safe: session ids are pid-unique, transcripts are advisory-locked, and memory.db runs WAL with a busy timeout.

### 4.2 Message format
**Native OpenAI tool-role messages.** An `assistant` message carries `tool_calls`; each result is a `role:"tool"` message keyed by `tool_call_id`. This is the standard OpenAI-compatible contract and maps 1:1 onto "tool results attached to the assistant response." (This deliberately diverges from Chalie, which flattens history into a single text user-message; the native format is leaner and standard here.)

### 4.3 Transcript on disk (full fidelity)
One file per session: YAML frontmatter (metadata) + a JSON body holding the **flat array of native-OpenAI messages**, plus an optional `summary`/`summary_covers` (persisted compaction state) key pair so `--session <id>` resumes with the compacted view. `session.py` also still loads/persists a legacy `episodic_watermark` key for transcript-schema back-compat, but no code outside `session.py` reads or advances it anymore (`grep episodic_watermark` across the tree hits only `session.py`): the row-count-gated episodic extractor it watermarked was retired with `memory/episodic.py` (M7), and its replacement, consolidation (§7), has no watermark — it enqueues unconditionally at the end of every turn that produced a final answer. Legacy transcripts predating these keys — a bare JSON array with no wrapping object — still load with no summary. Writes are atomic (temp file + `os.replace`), and an advisory `<transcript>.lock` file (containing the holding pid) prevents two processes from appending to the same transcript concurrently. Nothing is ever pruned from the record.

```
---
session_id: 2026-07-01T20-45-03-48213
cwd: /Users/dylangrech/dev/myproject
model: gpt-4o
created_at: 2026-07-01T20:45:03Z
---
{
  "messages": [
    {"role": "user", "content": "add a /health endpoint"},
    {"role": "assistant", "content": "Adding it.",
       "tool_calls": [{"id": "c1", "name": "create_file",
                       "arguments": {"path": "health.py", "content": "..."}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "create_file",
       "content": "[create_file(status=ok)]"},
    {"role": "assistant", "content": "Done — /health returns 200."}
  ]
}
```

### 4.4 Assembled context (the pruned view sent to the LLM)
The transcript on disk and the context sent to the model are **different**. `session.assemble_context()` builds the pruned view each step:

- **In-flight turn** (current user message + all its assistant/tool activity so far): **full** native scaffolding — the LLM must see what it just did.
- **Completed prior turns**: collapse to **`[user message, final assistant answer]` only**. All `tool_calls`, all `tool` result messages, and all mid-chain narration are dropped. (Dropping tool results *requires* stripping the `tool_calls` field too, since the API rejects dangling tool calls.) The record of *how* the work was done survives in the full-fidelity on-disk transcript (§4.3, never pruned), not the context window; durable, reusable insight distilled from it is separately captured as anchored knowledge atoms by consolidation (§7), not a narrative copy of the transcript.
- **Reactive compaction** sits on top: if even the pruned context would exceed `window − max(0.10·window, 8000)`, the oldest turns collapse into a single summary message.

**KV-cache-stable provider ordering:** `CONTEXT_PROVIDERS` blocks are joined into the system message in registration order — catalog (`main.register_catalog_provider`), then active-rules (`memory.graph.register_graph_provider`), then orientation (`memory.orientation.register_orientation_provider`) — all registered in that order by `main.session_start_jobs`, and that order is prefix-optimal for local-provider KV-cache reuse. `system_prompt` and `catalog` are byte-identical across turns by construction — `render_catalog_block()` lists every non-pinned tool in the registry regardless of `load_tool` activation state (the pinned first-touch set — see §5 — is named in its header and excluded from the listing; `PINNED` is a process-lifetime constant), so it never varies within a process lifetime. The `graph-rules` block only changes when a `record(kind='rule')` call fires mid-session (rare) and is registered *before* orientation. `orientation` is the block that reliably varies turn-to-turn — it runs on **every** turn now, unconditionally (`memory.orientation.orientation_maybe_seed` has no gate; retrieval is already task-relative), and its skeleton + recalled-atom content is a function of the current task text, so it differs whenever the task text does — and it is registered last, so it already sits at the tail of the system message. No `session.py` ordering change is required.

## 5. Tool catalog

~40 tools, every one a thin class delegating to a shared client. A small pinned core (`tools/registry.PINNED`: `load_tool`, `read_file`, `find`, `find_files`, `list_files`, `find_symbol`, `find_references`) ships in every request's tools array, so first-touch research costs no `load_tool` round-trip; every other tool is deferred behind the catalog and must be loaded with `load_tool(name)` first. Write and heavy tools stay gated on purpose — the load step is deliberate friction on destructive paths. Grouped by VS Code "area":

**Explorer (files)**
- `list_files(path?)` — tree listing, respects `.gitignore`
- `read_file(path, start_line?, end_line?)` — file contents, optional line range; output is cat -n style (each line prefixed with its true 1-based line number and a tab — display-only, never fed back into written content or `replace_*` search strings), and numbering reflects true file lines even when paging; errors with a use-the-range hint on oversized files
- `create_file(path, content)`
- `update_file(path, content)` — full overwrite of a small file; targeted changes should prefer `replace_one`/`edit_lines`
- `delete_file(path)`
- `move_file(src, dst)` — fires `workspace/willRenameFiles` so imports auto-update
- `create_folder(path)`

**Find / replace**
- `find(query, fuzzy?)` — ctrl+f; exact or fuzzy term search over file **contents** (ripgrep-backed)
- `find_files(pattern, path?)` — locate files by **name**/glob (`*scheduler*`, `*.sh`, `src/**/*.ts`); case-insensitive, gitignore-aware (ripgrep `--files`)
- `replace_one(file, old, new)` — replaces a **unique** string in one file; errors if the match is ambiguous (forces precise edits)
- `replace_many(old, new, glob?)` — **project-wide** text replace, optionally scoped by glob
- `edit_lines(path, start_line, end_line, new_text)` — replaces a **1-based inclusive line range** with `new_text` so large files change without a full rewrite; insertion is an empty range (`end_line = start_line - 1`; `start_line=1, end_line=0` inserts at the top; `start_line` one past the last line appends), deletion is an empty `new_text` (the result then says "deleted lines N-M" rather than claiming a replacement), and the result echoes a cat -n numbered preview of the changed region ± context; a length-changing edit skips the read-registry re-stamp so the file's next edit is refused until a fresh `read_file` (anchor-shift guard — stale line numbers can't land on the wrong lines)

**Code intelligence (LSP)**
- `go_to_definition(file, line, col)` · `go_to_implementation(...)` · `go_to_type_definition(...)`
- `find_references(file, line, col)` — all callers/usages
- `call_hierarchy(file, line, col, direction)` — incoming (callers) / outgoing (callees)
- `document_symbols(file)` — outline / breadcrumbs
- `find_symbol(query)` — fuzzy workspace symbol search (ctrl+T)
- `hover(file, line, col)` — type / signature / docs
- `signature_help(file, line, col)` — parameter hints

**Refactor / fix (LSP)**
- `rename_symbol(file, line, col, new_name)` — type-aware cross-file rename
- `code_actions(file, range)` — the lightbulb: quick-fixes, organize-imports, fix-all, extract (list + apply)
- `format(file)` — the built-in formatter (Black/Prettier/etc.)

**Problems**
- `get_diagnostics(file?)` — full diagnostics on demand (a one-line summary is auto-injected each turn; see §6)

**Source control**
- `git(subcommand, ...)` — safe subset: `status`, `diff`, `log`, `add`, `commit`, `branch`, `checkout -b`. Destructive operations (`reset --hard`, `clean`, `restore`, `checkout --`) are **excluded by default**, opt-in via `config.git.allow_destructive`.

**Terminal**
- `run_command(cmd, timeout?, background?)` — the integrated terminal, scoped to the project root; one-shot returns `{stdout, stderr, exit_code}`; `background:true` returns a process handle for long-lived servers (`artisan serve`, `npm run dev`), with companion `read_output(handle)` / `stop_process(handle)`. A deny-list blocks obviously destructive commands. When a foreground command exits nonzero, the result render appends a grounding line stating the exit code and the concrete absolute working directory the command ran in, so a failure caused by a hallucinated `cd` into the project self-corrects instead of retrying the wrong path; a zero-exit result carries no such line. A foreground run snapshots the project tree before and after and diffs it, so files a shell command creates/changes/deletes publish mutation events on the same bus the edit tools use (feeding files_changed accounting, the lint delta, and the verify gate); background commands do not publish mutation events.

**Tests (Test Explorer)**
- `run_tests(path?, pattern?)` — detects the framework, runs it, returns **structured** per-test pass/fail.
- `verify_scratch(snippet, interpreter?, timeout?)` — runs a throwaway verification snippet from a temp file **outside** the project tree, executed with the project root as cwd (so imports and relative paths resolve against the real code), then deletes it. `interpreter` is constrained to a fixed allow-list (`python`/`python3`/`node`/`php`/`bash`/`sh`) so nothing arbitrary reaches the shell. A non-zero exit returns an error (`verification-failed`) carrying stdout/stderr, so a *failed* verification never clears the verify gate. Exists so the agent verifies a fix without inlining a repro harness into a production file or repurposing its `if __name__ == "__main__"` block.

**Run & Debug (DAP)**
- `set_breakpoint(file, line, condition?)` · `clear_breakpoint(file, line)`
- `debug_start(target|config)` — launch a debug session
- `debug_control(action)` — `continue` | `step_over` | `step_into` | `step_out` | `pause`
- `debug_inspect(expression?|scope?)` — evaluate, list variables, or read the call stack at the current stop
- `debug_stop()`

**Web** (lifted from Chalie)
- `web_search(query)` — zero-config `ddgs` search
- `web_read(source, max_chars?)` — `trafilatura` extraction with the SSRF guard from Chalie's `web_fetch`

**Memory** (see §7)
- `remember(kind, key, value)` · `recall(query)` · `forget(...)`
- `record(kind, title, body, alternatives?, acceptance?, status?, supersedes?, implements?)` — one tool for every graph node. `kind` is `rule` (body = constraint), `decision` (body = rationale; `alternatives?`, `implements?`), `spec` (body = spec; `acceptance?`, `status?`), or `pivot` (body = why; `supersedes` required). `supersedes`/`implements` take node **titles or ids** — a title is resolved against the graph, and an ambiguous or unknown reference refuses the whole write and lists the candidates. The success result echoes the new node's id and every resolved edge target so ids become knowable for later calls.

**Subagents**
- `spawn_agents(specs)` — fans out `{prompt, cwd?}` entries to concurrent, memory-less one-shot children of this same harness (`main.py --no-memory -p -`, one process per spec), up to `subagents.max_concurrent` (config, default 4) at a time, each bounded by `subagents.timeout_s` (default 600). Each child sees only its own `prompt` — no shared context, transcript, or memory — so callers must write fully self-contained prompts. Returns per-child answer, exit code, and a stderr tail on failure/timeout. A depth guard (`CODING_AGENT_DEPTH` env var, incremented per generation) refuses to spawn once already two levels deep, so fan-out cannot recurse without bound. Not `parallel_safe` — children may mutate files.

### 5.1 Tool encapsulation convention
Every tool is one file/class:

```python
# tools/base.py
class Tool(ABC):
    name: str                       # unique tool name
    description: str                # LLM-facing description
    parameters: dict                # JSON Schema for arguments
    def run(self, **kwargs) -> ToolResult: ...   # the only entry point
```

`registry.py` auto-discovers every `Tool` subclass, renders the OpenAI `tools` array from each class's `name`/`description`/`parameters`, and dispatches calls to `run()`. **Altering a tool's schema is a single-file edit.** `ToolResult.ok(body, **meta)` / `ToolResult.err(message, code=...)` is the frozen result contract lifted from Chalie.

## 6. Diagnostics flow (hybrid)

`diagnostics.py` holds a `{uri: [diagnostics]}` store fed by LSP `publishDiagnostics`. After each agent step that mutated files, a compact one-line summary (e.g. `⚠ 3 errors, 2 warnings in 2 files`) is auto-appended to the next turn. Full detail is available on demand via `get_diagnostics(file?)`. This keeps context lean on clean runs while ensuring the LLM never drifts from real editor state.

## 7. Memory

Per-project SQLite store at `<project>/.coding_agent/memory.db` (sqlite-vec `vec0 float[768]` for KNN + FTS5 for lexical). Embeddings come from a **local `gte-modernbert-base`** model (768-d, offline, provider-agnostic) isolated behind `embedding.py`; if the model can't load, recall degrades to FTS-only.

Chalie merges "atomic recall" and "data-graph" into one table with mostly-aspirational edges. We **deliberately split them**, because the two have different write paths, lifecycles, and query patterns.

### 7.1 Layer 1 — Project skeleton (S-DERIVED — ephemeral, never persisted)
Answers "where things live": a gitignore-aware file tree (reusing `tools.list_files`' filter) plus the top-ranked symbols per most task-relevant file, personalised to the current task text by token-overlap scoring (filename tokens weigh 3x, directory tokens 1x; test-path matches de-prioritised, never excluded). Symbols come from an already-warm LSP server when one is running (queried with `spawn=False` — never blocks on a cold server start), falling back to a local `ast` (Python) or regex (other languages) extractor when no server is warm. Rebuilt fresh from source on every call, mtime-cached (`memory/skeleton.py`), and fit to a token budget (1000 tokens as called from orientation, `orientation.SKELETON_TOKEN_BUDGET`) that truncates the lowest-ranked symbols, then files, first. Zero LLM/embedder calls, and never written to `memory.db` — structural facts go stale silently, so they are *derived* every time, never remembered (MEMORY_REDESIGN.md §2).

### 7.2 Layer 2 — Atomic recall (`facts`)
Discrete knowledge atoms `(kind, key, value)`, value kept atomic; kinds are `project`, `convention`, `discovery`, `misc` (`memory.atomic.KINDS`). Written three ways, all through the same `remember()`/`forget()` primitive: the LLM's `remember` tool, `consolidation`'s task-end reconciliation (§7.4), and `explorer`'s persisted gap-fill bullets (§7.4) — each an ADD/UPDATE/DELETE/NOOP decision shown the top-N most-similar existing atoms before deciding. Recall is a hybrid vector + FTS composite score. Bi-temporal `valid_from/valid_to` handles contradiction (supersession sets `valid_to` on the old row, scoped by `(kind, key)`).

Every atom carries a **code anchor** (MEMORY_REDESIGN.md §5): `anchor_path`/`anchor_symbol` name what the insight is about (`NULL` = repo-wide), `anchor_hash` is the anchor file's content hash at learn time, `learned_commit` is the git HEAD when learned. **Staleness is anchor drift, not elapsed time** — there is no TTL or time-decay in this layer anymore. On recall (`recall_facts`), each candidate's anchor is re-hashed (`memory.anchor.is_stale`, cached per file for the call) and a missing/drifted anchor is down-weighted (×0.4) with its rendered text labelled `⚠ may be stale`, never silently trusted. Final score = FTS/vector relevance + `0.15·confidence` + `0.15·anchor-freshness`.

```
facts(id INTEGER PK, kind TEXT, key TEXT, value TEXT,
      salience_floor REAL, d_base REAL, retrieval_weight REAL,  -- legacy decay columns, unused since M7
      first_seen_at, last_confirmed_at, last_accessed_at,
      valid_from, valid_to, active INTEGER, deleted_at,
      anchor_path TEXT, anchor_symbol TEXT, anchor_hash TEXT, learned_commit TEXT,
      confidence REAL, source TEXT)  -- source: 'consolidation'|'explorer'|'explicit-tool'|'legacy'
facts_fts(key, value, kind)        -- fts5 porter-stemmed
facts_key_vec(vec0 float[768]); facts_value_vec(vec0 float[768])
```

### 7.3 Layer 3 — Data-graph (`graph_nodes` + `graph_edges`)
The project's "why" — and unlike Chalie, this graph has **real, working typed edges**. Four LLM-recorded node types, each timestamped:

- **Rule** — a standing constraint ("all API responses use snake_case").
- **Decision** — a chosen approach with rationale + alternatives.
- **Pivot** — a reversal/change of direction; inherently names the Decision(s) it overturns.
- **Spec** — a specification with body, acceptance criteria, and status.

```
graph_nodes(id INTEGER PK,
            type TEXT CHECK(type IN ('rule','decision','pivot','spec')),
            title TEXT, body TEXT,
            extra TEXT/*JSON: rationale|alternatives|acceptance|status|constraint*/,
            created_at, superseded_at, active INTEGER)
graph_edges(id INTEGER PK,
            from_id INTEGER REFERENCES graph_nodes(id) ON DELETE CASCADE,
            to_id   INTEGER REFERENCES graph_nodes(id) ON DELETE CASCADE,
            edge_type TEXT CHECK(edge_type IN
              ('supersedes','implements','constrains','refines','relates_to')),
            created_at, UNIQUE(from_id, to_id, edge_type))
graph_nodes_fts(title, body); graph_nodes_vec(vec0 float[768])
```

**Write path:** the single typed `record` tool (LLM-driven, mid-session — one call per `kind` of `rule`/`decision`/`pivot`/`spec`) and, since M-series, `consolidation`'s task-end pass (§7.4), which emits `DECISION`/`PIVOT` operations for a genuine design "why" or reversal it mines from the turn diff (`memory.consolidation.consolidate` → `graph.create_node`/`graph.record_pivot`). A `record(kind='pivot', supersedes=[title-or-id])` call (or the equivalent consolidation `PIVOT` op, whose `supersedes` entries are likewise resolved as node **titles or ids** via `graph.resolve_node_ref` — an unresolvable ref is dropped loudly and the op noops if none resolve) auto-creates `supersedes` edges and marks the target `superseded_at` — "superseded by" is answered by traversing the edge in reverse. A `decision`/`spec`/`rule` may also carry a `supersedes` (same edge + stamp) or, for a decision, an `implements` edge; `supersedes`/`implements` targets are given as node titles or ids and resolved before any write.

**Injection is special:** **active Rules are *always* injected** into context (they are constraints — a living coding-standard the LLM must always honor). Decisions / Specs / Pivots are similarity-recalled and 1-hop graph-expanded. Superseded nodes are excluded from active recall but remain reachable as history.

### 7.4 Integration (adapted for a CLI, not a daemon)
Chalie runs a 5-minute idle "subconscious" worker; a session-based CLI has no such loop, so:

- **Staleness is computed lazily at read-time** — anchor-liveness is a hash comparison, not a clock formula, so no cron is needed; `recall_facts` re-hashes each candidate's anchor file on every call (`memory.anchor.is_stale`).
- **No session-boundary maintenance sweep anymore.** M7 (MEMORY_REDESIGN.md §9) retired the old start-of-session eviction/TTL-purge/gist-mining background thread; `main.session_start_jobs` now only registers the always-on rules provider and the orientation provider (both synchronous, cheap — no LLM call, nothing to background). All durable-knowledge writes route through consolidation instead.
- **Consolidation** runs at **turn-end**, off the hot path: `agent.consolidation_maybe_extract` snapshots the turn's `files_changed` and a transcript tail and enqueues them on a background writer thread — unconditionally, with no row-count gate. `handle_user_message` fires it from a single choke point after the loop breaks, so it runs exactly once on **every** turn-exit path — a normal finalize, the blocked-loop escalation give-up (§9), and the over-cap give-up — not just the plain no-tool-call exit; a turn force-finalized after doing real file edits is mined into memory rather than silently dropped. `main.session_end_jobs` calls `memory.consolidation.drain_and_join()` at shutdown (REPL exit or one-shot teardown) so the final turn's pass is guaranteed to finish before the process exits, without ever firing twice for the same turn.
- **Injection** — the task-start seam descends from Chalie's turn-0 flashback pattern (stash a block on the session, render it via a `CONTEXT_PROVIDERS` entry), but the content and gating are unrelated: `memory.orientation.orientation_maybe_seed` runs on **every** turn (no terse/continuation gate — retrieval is already task-relative) and renders the derived skeleton (§7.1) + up to 8 anchor-liveness-checked recalled atoms (task text plus any file paths/symbols named in it, reranked to prefer atoms anchored to those paths) + up to 5 recalled Decisions/Specs. On a **cold** task area (fewer than 2 fresh anchored atoms), it additionally spawns a bounded, memory-less gap-fill explorer (`memory.explorer.explore`, ≤5 tool-calling rounds) whose brief the parent persists as anchored atoms (`explorer.persist_brief`) and injects into the current turn. Active **Rules** are injected separately and unconditionally by the always-on graph provider (every turn); the orientation brief reads them only to de-duplicate Decision/Spec hits, never to re-emit them. Both are stitched into the system message by the same `CONTEXT_PROVIDERS` seam, so the model sees rules + recall together.

**LLM-facing memory tools:** `remember` / `recall` / `forget` (knowledge atoms) and the single `record` graph tool (`kind` = rule/decision/pivot/spec, with title-or-id `supersedes`/`implements` edges).

## 8. Compaction

Reactive only: a pre-flight token estimate against `cap = window − max(0.10·window, 8000)`; if a send would exceed it (or the provider returns `context_length_exceeded`), compact then retry — no periodic polling. The summarization prompt is a coding-themed system prompt with fixed sections **Task / State / Files-touched / Open / Decisions / Last**. Persistence: replace everything above a watermark with a single summary message and keep the recent tail (no fork/watermark machinery). Token estimation uses `tiktoken` when available, else a `chars/4` heuristic.

**Folding the in-flight turn.** The fold boundary is not limited to *completed* prior turns — it may advance *into* the current in-flight turn, folding its older tool-call trail into the running summary. This is what lets a single long turn be compacted at all: a one-shot run (one user message + a long tool trail) has no completed turn before it, so a boundary pinned to the last user message could never make progress and the run would dead-end on the over-cap give-up (context over the model's token budget) `_fold_boundary` (compaction.py) keeps the last `keep_recent_messages` (default 8) verbatim, grows the folded region forward only while its rendered estimate stays within the summarizer's own budget (`cap` minus the summary prompt + any prior summary), and walks the boundary back off a leading `tool` row so a tool result is never orphaned from its originating assistant tool-call. Whenever the current user message is itself folded behind the watermark, `assemble_context` re-injects it verbatim as a task anchor so the model never loses the literal original request; the summary's paraphrase is a safety net, not a replacement.

**Usage-driven trigger.** Providers that report `usage.prompt_tokens` on a response give a real count — but it only measures the prompt of *that* request, and the context keeps growing after it (that response's own text, then new tool results). So the compaction gate compares against a composed signal, not the estimate alone: the session's most recent real `prompt_tokens` plus a calibrated estimate of only what was appended to the context since that request was built. When no real usage has been seen yet (or a compaction just reshaped the context, invalidating the baseline), the gate falls back to the plain estimate. Either way the estimate is scaled by an EMA (α = 0.3) of observed real-vs-estimated ratios carried on the session, so the `chars/4` fallback drifts toward the provider's real tokenizer over the life of a session instead of carrying a fixed, unverified undershoot.

## 9. Agent loop

```python
def handle_user_message(text):
    session.append_user(text)
    memory.orientation.orientation_maybe_seed(session)  # task-start brief: skeleton + anchored recall, zero-LLM, every turn
    while True:
        try:
            resp = llm.chat(session.assemble_context(), tools=registry.schemas())
        except OverCapError:
            compaction.run(session); continue         # reactive compaction, then retry
        session.append_assistant(resp.text, resp.tool_calls)
        if not resp.tool_calls:
            memory.consolidation.enqueue(session, client)  # task-end: off-thread, mines diff + transcript tail
            return print(resp.text)                    # turn ends
        for call in resp.tool_calls:
            session.append_tool_result(call.id, call.name, registry.dispatch(call))
        diagnostics.inject_summary(session)            # "⚠ 3 errors, 2 warnings"
```

**Streaming.** `llm.chat` accepts an optional `on_delta` callback; when set (and `llm.stream` is not disabled in config) the request is sent with `stream: true` and the SSE `delta.content` fragments are forwarded to the callback as they arrive, while tool-call fragments accumulate by index (ids synthesized as `call_<idx>` when a provider omits them). A provider that ignores `stream` and answers with plain JSON is handled transparently. The loop guarantees the final answer reaches `on_delta` exactly once — streamed live or delivered whole on fallback — followed by one `"\n"`, so callers that pass a sink never print the return value again. The REPL streams to stdout; one-shot mode streams to stderr, keeping stdout the pure final-answer channel.

**Bounded call latency.** Three independent timeouts bound one chat call, each guarding a distinct failure mode. A **connect** timeout caps establishing the socket, so a half-dead keep-alive connection cannot hang the agent. A **per-read-gap** timeout caps the silence *between* reads (not the whole response); on the non-streaming path nothing arrives until generation finishes, so that single gap spans the entire generation and already bounds the whole call — no extra ceiling is needed there. The streaming path is different: an SSE stream that keeps trickling chunks resets the per-read-gap timer on every line, so a pathologically slow-but-alive generation could otherwise run for the model's whole context window. A **whole-call wall-clock ceiling** closes that gap — a hard deadline on the total elapsed time of one streaming attempt, anchored when that attempt's request is issued (each retry attempt is anchored independently) and checked once per SSE line. On breach the attempt is abandoned with a distinct error whose message reports how much work is discarded (elapsed seconds, accumulated assistant-text characters, tool-call fragment count). This breach is **never retried**: it is raised only while draining a response whose 2xx status already returned and whose deltas were already forwarded, past the point the retry logic can re-issue the request. The ceiling is a generous backstop, not a latency target — a legitimately slow local model must never trip the production default.

**Parallel tool dispatch.** Tools carry a `parallel_safe` class attribute (default `False`), true only for tools that neither mutate state nor touch a main-thread-only resource (LSP document sync, the cached SQLite connection, process handles) — currently `read_file`, `list_files`, `find`, `get_diagnostics`, `web_search`, `web_read`. When an assistant batch has 2+ calls and every one is `parallel_safe`, the loop dispatches them on a `ThreadPoolExecutor` (≤8 workers) and appends results in original call order; any unsafe or unknown tool in the batch forces the fully sequential path, preserving effect ordering.

**Reactive steers and nudges.** A handful of per-turn watchdogs live inline in `handle_user_message`, each firing only on its trigger condition (zero cost on the happy path) and each capped so it can't loop forever. Two shapes: a nudge is a short suffix appended to the last tool result; a *steer* is a fresh turn-guidance message injected via `session.append_steer`. Steers ride the `user` role (the model weights user turns strongly) but are marked so nothing mistakes them for real human input — the content opens with an explicit marker spelling out that it is an automated harness message, not the user (`session.STEER_PREFIX`), and the message carries a `steer: True` flag. That flag is an internal key: `llm.to_wire_messages` is the single wire boundary that strips it (and any other non-wire key) from the provider payload, and the compaction summarizer renders a steered message under a `harness:` label rather than `user:`, so summaries never quote harness guidance as the user's own words.

When a verification steer fires (the reproduce-before-edit steer, the hard verify gate below), the harness also activates the verification tools that steer names — `run_command`, `run_tests`, `verify_scratch` — into the request's tool set at the moment the steer fires. Those tools are otherwise catalog-gated (callable only after an explicit `load_tool`), so a steer that told the model to run one without loading it would demand a tool it could not call; activating them keeps the guidance always actionable, and the tools appear in the very next round because the tool set is re-derived from the registry each round.

The `steer` flag is also read when the sent view is shaped for context management. That view keeps the in-flight turn verbatim and collapses every completed prior turn to just its user message and final answer; the boundary between the two is the last user-role message. Because a steer rides the `user` role, it must NOT count as a turn start: a steer is guidance injected *inside* a turn, not the beginning of a new one. So the boundary is anchored on the last *real* user message and skips steer rows. Treating a mid-turn steer as a turn start put the boundary after the turn's own work, folding that turn's assistant tool-call rows and their results into the completed-turn collapse — the model lost the record of what it had just done the moment a steer landed, re-derived it, tripped the repeat guard, and could spiral into a forced turn end. (Steer rows are still kept when a slice collapses, so harness guidance survives a compaction fold even though the scaffolding around it does not.)

- **Loop guard** — a repeated, byte-identical error envelope from the same tool gets a `[loop-guard]` suffix nudging the model to change approach instead of retrying the same call. A companion guard on repeated identical *successes* (the no-op loop where a model re-issues the exact same successful call) suffixes the second such result the same way, and a hard cap (`_REPEAT_CALL_CAP`) refuses an identical call outright once it has run that many times this turn (verification tools — `run_command`/`run_tests`/`verify_scratch` — are exempt, since a rebuild/retest cycle legitimately repeats). For a *read-only* repeated success (a `parallel_safe`, non-exempt tool) whose body has not changed, the re-render is **deduped**: the full body is replaced by a short stub pointing back at the earlier result, so an identical re-read costs a line instead of a whole duplicated body and stops rewarding the re-issue. Two safety conditions keep the stub honest, each defended by its own check: the render's content fingerprint must match the previous full render — a re-read of a file the model just edited legitimately differs and MUST get the fresh body — and the turn's compaction count must be unchanged since that render, because a fold could have dropped the earlier result out of context and a stub pointing at a vanished result would strand the model. If either fails, the full body is re-rendered (with the ordinary no-progress suffix) and re-stamped. Verification tools stay exempt from the suffix and the hard cap — a rebuild/retest cycle legitimately repeats — but they are **deduped on a stricter gate**: re-running a byte-identical command with *zero intervening file modifications* cannot produce a different result, so a repeated identical successful verification call is replaced by a distinct `[no-change]` stub (world fact plus one directive: change something before re-running verification) whenever the fingerprint, the compaction count, **and** a per-turn count of file-mutation events are all unchanged since the last full render. That mutation count spans both direct edit-tool writes and shell edits detected by the command runner's snapshot-diff, and it counts events rather than distinct paths so a second edit to an already-touched file still advances it; the moment it moves, the next identical run re-renders the full body plainly (no suffix — a genuine retest is progress) and re-stamps, so a later identical run with nothing further changed is deduped again. Read-only dedup carries this mutation stamp on its record but never gates on it: a re-read whose body is byte-identical is redundant regardless of an unrelated edit to some other file, and the fingerprint already speaks for content.
- **Blocked-loop escalation ladder** — the hard cap alone left a determined model able to re-issue a blocked call every round, each a full LLM round-trip that dispatches nothing; and once compaction folded the block error and the tool result out of context, the model lost even the feedback that it was stuck, so the loop could run unbounded until an external timeout. Two additions close it. First, a blocked round appends a plainly-worded *steer* (via `session.append_steer`) after all of the round's tool results — because the steer rides the `user` role, it lands past the compaction watermark and survives a fold even when the block error behind it does not, telling the model the earlier result was dropped to save space and it should not repeat the call. Second, a `blocked_streak` counter (reset to `0` the moment any call actually dispatches) force-finalizes the turn once it reaches `_BLOCKED_STREAK_CAP` consecutive blocked calls, so a stuck loop always terminates in a bounded number of rounds instead of spinning. The give-up answer is **synthesized truthfully from `turn_report`** (E14), not a static "a report of what was accomplished is unavailable" string that lies whenever the turn already applied and verified edits on disk: it leads with the harness's reason for ending the turn, then — only when the turn did work — lists the applied files and each verification run with its status (capped, with a `+N more` tail so a huge turn can't bloat the envelope) under a "Work already applied this turn." marker, and closes by warning the turn was cut short so parts may be incomplete and the applied changes should be reviewed before retrying; a turn that did no work keeps a plain give-up close to the original. No extra LLM call is made — the facts already live in `files_changed`/`verification_runs`. The over-cap give-up (§8) shares this same synthesis. Both give-up paths **stamp `turn_report["verified"]`** with the one shared formula (`_turn_verified` — mutated a file this turn and no unresolved verify gate), the same one `_finalize_answer` uses, so `verified` means the same thing on every turn-exit path and a give-up envelope never under-reports verified work. This give-up path breaks to the same single choke point as a normal finalize, so the turn-end consolidation hook (§7) still fires exactly once and the work done before the stall is mined into memory.
- **Web-search focus nudge** — 3+ consecutive successful `web_search` calls with no intervening `web_read` get a `[focus]` suffix telling the model to read a result instead of searching again.
- **Graph-memory usage nudge (H5)** — a turn whose mutations touch 3+ distinct files with no `record(kind='decision'/'spec')` call gets a one-line reminder appended to the last tool result, at most once per session.
- **Hard verify gate (H1 bounce, hardened by S3)** — when a final answer (no tool calls) ends a turn that mutated files with no successful `run_tests`/`run_command`/`verify_scratch` call since, the harness bounces once: it appends a harness steer (via `session.append_steer`, see above) telling the model to verify now or state explicitly that the change is unverified, then loops again instead of returning. If the *second* final answer still has neither a verification call nor an "unverified" declaration (case-insensitive match), the harness accepts it but prefixes the **returned** text (never the transcript) with `[UNVERIFIED CHANGES] `. At most one bounce per turn. The one `client.chat` call immediately after the bounce is delivered through `on_delta` whole rather than streamed fragment-by-fragment, since whether the marker applies can only be decided once the full response is in hand — this keeps the "final text exactly once, plus one trailing `\n`" streaming contract intact even when the gate rewrites the answer.
- **Reproduce-before-edit steer (E13)** — the verify gate above only fires when a turn tries to *end*, so a run that edits, re-edits, and never reaches the end-of-turn gate is never steered toward observed-output-first debugging. This steer closes that gap upstream: the first relevant file mutation of a turn whose `verification_runs` is still empty appends a plainly-worded steer (via `session.append_steer`) that first confirms the just-made edit was applied and forbids re-evaluating whether the task was needed, then gives one directive: before editing anything else, run the relevant command with `run_command` and read its actual output — reproducing the failure for a reported bug/crash/wrong-output, otherwise running the code to confirm the change — and base any further edits on that observed output rather than assumption. It rides the `user` role and is appended after the round's tool results (never between an assistant `tool_calls` row and its results), so it survives a compaction fold like the blocked-round steer. It fires at most once per turn, and because `verification_runs` records *every* `run_command`/`run_tests`/`verify_scratch` call regardless of exit status, even a crash-reproducing command that exits non-zero counts as "already observed something" and suppresses it.

**Structured result envelope (`--json`, S4).** One-shot mode accepts a `--json` flag (rejected by argparse unless `-p/--prompt` is also given) that swaps stdout's plain-text answer for exactly one JSON object and nothing else; stderr streaming/telemetry and exit codes are unchanged. Shape (version-pinned via the `envelope` field):

```json
{
  "envelope": 1,
  "status": "ok",
  "error": null,
  "answer": "<final answer text, WITHOUT the [UNVERIFIED CHANGES] prefix>",
  "verified": true,
  "declared_unverified": false,
  "files_changed": [{"path": "...", "tool": "replace_one"}],
  "verification_runs": [{"tool": "run_tests", "status": "success", "detail": "tests/"}],
  "usage": {"prompt_tokens": 11966, "completion_tokens": 409, "llm_calls": 6},
  "session_id": "...",
  "duration_s": 12.3
}
```

`handle_user_message` accumulates a per-turn report on `session.turn_report` alongside the existing trackers: `files_changed` is captured where the H1/S3 mutation tracking already correlates a dispatched call with its `_TURN_MUTATIONS` slice (deduped by path, first-tool-wins, in order of first mutation — not read back later, since `diagnostics_inject_summary` drains the module-level list at the end of each iteration); `verification_runs` records every `run_tests`/`run_command`/`verify_scratch` call with its result status and a short detail (the command, or the target path); `usage` sums `prompt_tokens`/`completion_tokens` across every `client.chat` call this turn and counts the calls. `verified`/`declared_unverified` carry the S3 gate's outcome, and in `--json` mode the `[UNVERIFIED CHANGES] ` marker is *not* prefixed onto `answer` — the two flags carry that state instead, so an orchestrator branches on structured fields rather than string-matching a prefix. Prose mode (no `--json`) is untouched: the returned string still gets the marker exactly as before. On a turn exception the envelope is still emitted, with `status: "error"`, `answer: null`, and whatever the report had accumulated up to that point — the caller exits 1 either way.

## 10. config.json

Config is per-agent-install, not per-project: it defaults to `config.json` next to `main.py` in the agent's own directory, and `--config <path>` overrides that for testing or alternate setups.

```json
{
   "llm": { "base_url": "https://api.provider.com/v1", "api_key": "…", "model": "…",
            "temperature": 0.2, "context_limit": 128000, "stream": true },
  "language_servers": {
    "python": "pyright-langserver --stdio",
    "php": "intelephense --stdio",
    "javascript": "typescript-language-server --stdio",
    "html": "vscode-html-language-server --stdio",
    "css": "vscode-css-language-server --stdio",
    "scss": "vscode-css-language-server --stdio"
  },
  "debug_adapters": { "python": "debugpy", "php": "php-debug", "javascript": "js-debug" },
  "test_runners": { "python": "pytest", "php": "phpunit", "javascript": "jest" },
  "compaction": { "reserve_ratio": 0.10, "reserve_min_tokens": 8000, "keep_recent_messages": 8 },
  "git": { "allow_destructive": false },
  "subagents": { "max_concurrent": 4, "timeout_s": 600 }
}
```

## 11. Subsystem notes

- **LSP client** (`lsp/client.py`): minimal JSON-RPC over stdio with `Content-Length` framing and a threaded reader; `manager.py` maps language → server, lazily starts servers, keeps them in sync with `didOpen`/`didChange` after every mutating tool call, and fans `publishDiagnostics` into `diagnostics.py`. Position-based tools translate `(file, line, col)` into LSP `textDocument/*` requests.
- **DAP client** (`dap/client.py`): same framing style as LSP; `manager.py` owns the **live debug session** — breakpoints, the launched adapter, and the current stop location persist between tool calls so the LLM can drive a session step by step. `debug_start` launches the adapter for the target language; `stopped` events return the location + stack for `debug_inspect`.
- **PHP debugging setup**: the Python adapter (`debugpy`) is resolved on `PATH` and needs no extra setup. PHP debugging uses [Xdebug](https://xdebug.org) plus the [vscode-php-debug](https://github.com/xdebug/vscode-php-debug) DAP adapter. Requirements: (1) install Xdebug for your PHP build (`pecl install xdebug`); (2) clone and build the adapter (`npm install && npm run build`); (3) point the `CODING_AGENT_PHP_ADAPTER` environment variable at the built `out/phpDebug.js`. Adapter command tokens in `debug_adapters` are expanded for `~` and `${ENV}` at load, so machine-specific paths stay out of the committed config. The launch synthesizer passes `-dxdebug.mode=debug -dxdebug.start_with_request=yes` on the PHP command line, so no global `php.ini` changes are required. If the variable is unset or the script is missing, PHP debugging fails with a single clear "adapter not fully configured / script not found" error and other languages are unaffected.
- **JavaScript debugging setup**: Node debugging uses Microsoft's [js-debug](https://github.com/microsoft/vscode-js-debug) adapter in **DAP-server mode**. Requirements: (1) download the `js-debug-dap` release tarball from the js-debug releases page (a prebuilt bundle — no `npm install` needed) and extract it; (2) point the `CODING_AGENT_JS_ADAPTER` environment variable at the extracted `src/dapDebugServer.js`. The adapter entry sets `"transport": "tcp"`: the manager spawns `node dapDebugServer.js <port> 127.0.0.1` (an ephemeral port), waits for it to listen, and connects a **parent** DAP session. js-debug then issues a `startDebugging` reverse request; the manager answers it by opening a **child** session on the same port, where breakpoints bind and the Node process actually stops. Child sessions report `threadId: 0`, so the manager treats thread id `0` as valid rather than missing. If the variable is unset or the script is missing, JS debugging fails with a single clear "adapter not fully configured / script not found" error and other languages are unaffected.
- **Runtime** (`runtime/process.py`, `tests.py`): `process.py` runs commands scoped to the project root with a timeout and a destructive-command deny-list, plus a background-handle registry for long-lived servers; `tests.py` detects the framework and parses its output (e.g. pytest `--json-report`, PHPUnit/Jest JUnit XML) into structured results.
- **Web** (`tools/web_search.py`, `tools/web_read.py`): `ddgs` for search; `trafilatura` extraction behind the SSRF guard, both adapted from Chalie's `read.py` / `web_fetch.py`.

## 12. Safety

- All file paths are confined to the project root; traversal outside it is rejected.
- Destructive git is opt-in; the terminal has a destructive-command deny-list; long-running processes are tracked handles, not fire-and-forget.
- `web_read` retains Chalie's SSRF guard (private-URL blocking before any socket).

## 13. Coding standards & testing

- **Bare-minimum LOC**; one encapsulated class per tool with an easily-altered schema; all heavy lifting delegated to the three shared clients. Governed by the project's `/code-standards`.
- **Feature tests, zero mocks**: end-to-end runs drive a throwaway sample project against **real** language servers / debug adapters / test runners, with a scripted OpenAI-compatible endpoint making the LLM side deterministic. This keeps the tool/protocol hot path un-mocked while removing model nondeterminism.

## 14. Decision ledger

| # | Decision | Choice |
|---|---|---|
| 1 | Code-intelligence engine | Thin **LSP client** → real language servers |
| 2 | Harness language | **Python** (reuse Chalie) |
| 3 | Diagnostics flow | **Hybrid** — one-line summary auto-injected + `get_diagnostics` |
| 4 | `replace_many` scope | **Project-wide** |
| 5 | Semantic rename | **Yes** — `rename_symbol` via LSP |
| 6 | Runtime surface | **Full parity** — terminal + `run_tests` + **DAP** debugger |
| 7 | Message format | **Native OpenAI** tool-role messages |
| 8 | Transcript | **Flat message array** under YAML frontmatter |
| 9 | Context assembly | Full in-flight turn; past turns → **final answer only**; reactive compaction on top |
| 10 | Memory scope | **Per-project** (`<project>/.coding_agent/memory.db`) |
| 11 | Memory layers | **Project skeleton** (S-DERIVED, ephemeral, zero-LLM) + **Atomic recall** (Mem0-style ADD/UPDATE/DELETE/NOOP, anchor-liveness) + **Data-graph** (Rule/Decision/Pivot/Spec, real edges) |
| 12 | Embeddings | **Local `gte-modernbert-base`** (768-d, ONNX), FTS-only fallback |
| 13 | Data-graph write path | **LLM-driven typed tool** (`record`, `kind`=rule/decision/pivot/spec) |

## 15. Deliberately out of scope (candidate phase 2)

- Atom reflection/compaction into higher-level notes at session boundaries, with low-confidence/stale-atom eviction (MEMORY_REDESIGN.md M8, stretch — not built).
- Cached, LLM-generated architecture-overview doc feeding repo-level orientation (MEMORY_REDESIGN.md M9, stretch — not built).
- Cross-project (global) memory layer.
- Multi-provider abstraction beyond OpenAI-compatible (Anthropic/Gemini native).
