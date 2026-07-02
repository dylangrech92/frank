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
  compaction.py      # Reactive over-cap compaction (Chalie-derived prompt + trigger)
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
    episodic.py      # Gist extraction, reconsolidation, decay, eviction
    atomic.py        # (kind,key,value) atoms; Mem0-style extraction; per-kind decay
    graph.py         # Rule/Decision/Pivot/Spec nodes + typed edges
    recall.py        # Cross-layer hybrid (vector + FTS) recall + composite scoring
    flashback.py     # Turn-0 seed injection (terse/continuation gates)
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
  sessions/2026-07-01T20-45-03.json     # a NEW transcript is created every launch
  memory.db                             # SQLite: episodes, facts, graph_nodes/edges (+ vec + fts)
```

## 4. Session, transcript & context assembly

### 4.1 Lifecycle
On launch the harness: (1) loads `config.json`; (2) captures the launch **CWD as the project root** (all file operations are sandboxed to it); (3) creates a **new session transcript** (fresh file every launch — no resume); (4) pre-warms LSP servers for languages detected in the tree; (5) enters a REPL: read user message → run the agent loop → print the assistant's text → persist the turn.

### 4.2 Message format
**Native OpenAI tool-role messages.** An `assistant` message carries `tool_calls`; each result is a `role:"tool"` message keyed by `tool_call_id`. This is the standard OpenAI-compatible contract and maps 1:1 onto "tool results attached to the assistant response." (This deliberately diverges from Chalie, which flattens history into a single text user-message; the native format is leaner and standard here.)

### 4.3 Transcript on disk (full fidelity)
One file per session: YAML frontmatter (metadata) + a JSON body that is a **flat array of native-OpenAI messages**. Nothing is ever pruned from the record.

```
---
session_id: 2026-07-01T20-45-03
cwd: /Users/dylangrech/dev/myproject
model: gpt-4o
created_at: 2026-07-01T20:45:03Z
---
[
  {"role": "user", "content": "add a /health endpoint"},
  {"role": "assistant", "content": "Adding it.",
     "tool_calls": [{"id": "c1", "name": "create_file",
                     "arguments": {"path": "health.py", "content": "..."}}]},
  {"role": "tool", "tool_call_id": "c1", "name": "create_file",
     "content": "[create_file(status=ok)]"},
  {"role": "assistant", "content": "Done — /health returns 200."}
]
```

### 4.4 Assembled context (the pruned view sent to the LLM)
The transcript on disk and the context sent to the model are **different**. `session.assemble_context()` builds the pruned view each step:

- **In-flight turn** (current user message + all its assistant/tool activity so far): **full** native scaffolding — the LLM must see what it just did.
- **Completed prior turns**: collapse to **`[user message, final assistant answer]` only**. All `tool_calls`, all `tool` result messages, and all mid-chain narration are dropped. (Dropping tool results *requires* stripping the `tool_calls` field too, since the API rejects dangling tool calls.) The record of *how* the work was done survives in episodic memory, not the context window.
- **Reactive compaction** sits on top: if even the pruned context would exceed `window − max(0.10·window, 8000)`, the oldest turns collapse into a single summary message.

## 5. Tool catalog

~40 tools, every one a thin class delegating to a shared client. Grouped by VS Code "area":

**Explorer (files)**
- `list_files(path?)` — tree listing, respects `.gitignore`
- `read_file(path, start_line?, end_line?)` — file contents, optional line range; errors with a use-the-range hint on oversized files
- `create_file(path, content)`
- `update_file(path, content)` — full overwrite
- `delete_file(path)`
- `move_file(src, dst)` — fires `workspace/willRenameFiles` so imports auto-update
- `create_folder(path)`

**Find / replace**
- `find(query, fuzzy?)` — ctrl+f; exact or fuzzy term search (ripgrep-backed)
- `replace_one(file, old, new)` — replaces a **unique** string in one file; errors if the match is ambiguous (forces precise edits)
- `replace_many(old, new, glob?)` — **project-wide** text replace, optionally scoped by glob

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
- `run_command(cmd, timeout?, background?)` — the integrated terminal, scoped to the project root; one-shot returns `{stdout, stderr, exit_code}`; `background:true` returns a process handle for long-lived servers (`artisan serve`, `npm run dev`), with companion `read_output(handle)` / `stop_process(handle)`. A deny-list blocks obviously destructive commands.

**Tests (Test Explorer)**
- `run_tests(path?, pattern?)` — detects the framework, runs it, returns **structured** per-test pass/fail.

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
- `record_rule(title, constraint)` · `record_decision(title, rationale, alternatives?, implements?)` · `record_pivot(title, why, supersedes)` · `record_spec(title, body, acceptance?, status?)`
- `link_nodes(from_id, to_id, edge_type)` — optional post-hoc relation

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

### 7.1 Layer 1 — Episodic (`episodes`)
The "what happened" narrative. Gists are auto-extracted at **turn-end**, count-gated (≥ N new transcript rows past a watermark) and run **off-thread** so the turn never blocks. The encoder returns `{gist, transcript_ids, has_open_loop, update_id, delete_id}`, supporting **reconsolidation** (update-in-place via `update_id`) and **obsolescence** (`delete_id`).

```
episodes(id TEXT PK, gist TEXT, salience INTEGER CHECK(1..10),
         created_at, last_relevant_at, last_accessed_at,
         transcript_ids TEXT/*JSON*/, has_open_loop INTEGER,
         facts_extracted_at,  -- set once the fact-extractor has mined this gist (§7.2)
         deleted_at)
episodes_fts(gist)                 -- fts5
episodes_vec(vec0 float[768])
```

**Erosion:** exponential decay `weight = (salience/10) · exp(−Δt_hours / τ)`, with `τ_leaf = 14d`, anchored on `last_relevant_at`. **Eviction:** hard-delete when `weight < 0.05 AND salience ≤ 3 AND age > 90d`. Chalie's UMAP+HDBSCAN super-episode clustering is **dropped** (heaviest dependency, marginal value here); leaf gists + decay + reconsolidation deliver "erode over time."

### 7.2 Layer 2 — Atomic recall (`facts`)
Discrete atoms `(kind, key, value)`, value kept atomic. Written **both** ways: the LLM's `remember` tool **and** a Mem0-style auto-extractor that mines new episode gists into facts (ADD / UPDATE / DELETE / NOOP, shown the top-N most-similar existing facts before deciding). Recall is a hybrid vector + FTS composite score. Bi-temporal `valid_from/valid_to` handles contradiction (supersession sets `valid_to` on the old row).

```
facts(id INTEGER PK, kind TEXT, key TEXT, value TEXT,
      salience_floor REAL, d_base REAL, retrieval_weight REAL,
      first_seen_at, last_confirmed_at, last_accessed_at,
      valid_from, valid_to, active INTEGER, deleted_at)
facts_fts(key, value, kind)        -- fts5 porter-stemmed
facts_key_vec(vec0 float[768]); facts_value_vec(vec0 float[768])
```

Kinds (coding-themed): `project` (facts about this codebase), `convention` (observed coding conventions), `discovery` (things learned, 14d TTL), `misc` (short TTL). Per-kind decay is power-law `rw = max(salience_floor, max(1, age_days)^(−d_base))` with per-kind TTL hard-purge.

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

**Write path:** LLM-driven only, via the typed `record_*` tools. A `record_pivot(..., supersedes=[id])` call auto-creates `supersedes` edges and marks the target `superseded_at` — "superseded by" is answered by traversing the edge in reverse; other relations are declared on the recording tool or via `link_nodes`.

**Injection is special:** **active Rules are *always* injected** into context (they are constraints — a living coding-standard the LLM must always honor). Decisions / Specs / Pivots are similarity-recalled and 1-hop graph-expanded. Superseded nodes are excluded from active recall but remain reachable as history.

### 7.4 Integration (adapted for a CLI, not a daemon)
Chalie runs a 5-minute idle "subconscious" worker; a session-based CLI has no such loop, so:

- **Decay is computed lazily at read-time** — it is a pure time formula, so no cron is needed; `retrieval_weight` is derived on recall from `last_relevant_at`.
- **Eviction + fact-extraction** run at **session boundaries** (start and/or end).
- **Episodic extraction** runs at **turn-end**, off-thread, count-gated.
- **Injection** uses Chalie's **turn-0 flashback** pattern: two zero-LLM gates first (skip on terse messages < 8 tokens; skip on continuations where the message embedding is close to the recent-conversation centroid), then render a compact block — active Rules + top Decisions/Specs + ≤ 3 dated episode gists + ≤ 5 atoms — recorded as context the model reads before iteration 0.

**LLM-facing memory tools:** `remember` / `recall` / `forget` (atoms + episodes) and the four `record_*` graph tools (+ optional `link_nodes`).

## 8. Compaction

Reactive only, mirroring Chalie: a pre-flight token estimate against `cap = window − max(0.10·window, 8000)`; if a send would exceed it (or the provider returns `context_length_exceeded`), compact then retry — no periodic polling. The summarization prompt is Chalie's `ChatHistoryCompactionSystemPrompt`, re-themed for coding with fixed sections **Task / State / Files-touched / Open / Decisions / Last**. Persistence is simplified: replace everything above a watermark with a single summary message and keep the recent tail (no fork/watermark machinery). Token estimation uses `tiktoken` when available, else a `chars/4` heuristic.

## 9. Agent loop

```python
def handle_user_message(text):
    session.append_user(text)
    memory.flashback.maybe_seed(session)              # turn-0 recall injection (gated)
    while True:
        try:
            resp = llm.chat(session.assemble_context(), tools=registry.schemas())
        except OverCapError:
            compaction.run(session); continue         # reactive compaction, then retry
        session.append_assistant(resp.text, resp.tool_calls)
        if not resp.tool_calls:
            memory.episodic.maybe_extract(session)     # count-gated, off-thread
            return print(resp.text)                    # turn ends
        for call in resp.tool_calls:
            session.append_tool_result(call.id, call.name, registry.dispatch(call))
        diagnostics.inject_summary(session)            # "⚠ 3 errors, 2 warnings"
```

## 10. config.json

```json
{
  "llm": { "base_url": "https://api.provider.com/v1", "api_key": "…", "model": "…",
           "temperature": 0.2, "max_tokens": 4096, "context_limit": 128000 },
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
  "compaction": { "reserve_ratio": 0.10, "reserve_min_tokens": 8000 },
  "git": { "allow_destructive": false }
}
```

## 11. Subsystem notes

- **LSP client** (`lsp/client.py`): minimal JSON-RPC over stdio with `Content-Length` framing and a threaded reader; `manager.py` maps language → server, lazily starts servers, keeps them in sync with `didOpen`/`didChange` after every mutating tool call, and fans `publishDiagnostics` into `diagnostics.py`. Position-based tools translate `(file, line, col)` into LSP `textDocument/*` requests.
- **DAP client** (`dap/client.py`): same framing style as LSP; `manager.py` owns the **live debug session** — breakpoints, the launched adapter, and the current stop location persist between tool calls so the LLM can drive a session step by step. `debug_start` launches the adapter for the target language; `stopped` events return the location + stack for `debug_inspect`.
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
| 11 | Memory layers | **Episodic** (decay) + **Atomic recall** (Mem0-style) + **Data-graph** (Rule/Decision/Pivot/Spec, real edges) |
| 12 | Embeddings | **Local `gte-modernbert-base`** (768-d, ONNX), FTS-only fallback |
| 13 | Data-graph write path | **LLM-driven typed tools** (`record_rule/decision/pivot/spec`) |

## 15. Deliberately out of scope (candidate phase 2)

- Episodic super-episode roll-up (UMAP + HDBSCAN clustering).
- Auto-mined / hybrid data-graph suggestion.
- Cross-project (global) memory layer.
- Session resume (each launch is intentionally a fresh transcript).
- Multi-provider abstraction beyond OpenAI-compatible (Anthropic/Gemini native).
