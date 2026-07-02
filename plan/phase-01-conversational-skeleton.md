## Phase 1 — Conversational skeleton

**Goal:** A runnable REPL: user message → LLM → optional tool round-trip → final answer, with the full-fidelity transcript on disk and every permanent seam in place.

**Scope (atomic chunks):**
- `config.py` + `config.json`: Config dataclass + loader; `llm` block (`base_url`, `api_key`, `model`, `temperature`, `max_tokens`, `context_limit`). Other blocks parsed-if-present but unused.
- `llm.py`: OpenAI-compatible `/v1/chat/completions` client; tool-schema conversion; native `tool_calls` parsing; typed `OverCapError` raised on provider `context_length_exceeded`.
- `session.py`: full-fidelity transcript — YAML frontmatter (session_id, cwd, model, created_at) + flat native-OpenAI JSON message array — persisted to `<project>/.coding_agent/sessions/<ts>.json` every turn; fresh file per launch (no resume); `assemble_context()` pass-through consulting the (empty) **context-provider registry** (providers render blocks appended to the system message — seam #3).
- `agent.py`: the §9 loop with four named hooks (`flashback.maybe_seed` no-op; `except OverCapError` → surface a clear over-cap error to the user and end the turn, **no retry** — P11 swaps in compact-then-retry; `diagnostics.inject_summary` no-op; `episodic.maybe_extract` no-op).
- `main.py`: CLI entry — load config, capture launch CWD as project root, create session, REPL (read → agent loop → print → persist); named no-op `session_start_jobs` / `session_end_jobs` hooks (seam #2).
- `tools/result.py`: `ToolResult.ok/err` lifted from Chalie `abilities/_result.py` (strip `mp` coupling and framework params).
- `tools/base.py`: `Tool` ABC — `name` / `description` / `parameters` (JSON Schema) / `run(**kwargs) -> ToolResult`.
- `tools/registry.py`: auto-discovers `Tool` subclasses → OpenAI `tools` array + dispatch; unknown-tool and bad-args paths return `ToolResult.err`.
- `tools/list_files.py`: first tool — `.gitignore`-respecting tree listing, confined by a `resolve_in_root()` sandbox helper (hardened into the shared fs-write helper in P2).
- Observability backbone: `--verbose` flag dumping the exact assembled context per LLM call; every tool call + result echoed to stderr.
- Seed the **P1 slice** of `samples/playground` per the pinned manifest (§0.4): `app.py` (`create_app` + seeded unused import), `lib/fib.py`, `tests/test_fib.py` (one failing test), `bin/run_fib.py`, `index.html`, `styles.scss`, small Python class hierarchy.

**Out of scope:** pruning (P11), any protocol client, memory, all other tools, JS/PHP fixture slices (P5/P7).

**Dependencies:** none.

**Live test:**
1. `python main.py` inside a scratch copy of `samples/playground`. Type `hi, what can you do?` → plain conversational reply, no tool calls.
2. `what files are in this project?` → `list_files` tool-call echo on stderr, then an answer naming the real files.
3. Follow-up `which one is the entrypoint?` → coherent multi-turn answer (history works).
4. Error paths, explicitly instructed: `call list_files with a parameter named depth set to "banana" — I'm verifying error handling` → `ToolResult.err` echoed, loop survives; `now call a tool named make_coffee` → unknown-tool `ToolResult.err`, loop survives.
5. Quit. Open `.coding_agent/sessions/<ts>.json` → valid YAML frontmatter; flat JSON array; the assistant message carries native `tool_calls`; the result is a `role:"tool"` message keyed by `tool_call_id`.
6. Relaunch → a **new** session file is created (no resume).
7. Run with `--verbose` → the exact assembled context prints before each LLM call.

**Acceptance criteria:**
- [ ] REPL round-trips both a talk-only turn and a tool-call turn against a real OpenAI-compatible endpoint.
- [ ] Transcript file matches §4.3 exactly (frontmatter keys, native message shapes); nothing pruned from disk.
- [ ] `registry.py` discovers `list_files` with no manual registration step.
- [ ] Bad-args and unknown-tool paths produce `ToolResult.err` fed back as the tool message, observed via live-test step 4.
- [ ] The four agent-loop hooks + two session-boundary hooks + context-provider registry exist as named seams at the §9 positions.
- [ ] `--verbose` + tool-call echo work; fresh session file per launch.

**Definition of done:** standard DoD + the P1 playground slice committed; its pytest suite runs standalone with its seeded failure.
