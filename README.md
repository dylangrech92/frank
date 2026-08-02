# Frank

An LLM-powered coding agent that acts as a headless IDE. It connects an LLM to a project through real tooling — LSP code intelligence, DAP debugging, sandboxed file operations, terminal access, test runners, profilers, a real browser, web search, and per-project long-term memory.

Frank is designed as a **companion to an orchestrator agent** — the orchestrator (Claude, opencode, GPT, or any MCP-compatible host) handles planning, sequencing, and user interaction. It delegates the deep code work to Frank via five mode-scoped tools: `research` (read-only), `code` (implement), `qa` (run tests and debug), `performance_debug` (measure), and `verify` (drive a real browser and return an evidence-backed verdict). Each tool invocation spins up a one-shot agent run against the target project.

Frank can also be used directly from the CLI as an interactive REPL or one-shot command.

---

## Quick start

### 1. Install dependencies

Python 3.10+ is required. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates a `.venv` and installs the pinned dependencies from the committed `uv.lock`: `requests`, `mcp`, `psutil` and `playwright` for the agent itself, plus the `dev` group (`pyyaml`) that the evaluation harness needs to parse tool output back. Run the agent with `uv run python main.py --mode <mode>`, or activate `.venv` and run `python main.py --mode <mode>` directly — see [Run directly from the CLI](#3b-run-directly-from-the-cli) for the available modes.

`verify` mode additionally needs a Chromium build. Install it once into the repo's own `.browsers/` directory (where the browser session looks for it, rather than the user-global Playwright cache):

```bash
uv run python scripts/install_browsers.py
```

Every other mode runs without it.

Optional packages (the agent degrades gracefully without each one):

| Package | What it enables |
|---|---|
| `tiktoken` | Accurate token counting (falls back to chars/4) |
| `sqlite-vec` | Vector search for memory recall (falls back to FTS-only) |
| `onnxruntime` + `numpy` | Local embeddings model for semantic memory (falls back to FTS-only) |
| `trafilatura` | Clean HTML extraction for `web_read` |
| `ddgs` | DuckDuckGo search for `web_search` |

### 2. Configure

```bash
cp config.example.json config.json
```

Edit `config.json` and set the `llm` block to point at any OpenAI-compatible endpoint (vLLM, Ollama, LMStudio, OpenAI itself, etc.):

```json
{
  "llm": {
    "base_url": "http://localhost:11434/v1",
    "api_key": "your-api-key",
    "model": "qwen2.5-coder:32b",
    "context_limit": 128000
  }
}
```

The rest of the config (linters, language servers, debug adapters) comes pre-filled with sensible defaults. See `config.example.json` for all options.

### 3a. Run as an MCP server (recommended for orchestrators)

```bash
uv run python mcp_server.py
```

This exposes five tools over stdio MCP transport:

- **`research(prompt, working_dir)`** — read-only investigation. Returns natural-language findings.
- **`code(prompt, working_dir)`** — implement changes. Returns JSON with files-changed and verification status.
- **`qa(prompt, working_dir)`** — run tests and debug. Returns JSON with test results.
- **`performance_debug(prompt, working_dir)`** — measure-first performance analysis of a target project: resource usage, wall time, hotspots, allocation sites, iteration counts, and nesting depth. Returns a measured report as a JSON envelope with status, answer, files_changed, and duration.
- **`verify(prompt, working_dir)`** — drive a real browser against a running system and return a verdict. Every `pass` assertion must cite evidence captured during the run (an accessibility snapshot, console output, a network response, an HTTP status, a URL, or a screenshot), or the report is refused and the agent has to try again. Returns a JSON envelope with `verdict` (`pass` / `fail` / `inconclusive`), the plan, the per-assertion evidence breakdown, observations, and the paths to the run's artifacts and Playwright trace.

**Browser prerequisite:** `verify` needs the Chromium build installed by `scripts/install_browsers.py` (see [Install dependencies](#1-install-dependencies)). Without it the first browser call fails loudly rather than silently reporting an unverifiable result.

**Profiling prerequisites:** Python profiling is fully bundled. Node profiling needs Node.js >= 12 on `PATH`. PHP profiling needs PHP with Xdebug >= 3.1. A missing runtime produces an explicit `interpreter-unavailable` or `xdebug-unavailable` error rather than silently degrading. The interpreter path can be overridden with `CODING_AGENT_PY_BIN`, `CODING_AGENT_NODE_BIN`, or `CODING_AGENT_PHP_BIN` (see `.env.example`).

Register it in your orchestrator's MCP config. For example, in opencode (`~/.config/opencode/opencode.jsonc`):

```json
{
  "mcp": {
    "frank": {
      "type": "local",
      "command": ["/path/to/frank/.venv/bin/python", "/path/to/frank/mcp_server.py"]
    }
  }
}
```

Or in Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "frank": {
      "command": "/path/to/frank/.venv/bin/python",
      "args": ["/path/to/frank/mcp_server.py"]
    }
  }
}
```

### 3b. Run directly from the CLI

`--mode` is required for every launch that talks to an LLM — it selects the tool set (`research`, `code`, `qa`, `performance_debug`, or `verify`) loaded into the request from turn 0.

```bash
# Interactive REPL
uv run python main.py --mode code

# One-shot task (final answer to stdout, telemetry to stderr)
uv run python main.py --mode research -p "Find all uses of the deprecated API and list the files"

# One-shot with JSON result envelope (for scripting / piping)
echo "Add a docstring to foo()" | uv run python main.py --mode code -p - --json

# Resume a previous session
uv run python main.py --list-sessions
uv run python main.py --mode qa --session 2026-07-11T14-30-00-12345
```

Run from inside the target project directory — the launch CWD becomes the sandboxed project root.

---

## How it works

Frank treats every IDE operation as an LLM-driveable tool. Every launch declares a mode — `research`, `code`, `qa`, `performance_debug`, or `verify` — and the LLM receives that mode's complete tool set with full schemas from turn 0. There is no discovery step and no way to load a tool outside the declared mode; each mode carries exactly the tools its task needs, which keeps context lean without making the model guess what's callable.

**Tool categories:**
- **Navigation**: `find_symbol` (name search, plus `action=` for definition / references / implementations / type_definition / hover), `call_hierarchy`, `document_symbols`
- **Editing**: `write_file` (create/new or full overwrite), `edit_file` (targeted search/replace), `rename_symbol`, `code_actions`, `format`, `move_file`
- **Terminal**: `run_command`, `read_output`, `stop_process`
- **Testing**: `run_tests`, `verify_scratch`
- **Debugging** (DAP): `set_breakpoint`, `debug_start`, `debug_control`, `debug_inspect`, `debug_stop`
- **Profiling**: `profile_command`, `profile_hotspots`, `profile_memory`, `trace_execution`
- **Browser** (Playwright, `verify` mode): `navigate`, `snapshot`, `click`, `fill`, `press`, `hover_element`, `select_option`, `scroll`, `wait_for`, `screenshot`, `console_logs`, `network_requests`, `http_request`, `handle_dialog`, `report`
- **Web**: `web_search`, `web_read`
- **Memory**: `remember`, `recall`, `forget`, `record`
- **Subagents**: `spawn_agents` (fan out independent tasks)
- **Search**: `find` (ripgrep), `find_files`, `list_files`
- **Harness feedback**: `report_issue` (log a tool or harness failure for the maintainer)

**Per-project memory** lives in `<project>/.coding_agent/memory.db` — a SQLite store with vector search (sqlite-vec) and full-text search (FTS5). It records knowledge atoms (facts anchored to code with content hashes for staleness detection), plus a typed graph of decisions, rules, specs, and pivots. A local `gte-modernbert-base` ONNX model provides embeddings offline; if it can't load, recall degrades to FTS-only.

**Session transcripts** are full-fidelity JSON arrays of native OpenAI messages, stored under `<project>/.coding_agent/sessions/`. An advisory lock prevents concurrent writes to the same transcript. Sessions can be resumed with `--session <id>`.

---

## Long-running tasks and MCP timeouts

MCP clients impose a request timeout on tool calls (the MCP SDK defaults to 60 seconds; some hosts like opencode default to 30). Since Frank spawns a subprocess that makes its own LLM calls, a complex task can easily take minutes.

Frank's MCP server sends **progress heartbeats** every 10 seconds while the subprocess runs. Clients that set `resetTimeoutOnProgress` (opencode does this by default) will reset their timeout clock on each heartbeat, giving the agent up to the full subprocess timeout (30 minutes by default) to complete.

**Host compatibility:**

| Host | Heartbeats work? | Why |
|---|---|---|
| **opencode** | Yes | Sets `resetTimeoutOnProgress: true` in its MCP client |
| **Claude Desktop / Claude Code** | Likely not | Uses the MCP SDK default (`resetTimeoutOnProgress: false`). Known to timeout on long-running tools. |

If your host doesn't reset on progress, the workaround is to use the CLI directly (`uv run python main.py --mode code -p - --json`) which has no timeout, or keep MCP tool calls small enough to finish within the host's window.

---

## Dashboard

Frank records per-session usage telemetry (runtime, token counts, tool calls) to `stats.json` (gitignored). Open `dashboard.html` in a browser to view an interactive chart of token consumption and KPIs. The data file is regenerated after each session — no server required.

---

## Environment variables

See `.env.example` for all supported variables. The key ones:

- `CODING_AGENT_PHP_ADAPTER` — path to the vscode-php-debug adapter (for PHP debugging)
- `CODING_AGENT_JS_ADAPTER` — path to the js-debug dapDebugServer.js (for JS debugging)
- `CODING_AGENT_PY_BIN` — override the Python interpreter used by the profiling tools
- `CODING_AGENT_NODE_BIN` — override the Node.js interpreter used by the profiling tools
- `CODING_AGENT_PHP_BIN` — override the PHP interpreter used by the profiling tools
- `CODING_AGENT_EMBED_MODEL` — override the default local embedding model path
- `CODING_AGENT_LSP_TRACE=1` — trace LSP protocol messages to stderr
- `CODING_AGENT_DAP_TRACE=1` — trace DAP protocol messages to stderr

---

## Project structure

```
main.py             CLI entry point (REPL + one-shot modes)
mcp_server.py       MCP server (research / code / qa / performance_debug / verify tools)
modes.py            The five modes; each launch declares exactly one
agent.py            Agent loop: turn orchestration and tool dispatch
turn/               Per-turn helpers (guards, steering, verification, rendering)
config.py           Config loading and validation
llm.py              OpenAI-compatible chat-completions client
session.py          Live conversation state (messages, summary, usage totals)
session_store.py    Transcript files on disk (read / write / list)
session_context.py  Assembling the message list sent to the LLM
session_lock.py     Advisory single-writer lock on a transcript
compaction.py       Reactive context compaction
jsonrpc.py          JSON-RPC framing for LSP/DAP
diagnostics.py      Diagnostic store (LSP publishDiagnostics)
ui.py               Terminal output formatting
stats.py            Per-session usage telemetry
tools/              One file per tool (auto-discovered via registry.py)
lsp/                LSP client + manager (language servers)
dap/                DAP client + manager (debug adapters)
runtime/            Process runner, test runner, web fetch, browser session
memory/             Per-project memory (atoms, graph, recall, consolidation)
samples/            Sample projects for testing (Python, JS, PHP)
evals/              Evaluation harness and scenarios
```

## License

MIT
