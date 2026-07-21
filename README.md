# Frank

An LLM-powered coding agent that acts as a headless IDE. It connects an LLM to a project through real tooling — LSP code intelligence, DAP debugging, sandboxed file operations, terminal access, test runners, web search, and per-project long-term memory.

Frank is designed as a **companion to an orchestrator agent** — the orchestrator (Claude, opencode, GPT, or any MCP-compatible host) handles planning, sequencing, and user interaction. It delegates the deep code work to Frank via four mode-scoped tools: `research` (read-only), `code` (implement), `test` (verify), and `performance_debug` (measure). Each tool invocation spins up a one-shot agent run against the target project.

Frank can also be used directly from the CLI as an interactive REPL or one-shot command.

---

## Quick start

### 1. Install dependencies

Python 3.10+ is required. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates a `.venv` and installs the pinned dependencies (`requests`, `mcp`, `psutil`) from the committed `uv.lock`. Run the agent with `uv run python main.py`, or activate `.venv` and run `python main.py` directly.

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

This exposes four tools over stdio MCP transport:

- **`research(prompt, working_dir)`** — read-only investigation. Returns natural-language findings.
- **`code(prompt, working_dir)`** — implement changes. Returns JSON with files-changed and verification status.
- **`test(prompt, working_dir)`** — run tests and debug. Returns JSON with test results.
- **`performance_debug(prompt, working_dir)`** — measure-first performance analysis of a target project: resource usage, wall time, hotspots, allocation sites, iteration counts, and nesting depth. Returns a measured report as a JSON envelope with status, answer, files_changed, and duration.

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

```bash
# Interactive REPL
uv run python main.py

# One-shot task (final answer to stdout, telemetry to stderr)
uv run python main.py -p "Find all uses of the deprecated API and list the files"

# One-shot with JSON result envelope (for scripting / piping)
echo "Add a docstring to foo()" | uv run python main.py -p - --json

# Resume a previous session
uv run python main.py --list-sessions
uv run python main.py --session 2026-07-11T14-30-00-12345
```

Run from inside the target project directory — the launch CWD becomes the sandboxed project root.

---

## How it works

Frank treats every IDE operation as an LLM-driveable tool. Only `load_tool` is loaded by default; the system message advertises every other tool as `name(params): summary`, and the LLM calls `load_tool(name)` to activate what it needs. This keeps context lean on any given turn.

**Tool categories:**
- **Navigation**: `find_symbol`, `go_to_definition`, `find_references`, `call_hierarchy`, `hover`, `document_symbols`, `signature_help`
- **Editing**: `replace_one`, `replace_many`, `edit_lines`, `update_file`, `create_file`, `rename_symbol`, `code_actions`, `format`, `move_file`
- **Terminal**: `run_command`, `read_output`, `stop_process`
- **Testing**: `run_tests`, `verify_scratch`
- **Debugging** (DAP): `set_breakpoint`, `debug_start`, `debug_control`, `debug_inspect`, `debug_stop`
- **Profiling**: `profile_command`, `profile_hotspots`, `profile_memory`, `trace_execution`
- **Web**: `web_search`, `web_read`
- **Memory**: `remember`, `recall`, `forget`, `record`
- **Subagents**: `spawn_agents` (fan out independent tasks)
- **Search**: `find` (ripgrep), `find_files`, `list_files`

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

If your host doesn't reset on progress, the workaround is to use the CLI directly (`uv run python main.py -p - --json`) which has no timeout, or keep MCP tool calls small enough to finish within the host's window.

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
mcp_server.py       MCP server (research / code / test / performance_debug tools)
agent.py            Agent loop (message handling, tool dispatch, loop guards)
config.py           Config loading and validation
llm.py              OpenAI-compatible chat-completions client
session.py          Session lifecycle, transcript persistence, compaction
compaction.py       Reactive context compaction
jsonrpc.py          JSON-RPC framing for LSP/DAP
diagnostics.py      Diagnostic store (LSP publishDiagnostics)
ui.py               Terminal output formatting
stats.py            Per-session usage telemetry
tools/              One file per tool (auto-discovered via registry.py)
lsp/                LSP client + manager (language servers)
dap/                DAP client + manager (debug adapters)
runtime/            Process runner, test runner, web fetch
memory/             Per-project memory (atoms, graph, recall, consolidation)
samples/            Sample projects for testing (Python, JS, PHP)
evals/              Evaluation harness and scenarios
```

## License

MIT
