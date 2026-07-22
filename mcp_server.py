"""MCP server exposing coding_agent behind three mode-specific tools.

Each tool takes a ``prompt`` (simple natural-language direction from the
orchestrator) and a ``working_dir`` (the target project root).  The server
wraps the prompt with a mode-specific instruction block, invokes the coding
agent as a one-shot subprocess (``main.py -p - --json``), parses the JSON
envelope, and returns the result.

The four tools:

- ``research`` — read-only investigation (LSP, search, navigation).  Returns
  natural-language findings.
- ``code`` — make precise code changes (edit, refactor, format).  Returns JSON
  with files-changed, verification status, and a summary.
- ``test`` — run tests and debug (test runner, commands, DAP).  Returns JSON
  with verification runs and findings.
- ``performance_debug`` — profile runtime performance and report hotspots.

The mode instruction block is prepended to the user message — the agent's own
SYSTEM_PROMPT is never altered.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Context

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

AGENT_DIR = Path(__file__).resolve().parent
MAIN_PY = AGENT_DIR / "main.py"

DEFAULT_TIMEOUT_S = 1800

_STDERR_TAIL_LINES = 20

_HEARTBEAT_INTERVAL_S = 10


# ---------------------------------------------------------------------------
# Mode-specific instruction blocks
# ---------------------------------------------------------------------------

RESEARCH_MODE = """\
[MODE: RESEARCH — READ-ONLY INVESTIGATION]

You are operating in research mode.  Gather information about the codebase and
report findings.  This is strictly read-only.

Tools to use:
- find_symbol to locate names, then go_to_definition / go_to_implementation /
  find_references / call_hierarchy to trace how a symbol is used.
- find for content search, find_files for filename search.
- read_file to inspect specific ranges once you have located a symbol.
- hover for type/signature info, document_symbols for file outlines.
- signature_help for parameter hints.
- spawn_agents to fan out independent research questions concurrently.
- web_search / web_read for external documentation.

Constraints:
- DO NOT create, modify, move, or delete any files.
- DO NOT run tests or execute project code.
- Ground every claim in what you actually observed — cite file paths.

Return a clear, structured natural-language summary of your findings.
[end: MODE]\
"""

CODE_MODE = """\
[MODE: CODE — IMPLEMENT CHANGES]

You are operating in code mode.  Make precise, minimal code changes to satisfy
the task.

Tools to use:
- replace_one for unique targeted changes, replace_many for project-wide text
  replacement, edit_lines for range-based edits, update_file for full
  overwrites of small files, create_file for new files.
- rename_symbol for type-aware cross-file renames.
- code_actions for quick-fixes, organize-imports, and refactors.
- move_file to rename/move files (imports auto-update).
- format after structural changes.
- run_command for builds, type checks, and scripts (NOT for running tests).

Lint and LSP diagnostics surface automatically after each edit — resolve every
diagnostic your edits introduce and state the final diagnostics status.

Constraints:
- DO NOT run unit tests or the test suite (run_tests, pytest).  The orchestrator
  runs tests separately via the test tool.
- Touch only what the task requires — no scope creep, no unrelated refactors.

Return a file-change list: every file you changed with a one-line "what changed
and why" summary.
[end: MODE]\
"""

TEST_MODE = """\
[MODE: TEST — VERIFY AND DEBUG]

You are operating in test mode.  Verify code correctness by running tests and
debugging failures.  This is observe-and-report.

Tools to use:
- run_tests to run the test suite with structured per-test pass/fail output.
- run_command to run specific commands (builds, type checks, scripts) and
  observe their output.
- verify_scratch for throwaway verification snippets executed outside the
  project tree.
- If a test fails and the cause is not obvious from the output, use the DAP
  debugger: set_breakpoint, debug_start, debug_control (step_over / step_into /
  step_out / continue), debug_inspect (variables / stack / evaluate),
  debug_stop.

Constraints:
- DO NOT create, modify, or delete any project files.
- Report what you observe — do not fix code here.  If a fix is needed, state
  what should be changed and let the orchestrator delegate to code mode.

Return a clear summary: what passed, what failed, and the failure details (error
messages, stack traces, assertion failures).
[end: MODE]\
"""

PERFORMANCE_DEBUG_MODE = """\
[MODE: PERFORMANCE_DEBUG — MEASURE AND LOCATE BOTTLENECKS]

You are operating in performance-debug mode.  Measure how the project's code
actually performs, find the bottlenecks, and report where improvements can be
gained.  Measure first — never guess, and never optimize here.

Tools to use:
- profile_command to run any command under resource measurement: wall time,
  user/sys CPU, peak RSS, and a sampled timeline of the whole process tree.
  Use repeats=3 when you need stable wall times.
- profile_hotspots for per-function CPU profiles — self/cumulative time and
  call counts (Python cProfile, Node --cpu-prof, PHP Xdebug).  Use focus= to
  expand one function's callers and callees.
- profile_memory for allocation profiles — top allocation sites and peak
  usage (Python tracemalloc, Node --heap-prof, PHP Xdebug memory events).
- trace_execution (Python) for exact call counts, per-line hit counts in a
  chosen file, and max stack depth — the tool for hidden iteration blow-ups
  and deep recursion.
- read_file / find_symbol / find_references to read the code behind every
  hotspot before you explain it.

Constraints:
- DO NOT modify project files.  Profiled code may itself write files — the
  tool results name any files a run touched; report them.
- Profile a bounded, realistic workload that exits on its own; state the
  exact command or entry point you measured.
- Ground every claim in measured numbers from a tool result — never report a
  bottleneck you did not observe.

Return a performance report: what was measured (with wall/CPU/memory numbers),
the ranked hotspots with their numbers, and concrete improvement opportunities
tied to specific files and functions.
[end: MODE]\
"""

MODE_INSTRUCTIONS: dict[str, str] = {
    "research": RESEARCH_MODE,
    "code": CODE_MODE,
    "test": TEST_MODE,
    "performance_debug": PERFORMANCE_DEBUG_MODE,
}

# tools a mode pre-activates in the agent subprocess so every tool its
# instruction block names is guaranteed callable (passed via --activate-tools).
MODE_TOOLS: dict[str, list[str]] = {
    "performance_debug": ["profile_command", "profile_hotspots", "profile_memory", "trace_execution"],
}


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


def _wrap_prompt(mode: str, prompt: str) -> str:
    """Prepend the mode-specific instruction block to the orchestrator's prompt."""
    return f"{MODE_INSTRUCTIONS[mode]}\n\n---\n\n{prompt}"


def _stderr_tail(stderr: str) -> str:
    """Return the last N lines of stderr for error diagnostics."""
    lines = stderr.splitlines()
    if len(lines) <= _STDERR_TAIL_LINES:
        return stderr.strip()
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


async def _run_agent(
    mode: str,
    prompt: str,
    working_dir: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    ctx: Context | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Invoke the coding agent as a subprocess and return the parsed JSON envelope.

    Sends periodic progress notifications via *ctx* while the subprocess runs.
    Clients that set ``resetTimeoutOnProgress`` (e.g. opencode) will reset their
    request timeout on each notification, so the call can run for the full
    ``timeout_s`` without the client giving up.

    Args:
        mode: One of ``"research"``, ``"code"``, ``"test"``, ``"performance_debug"`` — selects the
            instruction block prepended to the prompt.
        prompt: The orchestrator's natural-language direction.
        working_dir: The target project root (becomes the agent's CWD).
        timeout_s: Wall-clock timeout for the subprocess.
        ctx: Optional MCP context for progress notifications.
        extra_args: Extra argv args to splice after ``--json`` (e.g. ``--activate-tools``).

    Returns:
        The parsed JSON envelope dict from the agent's ``--json`` output.

    Raises:
        RuntimeError: If the subprocess times out, fails to start, or produces
            unparseable output.
    """
    working_path = Path(working_dir).expanduser().resolve()
    if not working_path.is_dir():
        raise RuntimeError(f"working_dir does not exist or is not a directory: {working_path}")

    wrapped = _wrap_prompt(mode, prompt.strip())

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(MAIN_PY), "-p", "-", "--json", *(extra_args or []),
            cwd=str(working_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(os.environ),
        )
    except OSError as exc:
        raise RuntimeError(f"failed to start coding agent subprocess: {exc}") from exc

    async def _heartbeat() -> None:
        elapsed = 0
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            elapsed += _HEARTBEAT_INTERVAL_S
            if ctx is not None:
                await ctx.report_progress(
                    elapsed, timeout_s,
                    f"agent running ({elapsed}s elapsed)",
                )

    heartbeat = asyncio.create_task(_heartbeat())
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=wrapped.encode("utf-8")),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"coding agent timed out after {timeout_s}s")
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        # A nonzero exit usually still carries the real failure reason in the
        # stdout JSON envelope (main.py exits 1 on a status=error envelope) —
        # include its head so the caller sees the cause, not just telemetry.
        tail = _stderr_tail(stderr_text)
        raise RuntimeError(
            f"coding agent exited with code {proc.returncode}.\n"
            f"stdout (first 500 chars): {stdout_text[:500]}\n"
            f"stderr (tail):\n{tail}"
        )

    try:
        return json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        tail = _stderr_tail(stderr_text)
        raise RuntimeError(
            f"coding agent produced unparseable JSON output: {exc}\n"
            f"stdout (first 500 chars): {stdout_text[:500]}\n"
            f"stderr (tail):\n{tail}"
        ) from exc


def _envelope_error(envelope: dict) -> str:
    """Extract a human-readable error from a status=error envelope."""
    return envelope.get("error") or "(agent returned an error with no message)"


def _format_code_result(envelope: dict) -> str:
    """Format the envelope for code mode as a focused JSON string."""
    return json.dumps({
        "status": envelope.get("status"),
        "error": envelope.get("error"),
        "answer": envelope.get("answer"),
        "files_changed": envelope.get("files_changed", []),
        "verified": envelope.get("verified"),
        "declared_unverified": envelope.get("declared_unverified", False),
    }, indent=2)


def _format_test_result(envelope: dict) -> str:
    """Format the envelope for test mode as a focused JSON string."""
    return json.dumps({
        "status": envelope.get("status"),
        "error": envelope.get("error"),
        "answer": envelope.get("answer"),
        "verification_runs": envelope.get("verification_runs", []),
    }, indent=2)


def _format_perf_result(envelope: dict) -> str:
    """Format the envelope for performance_debug mode as a focused JSON string."""
    return json.dumps({
        "status": envelope.get("status"),
        "error": envelope.get("error"),
        "answer": envelope.get("answer"),
        "files_changed": envelope.get("files_changed", []),
        "duration_s": envelope.get("duration_s"),
    }, indent=2)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

app = FastMCP("coding-agent")


@app.tool()
async def research(prompt: str, working_dir: str, ctx: Context) -> str:
    """Investigate a codebase in read-only mode using LSP code intelligence,
    fuzzy search, and file reading.  Returns findings as natural language.

    Use for: understanding code structure, finding symbol usages, tracing call
    chains, locating implementations, answering questions about the codebase.
    Cannot modify files.

    Args:
        prompt: Natural-language research direction, e.g. "Find all subclasses
            of BaseRepository and show how they override save()".
        working_dir: Absolute path to the target project root.
    """
    try:
        envelope = await _run_agent("research", prompt, working_dir, ctx=ctx)
    except RuntimeError as exc:
        return f"Error: {exc}"

    if envelope.get("status") != "ok":
        return f"Agent error: {_envelope_error(envelope)}"

    return envelope.get("answer") or "(agent returned no answer)"


@app.tool()
async def code(prompt: str, working_dir: str, ctx: Context) -> str:
    """Make precise code changes in a project using LSP-aware editing tools,
    refactoring, and formatting.  Returns JSON with files-changed, verification
    status, and a summary of what was done.

    Does NOT run tests — use the 'test' tool for verification.

    Use for: implementing features, fixing bugs, refactoring, renaming symbols,
    applying quick-fixes, formatting.

    Args:
        prompt: Natural-language coding direction, e.g. "Merge UserRepository
            and AccountRepository into a single Repository class following SRP".
        working_dir: Absolute path to the target project root.
    """
    try:
        envelope = await _run_agent("code", prompt, working_dir, ctx=ctx)
    except RuntimeError as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)

    return _format_code_result(envelope)


@app.tool()
async def test(prompt: str, working_dir: str, ctx: Context) -> str:
    """Run tests and verify code correctness in a project.  Returns JSON with
    test results and findings.

    Can run test suites, execute commands, run verification snippets, and debug
    failures with breakpoints and stepping (DAP).

    Does NOT modify code — use the 'code' tool for changes.

    Use for: running tests, reproducing bugs, debugging failures, verifying
    changes made by the code tool.

    Args:
        prompt: Natural-language test direction, e.g. "Run the test suite in
            tests/ and report any failures with their stack traces".
        working_dir: Absolute path to the target project root.
    """
    try:
        envelope = await _run_agent("test", prompt, working_dir, ctx=ctx)
    except RuntimeError as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)

    return _format_test_result(envelope)


@app.tool()
async def performance_debug(prompt: str, working_dir: str, ctx: Context) -> str:
    """Profile a project's runtime performance and report where improvements
    can be gained.  Measures wall time, CPU, memory, per-function hotspots,
    call counts, and stack depth using native profiling tools (Python/Node/PHP).
    Returns a measured performance report plus any files the profiled runs
    touched.

    Does NOT modify code — use the 'code' tool for changes.

    Use for: finding performance bottlenecks, measuring wall/CPU/memory costs,
    identifying hot functions, profiling memory allocations, tracing call counts
    and recursion depth.

    Args:
        prompt: Natural-language profiling direction, e.g. "Profile
            scripts/import.py and find why it is slow".
        working_dir: Absolute path to the target project root.
    """
    try:
        envelope = await _run_agent(
            "performance_debug",
            prompt,
            working_dir,
            ctx=ctx,
            extra_args=["--activate-tools", ",".join(MODE_TOOLS["performance_debug"])],
        )
    except RuntimeError as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)

    return _format_perf_result(envelope)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(transport="stdio")
