"""MCP server exposing coding_agent behind four mode-specific tools.

Each tool takes a ``prompt`` (simple natural-language direction from the
orchestrator) and a ``working_dir`` (the target project root).  The server
invokes the coding agent as a one-shot subprocess (``main.py -p - --json
--mode <mode>``), parses the JSON envelope, and returns the result.

The four tools:

- ``research`` — read-only investigation (LSP, search, navigation).  Returns
  natural-language findings.
- ``code`` — make precise code changes (edit, refactor, format).  Returns JSON
  with files-changed, verification status, and a summary.
- ``test`` — run tests and debug (test runner, commands, DAP).  Returns JSON
  with verification runs and findings.
- ``performance_debug`` — profile runtime performance and report hotspots.

Each mode's instructions and complete tool set live in ``modes.py``. The
agent subprocess loads them itself via ``--mode``; this server forwards the
orchestrator's prompt unmodified and never alters the agent's own
SYSTEM_PROMPT.
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
# Agent invocation
# ---------------------------------------------------------------------------


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
) -> dict:
    """Invoke the coding agent as a subprocess and return the parsed JSON envelope.

    Sends periodic progress notifications via *ctx* while the subprocess runs.
    Clients that set ``resetTimeoutOnProgress`` (e.g. opencode) will reset their
    request timeout on each notification, so the call can run for the full
    ``timeout_s`` without the client giving up.

    Args:
        mode: One of ``"research"``, ``"code"``, ``"test"``, ``"performance_debug"`` — passed to
            the subprocess as ``--mode``, which loads that mode's complete tool set and
            instructions inside the agent itself.
        prompt: The orchestrator's natural-language direction, sent as-is.
        working_dir: The target project root (becomes the agent's CWD).
        timeout_s: Wall-clock timeout for the subprocess.
        ctx: Optional MCP context for progress notifications.

    Returns:
        The parsed JSON envelope dict from the agent's ``--json`` output.

    Raises:
        RuntimeError: If the subprocess times out, fails to start, or produces
            unparseable output.
    """
    working_path = Path(working_dir).expanduser().resolve()
    if not working_path.is_dir():
        raise RuntimeError(f"working_dir does not exist or is not a directory: {working_path}")

    task = prompt.strip()

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(MAIN_PY), "-p", "-", "--json", "--mode", mode,
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
            proc.communicate(input=task.encode("utf-8")),
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
        envelope = await _run_agent("performance_debug", prompt, working_dir, ctx=ctx)
    except RuntimeError as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)

    return _format_perf_result(envelope)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(transport="stdio")
