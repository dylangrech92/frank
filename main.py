"""CLI entry point for coding-agent: connects an LLM to a project.

Runs either an interactive REPL (default) or a one-shot task (``-p/--prompt``)
whose final answer goes to stdout — everything else prints to stderr.
"""

import argparse
import json
import os
import sys
import threading
import time

import ui
from agent import handle_user_message
from config import Config, load as config_load
from dap.manager import DAPManager, DebugUnavailableError
from lsp.manager import LSPManager, LSPUnavailableError
from llm import LLMClient
from session import Session, list_sessions
from runtime.process import reap_all
from tools.registry import discover
from diagnostics import STORE

# When run as a script this module is "__main__"; alias it as "main" so that
# `import main` inside tools resolves to this running instance, not a copy.
sys.modules.setdefault("main", sys.modules[__name__])


# =============================================================================
# Module-level hook hooks — no-op seams for later phases
# =============================================================================


def register_catalog_provider() -> None:
    """Register the tool-catalog system-message block (deferred tool loading).

    Memory-independent — the model cannot load tools without it.
    """
    try:
        from session import CONTEXT_PROVIDERS
        from tools.registry import render_catalog_block

        CONTEXT_PROVIDERS.append(lambda _session: render_catalog_block())
    except Exception:
        pass


def session_start_jobs(session: Session) -> None:
    """Run once after the session is created.

    Registers the always-on rules provider and the orientation provider
    synchronously (cheap, needed before turn 0). There is no longer a
    background maintenance sweep here: the old episodic eviction / facts-TTL
    purge / gist-mining jobs were retired in M7
    -- ``memory.consolidation`` now owns all durable-knowledge writes,
    entirely off the hot path via its own background writer (see
    ``agent.consolidation_maybe_extract`` / ``session_end_jobs`` below).
    """
    try:
        from memory.graph import register_graph_provider

        register_graph_provider()
    except Exception:
        pass

    try:
        from memory.orientation import register_orientation_provider

        register_orientation_provider()
    except Exception:
        pass


def _repair_interrupted_turn(session: Session) -> int:
    """Repair a transcript left mid-turn by a Ctrl-C interrupt.

    ``handle_user_message`` appends the assistant's tool-calling message to the
    transcript *before* dispatching the calls one at a time (see
    ``agent.handle_user_message``), so a ``KeyboardInterrupt`` delivered while a
    tool is running (or between two tool calls in the sequential-dispatch loop)
    can leave that assistant message's ``tool_calls`` only partially answered.
    An OpenAI-format request with an assistant ``tool_calls`` entry that has no
    matching ``tool`` message for one of its call ids is rejected by strict
    providers on the very next turn, so this must be repaired before the REPL
    accepts another prompt.

    The smallest correct fix that fits ``Session``'s existing API: find the
    last assistant message carrying ``tool_calls``, determine which of its call
    ids already have a ``tool`` result appended after it, and append a
    synthetic ``[interrupted]`` tool result (via the existing
    ``append_tool_result`` method — no new ``Session`` API needed) for every
    call id still missing one, in call order.

    Args:
        session: The active session whose in-memory/on-disk transcript may be
            mid-turn.

    Returns:
        The number of synthetic tool results appended (0 when the turn was
        interrupted before any tool-calling assistant message was recorded, or
        when every call already had a result).
    """
    messages = session._messages

    last_idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            last_idx = i
            break

    if last_idx is None:
        return 0

    tool_calls = messages[last_idx]["tool_calls"]
    answered_ids = {
        m.get("tool_call_id")
        for m in messages[last_idx + 1:]
        if m.get("role") == "tool"
    }

    repaired = 0
    for call in tool_calls:
        call_id = call.get("id")
        if call_id in answered_ids:
            continue
        name = (call.get("function") or {}).get("name", "unknown")
        session.append_tool_result(
            call_id,
            name,
            "[interrupted] Tool call aborted: the turn was cancelled by the "
            "user (Ctrl-C) before this call produced a result.",
        )
        repaired += 1

    return repaired


def session_end_jobs(session: Session, client: LLMClient) -> None:
    """Run once when the REPL exits (and once, right after the answer is
    already printed, in one-shot ``-p`` mode): drain the consolidation write
    queue so the final turn's off-thread pass -- enqueued from
    ``agent.consolidation_maybe_extract`` -- is guaranteed to finish before
    shutdown. This drains rather than re-invokes ``consolidation.consolidate``
    so a one-shot task's single turn is consolidated exactly once."""
    try:
        import memory.consolidation as consolidation

        consolidation.drain_and_join()
        last = consolidation.pop_last_run()
        if last is not None:
            print(f"consolidation: final pass {last}", file=sys.stderr)
    except Exception as exc:
        print(f"session-end-consolidation-error: {exc}", file=sys.stderr)


# =============================================================================
# Constants
# =============================================================================

SYSTEM_PROMPT = (
    "You are a coding agent operating on the user's project through tools. Only "
    "`load_tool` is loaded by default; the system message lists every other tool "
    "as name(params): summary — call load_tool(name) and that tool becomes "
    "callable immediately. You are a software engineering agent: if a request is "
    "unrelated to this project or software work, say so briefly and decline "
    "rather than pursuing it.\n"
    "\n"
    "Working rules:\n"
    "- Ground claims in tool results; if you have not looked, look before answering.\n"
    "- Read code before editing it; after editing, check diagnostics before moving on.\n"
    "- Navigate large or unfamiliar code with the symbol tools: find_symbol to "
    "locate a name, then go_to_definition / find_references / call_hierarchy to "
    "trace how it is used, rather than paging through whole files with read_file "
    "just to find where something is. Read the specific range around a symbol "
    "once you have located it.\n"
    "- Fix causes, not symptoms, and prefer the smallest change that achieves the goal.\n"
    "- Never suppress or work around an error you do not understand — investigate it.\n"
    "- When a tool call errors, fix the specific problem named in the error before "
    "retrying; never resend identical arguments. If the same call fails twice, "
    "change approach or tell the user.\n"
    "- Tool outputs from earlier turns are pruned from your context; restate "
    "load-bearing paths, values, and excerpts in your replies so they survive.\n"
    "- Once you have what you need, stop calling tools and answer. When you change "
    "code, verify by running the relevant code or tests.\n"
    "- Never add verification/repro code to a production file or repurpose its "
    "`if __name__ == \"__main__\"` block; verify with verify_scratch (a throwaway "
    "snippet) or run_tests instead.\n"
    "- Always finish your turn with a plain-text answer describing what you did or "
    "found; never end on a tool call with no answer or an empty message.\n"
    "- If the request is ambiguous, state the assumption you are proceeding on in "
    "your answer.\n"
    "- Touch only what the task requires.\n"
    "\n"
    "Answer conversationally when no tool is needed."
)

# Module-level handle to the LSP manager so tools can reach it later.
MANAGER: LSPManager | None = None

# Module-level handle to the DAP debug manager so tools/repl can drive debugging.
DEBUG_MANAGER: DAPManager | None = None


def _build_envelope(
    session: Session, status: str, error: str | None, duration_s: float
) -> dict:
    """Assemble the S4 structured result envelope for one-shot ``--json`` mode.

    Reads ``session.turn_report`` -- the per-turn accumulator ``agent.handle_user_message``
    builds and keeps up to date throughout the turn (see its docstring) -- so this
    still produces a valid envelope even when *status* is ``"error"`` because the
    turn raised partway through: whatever the report accumulated up to that point
    (files already changed, verification runs already made, usage already billed)
    is reported as-is, with ``answer`` left ``None``.

    Args:
        session: The session the turn ran against; ``turn_report`` may be absent
            entirely if the turn never started (falls back to all-empty/defaults).
        status: ``"ok"`` or ``"error"``.
        error: ``None`` on success, else a short message describing the failure.
        duration_s: Wall-clock seconds the turn took, from ``time.monotonic()``.

    Returns:
        A JSON-serializable dict matching the one-shot result envelope schema.
    """
    report = getattr(session, "turn_report", None) or {}
    usage = report.get("usage") or {}
    return {
        "envelope": 1,
        "status": status,
        "error": error,
        "answer": report.get("answer"),
        "verified": report.get("verified", False),
        "declared_unverified": report.get("declared_unverified", False),
        "files_changed": report.get("files_changed", []),
        "verification_runs": report.get("verification_runs", []),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "llm_calls": usage.get("llm_calls", 0),
        },
        "session_id": session.session_id,
        "duration_s": duration_s,
    }


# =============================================================================
# Main
# =============================================================================


def _activate_cli_tools(spec: str) -> list[str]:
    """Validate and activate each comma-separated tool name; raises ValueError naming any unknown tool.

    Splits on commas, strips whitespace, skips empty entries. For each name,
    verifies it exists in the registry via ``get_tool`` (which returns ``None``
    for an unknown name, in which case this raises ``ValueError`` naming the bad
    name), then calls ``activate`` to add it to the active set.

    Args:
        spec: The raw ``--activate-tools`` value, e.g. ``"run_command,run_tests"``.

    Returns:
        The list of tool names that were activated, in input order.

    Raises:
        ValueError: If any name is not in the registry, naming that name.
    """
    from tools.registry import activate, get_tool

    names: list[str] = []
    for raw in spec.split(","):
        name = raw.strip()
        if not name:
            continue
        tool = get_tool(name)
        if tool is None:
            raise ValueError(f"unknown tool in --activate-tools: {name!r}")
        activate(name)
        names.append(name)
    return names


def main() -> None:
    """Parse args, load config, build client and session, then run the REPL or one-shot task."""
    parser = argparse.ArgumentParser(description="Coding agent CLI")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Dump the exact assembled context per LLM call and echo tool activity",
    )
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    parser.add_argument(
        "--config",
        default=default_config,
        help=f"Path to the config file (default: {default_config})",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help="Resume an existing session by id instead of starting a fresh one",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List available session ids for the current project directory and exit",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Colorize interactive output (tool calls, results, telemetry, errors)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        metavar="TEXT",
        help=(
            "One-shot mode: run this single task instead of the REPL, print the "
            "final answer to stdout, and exit (0 on success, 1 on error). "
            "Pass '-' to read the task from stdin."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "One-shot mode only (requires -p/--prompt): print exactly one JSON "
            "result envelope to stdout instead of prose. Streaming/telemetry on "
            "stderr and exit codes are unchanged."
        ),
    )
    parser.add_argument(
        "--activate-tools",
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated tool names to pre-activate into the request tools "
            "array (as if load_tool had been called), e.g. "
            "--activate-tools run_command,run_tests"
        ),
    )
    args = parser.parse_args()
    if args.json and args.prompt is None:
        parser.error("--json requires -p/--prompt")
    ui.enable(args.pretty)

    project_root: str = os.getcwd()

    if args.list_sessions:
        sessions = list_sessions(project_root)
        if not sessions:
            print("No sessions found for this project.")
        else:
            for session_id, count in sessions:
                count_str = f"{count} message(s)" if count is not None else "unknown message count"
                print(f"{session_id}  ({count_str})")
        return

    os.environ["CODING_AGENT_CONFIG"] = os.path.abspath(args.config)

    try:
        cfg: Config = config_load(args.config)
    except FileNotFoundError as exc:
        print(ui.error(str(exc)), file=sys.stderr)
        sys.exit(1)

    print(ui.telemetry(f"config: {os.path.abspath(args.config)}"), file=sys.stderr)

    discover()
    if args.activate_tools:
        try:
            activated = _activate_cli_tools(args.activate_tools)
        except ValueError as exc:
            print(ui.error(str(exc)), file=sys.stderr)
            sys.exit(2)
        print(ui.telemetry(f"activated tools: {', '.join(activated)}"), file=sys.stderr)
    client = LLMClient(cfg.llm)
    try:
        if args.session:
            session = Session.resume(project_root, cfg.llm.model, SYSTEM_PROMPT, args.session)
        else:
            session = Session(project_root, cfg.llm.model, SYSTEM_PROMPT)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(ui.error(str(exc)), file=sys.stderr)
        sys.exit(1)

    exit_code = 0
    try:
        register_catalog_provider()
        session_start_jobs(session)

        global MANAGER
        manager = LSPManager(cfg.language_servers, project_root)
        MANAGER = manager
        manager.on_client_start = lambda client: client.on_notification(
            "textDocument/publishDiagnostics", STORE.handle_publish
        )
        manager.on_purge(STORE.purge)

        def _prewarm() -> None:
            for line in manager.prewarm():
                print(ui.telemetry(line), file=sys.stderr)

        # Prewarm off the critical path: server spawns overlap with the first
        # LLM round-trip instead of delaying it. get_client is lock-guarded, so
        # a tool call racing the prewarm at worst waits for one server init.
        prewarm_thread = threading.Thread(target=_prewarm, name="lsp-prewarm", daemon=True)
        prewarm_thread.start()

        print(
            ui.telemetry(
                f"Starting coding-agent on model '{session.model}' "
                f"| session id: {session.session_id} "
                f"| transcript at {session.transcript_path}"
            ),
            file=sys.stderr,
        )

        global DEBUG_MANAGER
        DEBUG_MANAGER = DAPManager(cfg.debug_adapters, project_root)

        if args.prompt is not None:
            task = sys.stdin.read() if args.prompt == "-" else args.prompt
            task = task.strip()
            if not task:
                if args.json:
                    print(json.dumps(
                        _build_envelope(session, "error", "one-shot task is empty", 0.0)
                    ))
                else:
                    print(ui.error("one-shot task is empty"), file=sys.stderr)
                exit_code = 1
            else:
                # Stream deltas to stderr as live progress; stdout stays the
                # pure final-answer channel for the orchestrating caller (either
                # the raw prose answer, or -- with --json -- exactly one result
                # envelope) in both modes.
                def _stderr_delta(piece: str) -> None:
                    sys.stderr.write(piece)
                    sys.stderr.flush()

                start_t = time.monotonic()
                try:
                    answer: str = handle_user_message(
                        task, session, client, args.verbose, cfg.compaction,
                        on_delta=_stderr_delta,
                    )
                    # flush=True: stdout is block-buffered when piped, and
                    # teardown (consolidation drain) still runs after this —
                    # a caller-side kill in that window must not lose the
                    # already-produced answer.
                    if args.json:
                        duration_s = time.monotonic() - start_t
                        print(
                            json.dumps(_build_envelope(session, "ok", None, duration_s)),
                            flush=True,
                        )
                    else:
                        print(answer, flush=True)
                except Exception as exc:
                    duration_s = time.monotonic() - start_t
                    error_msg = f"{type(exc).__name__}: {exc}"
                    if args.json:
                        print(json.dumps(
                            _build_envelope(session, "error", error_msg, duration_s)
                        ), flush=True)
                    else:
                        print(ui.error(f"Error: {error_msg}"), file=sys.stderr)
                    exit_code = 1
        else:
            while True:
                try:
                    text: str = input(ui.prompt_marker("> "))
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                stripped = text.strip()
                if not stripped:
                    continue
                if stripped in ("exit", "quit"):
                    break

                # Stream assistant text to stdout as it arrives. The agent loop
                # guarantees the final answer reaches this sink exactly once
                # (streamed or delivered whole on fallback), so nothing is
                # printed again after handle_user_message returns.
                first_piece = True

                def _stdout_delta(piece: str) -> None:
                    nonlocal first_piece
                    if first_piece:
                        first_piece = False
                        sys.stdout.write(ui.assistant(""))
                    sys.stdout.write(piece)
                    sys.stdout.flush()

                try:
                    handle_user_message(
                        stripped, session, client, args.verbose, cfg.compaction,
                        on_delta=_stdout_delta,
                    )
                except KeyboardInterrupt:
                    # Mid-turn Ctrl-C: abandon this turn but keep the session alive.
                    # A partial streamed answer may have left stdout without a
                    # trailing newline — start the notice on its own line.
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    repaired = _repair_interrupted_turn(session)
                    note = "Turn interrupted (Ctrl-C) — back at the prompt."
                    if repaired:
                        note += f" Repaired {repaired} pending tool result(s) in the transcript."
                    print(ui.error(note), file=sys.stderr)
                    continue
                except Exception as exc:
                    print(ui.error(f"Error: {type(exc).__name__}: {exc}"), file=sys.stderr)
                    continue

        reaped = reap_all()
        if reaped:
            print(
                f'Reaped {len(reaped)} background process(es).',
                file=sys.stderr,
            )

        prewarm_thread.join(timeout=10)
        if MANAGER is not None:
            MANAGER.shutdown_all()

        if DEBUG_MANAGER is not None and DEBUG_MANAGER.active:
            DEBUG_MANAGER.stop()

        session_end_jobs(session, client)
    finally:
        session.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
