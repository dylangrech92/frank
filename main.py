"""CLI entry point for coding-agent: a REPL that connects an LLM to a project."""

import argparse
import os
import sys

from agent import handle_user_message
from config import Config, load as config_load
from dap.manager import DAPManager, DebugUnavailableError
from lsp.manager import LSPManager, LSPUnavailableError
from llm import LLMClient
from session import Session
from runtime.process import reap_all
from tools.registry import discover
from diagnostics import STORE

# When run as a script this module is "__main__"; alias it as "main" so that
# `import main` inside tools resolves to this running instance, not a copy.
sys.modules.setdefault("main", sys.modules[__name__])


# =============================================================================
# Module-level hook hooks — no-op seams for later phases
# =============================================================================


def session_start_jobs(session: Session, client: LLMClient) -> None:
    """Run once after the session is created.

    Registers the always-on rules provider and the gated flashback provider, then
    runs long-term-memory maintenance: evict stale episodes, purge expired TTL
    facts, and mine any not-yet-extracted episode gists into durable facts (a
    catch-up sweep for episodes left unmined by a prior session). Every step is
    isolated so one failure never blocks session startup.
    """
    try:
        from memory.graph import register_graph_provider

        register_graph_provider()
    except Exception:
        pass

    try:
        from memory.flashback import register_flashback_provider

        register_flashback_provider()
    except Exception:
        pass

    try:
        from memory.recall import get_memory
        import memory.episodic as episodic
        import memory.atomic as atomic

        ctx = get_memory(session.project_root)
        try:
            evicted = episodic.evict_episodes(ctx.store)
            if evicted:
                print(f"episodic: evicted {evicted} stale episode(s)", file=sys.stderr)
        except Exception as exc:
            print(f"session-start-evict-error: {exc}", file=sys.stderr)
        try:
            purged = atomic.purge_expired(ctx)
            if purged:
                print(f"facts: purged {purged} expired atom(s)", file=sys.stderr)
        except Exception as exc:
            print(f"session-start-purge-error: {exc}", file=sys.stderr)
        try:
            stats = atomic.extract_facts(ctx, client)
            if stats.get("processed"):
                print(f"facts: start-of-session extractor {stats}", file=sys.stderr)
        except Exception as exc:
            print(f"session-start-extract-error: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"session-start-memory-error: {exc}", file=sys.stderr)


def session_end_jobs(session: Session, client: LLMClient) -> None:
    """Run once when the REPL exits: drain the episodic write queue so an in-flight
    extraction from the final turn is not lost, then mine the freshly-written
    episodes into durable facts before shutdown."""
    try:
        import memory.episodic as episodic
        episodic.drain_and_join()
        last = episodic.pop_last_run()
        if last is not None:
            print(
                f"episodic: final extraction ran={last.get('ran')} "
                f"stored={last.get('stored')} updated={last.get('updated')} "
                f"deleted={last.get('deleted')} reason={last.get('reason')}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"session-end-episodic-error: {exc}", file=sys.stderr)

    try:
        from memory.recall import get_memory
        import memory.atomic as atomic

        ctx = get_memory(session.project_root)
        stats = atomic.extract_facts(ctx, client)
        if stats.get("processed"):
            print(f"facts: end-of-session extractor {stats}", file=sys.stderr)
    except Exception as exc:
        print(f"session-end-extract-error: {exc}", file=sys.stderr)


# =============================================================================
# Constants
# =============================================================================

SYSTEM_PROMPT = (
    "You are a coding agent operating on the user's project through tools. "
    "Call tools when you need real information about the project. "
    "Answer conversationally otherwise."
)

# Module-level handle to the LSP manager so tools can reach it later.
MANAGER: LSPManager | None = None

# Module-level handle to the DAP debug manager so tools/repl can drive debugging.
DEBUG_MANAGER: DAPManager | None = None


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Parse args, load config, build client and session, then run the REPL loop."""
    parser = argparse.ArgumentParser(description="Coding agent CLI")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Dump the exact assembled context per LLM call and echo tool activity",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the config file (default: config.json)",
    )
    args = parser.parse_args()
    os.environ["CODING_AGENT_CONFIG"] = os.path.abspath(args.config)

    try:
        cfg: Config = config_load(args.config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    project_root: str = os.getcwd()
    discover()
    client = LLMClient(cfg.llm)
    session = Session(project_root, cfg.llm.model, SYSTEM_PROMPT)
    session_start_jobs(session, client)

    global MANAGER
    MANAGER = LSPManager(cfg.language_servers, project_root)
    MANAGER.on_client_start = lambda client: client.on_notification(
        "textDocument/publishDiagnostics", STORE.handle_publish
    )
    MANAGER.on_purge(STORE.purge)
    for line in MANAGER.prewarm():
        print(line, file=sys.stderr)

    print(
        f"Starting coding-agent on model '{session.model}' "
        f"with transcript at {session.transcript_path}",
        file=sys.stderr,
    )

    global DEBUG_MANAGER
    DEBUG_MANAGER = DAPManager(cfg.debug_adapters, project_root)

    while True:
        try:
            text: str = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = text.strip()
        if not stripped:
            continue
        if stripped in ("exit", "quit"):
            break

        try:
            assistant_text: str = handle_user_message(stripped, session, client, args.verbose, cfg.compaction)
            print(assistant_text)
        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

    reaped = reap_all()
    if reaped:
        print(
            f'Reaped {len(reaped)} background process(es).',
            file=sys.stderr,
        )

    if MANAGER is not None:
        MANAGER.shutdown_all()

    if DEBUG_MANAGER is not None and DEBUG_MANAGER.active:
        DEBUG_MANAGER.stop()

    session_end_jobs(session, client)


if __name__ == "__main__":
    main()
