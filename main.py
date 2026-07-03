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


def session_start_jobs(session: Session) -> None:
    """Run once after the session is created.

    Currently a no-op extension point.
    """
    pass


def session_end_jobs(session: Session) -> None:
    """Run once when the REPL exits.

    Currently a no-op extension point.
    """
    pass


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
    session_start_jobs(session)

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
            assistant_text: str = handle_user_message(stripped, session, client, args.verbose)
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

    session_end_jobs(session)


if __name__ == "__main__":
    main()
