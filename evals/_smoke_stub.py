"""Canned stand-in for main.py, used only by ``evals/run.py --smoke``.

Prints hand-crafted stdout/stderr per scenario name so the runner's
check-evaluation and table-rendering paths can be exercised end-to-end
without ever spawning a real agent session or hitting a live LLM endpoint.
Accepts (and ignores) the same ``--config`` flag main.py takes, plus a
required ``--scenario`` flag naming which canned transcript to emit, and
consumes stdin like main.py's REPL would (ignoring its contents — the
canned output does not depend on what turns were "sent").

For the json_multiline scenario it also writes haiku.txt into the current
directory (the temp project dir), since that scenario's check asserts the
file exists on disk with a minimum line count — real main.py would have
created it via the create_file tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CANNED: dict[str, dict[str, str]] = {
    "json_multiline": {
        "stdout": (
            'haiku.txt now contains:\nCode compiles at dawn\nSyntax trees '
            'whisper their truth\nBinaries take flight\n\n- qwen\n'
        ),
        "stderr": (
            "Tool call: create_file({\"path\": \"haiku.txt\", \"content\": "
            "\"Code compiles at dawn\\nSyntax trees whisper their truth\\n"
            "Binaries take flight\\n\\n- qwen\\n\"})\n"
            "[create_file(success)]\n"
            "Tool call: read_file({\"path\": \"haiku.txt\"})\n"
            "[read_file(success)]\n"
        ),
    },
    "oversize_central": {
        "stdout": "The command's output was too large to include in context, so it was discarded rather than truncated.\n",
        "stderr": (
            "Tool call: run_command({\"command\": \"cat bigfile.py\"})\n"
            "[run_command(error code=result-too-large)]\n"
            "run_command failed to complete the operation — result is too "
            "large to fit in context — narrow the request or use a more "
            "specific tool\n"
        ),
    },
    "oversize_readfile_paging": {
        "stdout": "bigfile.py is a generated fixture module of small padding functions used to inflate file size for eval scenarios.\n",
        "stderr": (
            "Tool call: read_file({\"path\": \"bigfile.py\"})\n"
            "[read_file(error code=file-too-large)]\n"
            "hint: file too large — pass start_line/end_line to read a slice.\n"
            "Tool call: read_file({\"path\": \"bigfile.py\", \"start_line\": 1, \"end_line\": 200})\n"
            "[read_file(success)]\n"
        ),
    },
    "scope_decline": {
        "stdout": (
            "I'm a software engineering agent scoped to this codebase/project, "
            "so I'll decline planning a dinner menu — happy to help with "
            "anything code-related instead.\n"
        ),
        "stderr": "",
    },
    "search_focus_trace": {
        "stdout": (
            "Nothing in newer Python releases meaningfully simplifies "
            "mathlib.py — its functions are already minimal.\n"
        ),
        "stderr": (
            "Tool call: web_search({\"query\": \"Python 3.13 changes simplify small math helper functions\"})\n"
            "[web_search(success)]\n"
            "Tool call: web_search({\"query\": \"Python 3.13 statistics module mean function\"})\n"
            "[web_search(success)]\n"
            "Tool call: web_search({\"query\": \"Python 3.13 built-in even check\"})\n"
            "[web_search(success)]\n"
            "[focus] This is web search #3 this turn without reading any "
            "result. Searching again is unlikely to add new information — "
            "pick the most relevant result and web_read it, or answer with "
            "what you already have.\n"
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    # Consume stdin like the real REPL would, but ignore its contents — the
    # canned output is fixed per scenario regardless of what turns were sent.
    sys.stdin.read()

    canned = _CANNED.get(args.scenario, {"stdout": "", "stderr": ""})

    if args.scenario == "json_multiline":
        Path("haiku.txt").write_text(
            "Code compiles at dawn\n"
            "Syntax trees whisper their truth\n"
            "Binaries take flight\n"
            "\n"
            "- qwen\n",
            encoding="utf-8",
        )

    sys.stdout.write(canned["stdout"])
    sys.stderr.write(canned["stderr"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
