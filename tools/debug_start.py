"""debug_start tool — launch a debug session for a target script."""

from __future__ import annotations

import os
from typing import Any

from tools.base import Tool
from tools.result import ToolResult
from dap.manager import DebugUnavailableError


def _format_stop(d: dict, root: str) -> str:
    state = d.get("state")
    if state == "terminated":
        return "program terminated"
    loc = d.get("location") or {}
    f = loc.get("file")
    ln = loc.get("line")
    try:
        rel = os.path.relpath(f, root) if f else "?"
    except Exception:
        rel = f or "?"
    lines = [f"stopped ({d.get('reason','?')}) at {rel}:{ln}"]
    stack = d.get("stack") or []
    if stack:
        lines.append("call stack (top first):")
        for i, fr in enumerate(stack[:8]):
            try:
                frel = os.path.relpath(fr.get('file'), root) if fr.get('file') else '?'
            except Exception:
                frel = fr.get('file') or '?'
            lines.append(f"  #{i} {fr.get('name','?')} at {frel}:{fr.get('line')}")
    return "\n".join(lines)


class DebugStart(Tool):
    """Start a debug session for a script at ``path`` (or a raw launch-config dict).

    Runs to the first breakpoint or program end and reports where execution stopped,
    with the call stack.
    """

    name = "debug_start"
    summary = 'Start a debug session for a script, run to first breakpoint.'
    description = (
        'Start a debug session for a script at path (or a raw launch-config dict). '
        'Runs to the first breakpoint or program end and reports where execution stopped, '
        'with the call stack.'
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the script to debug (relative to project root or absolute).",
            },
            "config": {
                "type": "object",
                "description": "A raw DAP launch-config dict (advanced; used instead of path).",
            },
            "language": {
                "type": "string",
                "description": "Debug adapter language. If omitted, it is inferred from the path's file extension (e.g. .py->python, .php->php, .js->javascript); an unknown extension yields no adapter and a clear refusal.",
            },
        },
    }

    def run(self, **kwargs: Any) -> ToolResult:
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.DEBUG_MANAGER is None:
            return ToolResult.err(
                "debugger not initialised",
                code="debug-unavailable",
            )

        target = kwargs.get("path")
        config = kwargs.get("config")
        language = kwargs.get("language")
        if not language:
            if isinstance(target, str) and target:
                ext = os.path.splitext(target)[1].lstrip(".").lower()
                language = {
                    "py": "python",
                    "php": "php",
                    "js": "javascript",
                    "mjs": "javascript",
                    "cjs": "javascript",
                    "ts": "typescript",
                }.get(ext, ext or "python")
            else:
                language = "python"

        try:
            d = main_module.DEBUG_MANAGER.start(target=target, config=config, language=language)
        except DebugUnavailableError as exc:
            return ToolResult.err(str(exc), code="debug-adapter-unavailable")
        except Exception as exc:
            return ToolResult.err(f"debug session failed to start: {exc}", code="debug-error")

        root = main_module.DEBUG_MANAGER._root
        return ToolResult.ok(_format_stop(d, root), state=d.get("state", "?"))
