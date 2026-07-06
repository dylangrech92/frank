"""debug_control tool — resume or step a paused debug session."""

from __future__ import annotations

import os
from typing import Any

from tools.base import Tool
from tools.result import ToolResult


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


class DebugControl(Tool):
    """Resume or step a paused debug session: continue, step_over, step_into, step_out, or pause."""

    name = "debug_control"
    summary = 'Resume or step a paused debug session.'
    description = (
        'Resume or step a paused debug session: continue, step_over, step_into, step_out, or pause. '
        'Reports where execution stops next (or program termination).'
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["continue", "step_over", "step_into", "step_out", "pause"],
                "description": "Debug control action to perform.",
            },
        },
        "required": ["action"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.DEBUG_MANAGER is None:
            return ToolResult.err(
                "debugger not initialised",
                code="debug-unavailable",
            )

        action = kwargs["action"]

        if not main_module.DEBUG_MANAGER.active:
            return ToolResult.err(
                "no active debug session",
                code="debug-no-session",
            )

        try:
            d = main_module.DEBUG_MANAGER.control(action)
        except ValueError as exc:
            return ToolResult.err(str(exc), code="debug-bad-action")
        except Exception as exc:
            return ToolResult.err(f"debug {action} failed: {exc}", code="debug-error")

        root = main_module.DEBUG_MANAGER._root
        return ToolResult.ok(_format_stop(d, root), state=d.get("state", "?"))
