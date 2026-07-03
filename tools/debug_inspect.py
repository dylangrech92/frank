"""debug_inspect tool — evaluate expressions, list scoped variables, or read the call stack."""

from __future__ import annotations

import os
from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class DebugInspect(Tool):
    """Inspect the paused debug session: evaluate an expression, list variables in scope, or read the call stack at the current stop."""

    name = "debug_inspect"
    description = (
        'Inspect the paused debug session: evaluate an expression, list variables in scope, '
        'or read the call stack at the current stop.'
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["evaluate", "variables", "stack"],
                "description": "Inspection mode.",
            },
            "expression": {
                "type": "string",
                "description": "Expression to evaluate (required when mode == evaluate).",
            },
            "scope": {
                "type": "string",
                "description": "Filter scopes by name substring (for variables mode).",
            },
        },
        "required": ["mode"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.DEBUG_MANAGER is None:
            return ToolResult.err(
                "debugger not initialised",
                code="debug-unavailable",
            )

        mgr = main_module.DEBUG_MANAGER

        if not mgr.active:
            return ToolResult.err(
                "no active debug session",
                code="debug-no-session",
            )

        mode = kwargs.get("mode")
        root = mgr._root

        if mode == "evaluate":
            expr = kwargs.get("expression")
            if not expr:
                return ToolResult.err(
                    "evaluate mode requires 'expression'",
                    code="debug-bad-args",
                )
            try:
                r = mgr.evaluate(expr)
            except Exception as exc:
                return ToolResult.err(
                    f"evaluate failed: {exc}",
                    code="debug-error",
                )
            body = f"{expr} = {r.get('result')}"
            if r.get("type"):
                body += f"  ({r.get('type')})"
            return ToolResult.ok(body)

        if mode == "variables":
            scope = kwargs.get("scope")
            try:
                v = mgr.variables(scope)
            except Exception as exc:
                return ToolResult.err(
                    f"variables failed: {exc}",
                    code="debug-error",
                )
            if not v:
                return ToolResult.ok("no variables in scope")
            lines = []
            for scope_name, vars_ in v.items():
                lines.append(f"[{scope_name}]")
                if not vars_:
                    lines.append("  (empty)")
                for name, val in vars_.items():
                    lines.append(f"  {name} = {val}")
            return ToolResult.ok("\n".join(lines))

        if mode == "stack":
            frames = mgr.stack()
            if not frames:
                return ToolResult.ok("no call stack (not paused)")
            lines = ["call stack (top first):"]
            for i, fr in enumerate(frames):
                f = fr.get("file")
                try:
                    rel = os.path.relpath(f, root) if f else "?"
                except Exception:
                    rel = f or "?"
                lines.append(f"  #{i} {fr.get('name','?')} at {rel}:{fr.get('line')}")
            return ToolResult.ok("\n".join(lines), depth=len(frames))

        return ToolResult.err(
            f"unknown mode: {mode}",
            code="debug-bad-args",
        )
