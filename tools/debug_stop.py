"""debug_stop tool — tear down the current debug session."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class DebugStop(Tool):
    """Stop the current debug session and tear down the adapter cleanly."""

    name = "debug_stop"
    description = (
        "Stop the current debug session and tear down the adapter cleanly."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    def run(self, **kwargs: Any) -> ToolResult:
        import main as main_module  # pylint: disable=import-outside-toplevel

        if main_module.DEBUG_MANAGER is None:
            return ToolResult.err(
                "debugger not initialised",
                code="debug-unavailable",
            )

        if not main_module.DEBUG_MANAGER.active:
            return ToolResult.ok("no active debug session")

        main_module.DEBUG_MANAGER.stop()
        return ToolResult.ok("debug session stopped")
