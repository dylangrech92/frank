"""Console-logs tool: every console message and page error captured so far."""

from __future__ import annotations

from typing import Any

from runtime.browser import get_session
from tools.base import Tool
from tools.result import ToolResult


class ConsoleLogs(Tool):
    """Report every console message and page error captured since the session started."""

    name = "console_logs"
    description = (
        "Return every console message (log/warn/error/etc, full text, source "
        "location) and every uncaught page error captured since the browser "
        "session started, in the order they occurred. Nothing is truncated."
    )
    action = "list console logs"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the console_logs tool.

        Returns:
            A ``ToolResult`` whose body lists every captured console message and
            page error, full text, one per line.
        """
        session = get_session()
        session.page  # ensure the session (and its listeners) has started

        lines: list[str] = []
        for message in session.console_messages:
            location = message["location"] or {}
            loc_url = location.get("url", "")
            line_no = location.get("lineNumber", location.get("line", 0))
            col_no = location.get("columnNumber", location.get("column", 0))
            lines.append(f"[console.{message['type']}] {message['text']} ({loc_url}:{line_no}:{col_no})")
        for error in session.page_errors:
            lines.append(f"[pageerror] {error}")

        body = "\n".join(lines) if lines else "no console messages or page errors captured yet."
        return ToolResult.ok(body, count=len(lines))
