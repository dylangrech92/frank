"""Network-requests tool: every response captured so far."""

from __future__ import annotations

from typing import Any

from runtime.browser import get_session
from tools.base import Tool
from tools.result import ToolResult


class NetworkRequests(Tool):
    """Report every network response captured since the session started."""

    name = "network_requests"
    description = (
        "Return every HTTP response the page has received since the browser "
        "session started — method, URL, status, and resource type — one per "
        "line, in the order they occurred. Nothing is truncated."
    )
    action = "list network requests"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the network_requests tool.

        Returns:
            A ``ToolResult`` whose body lists every captured response, one per
            line: ``METHOD URL STATUS RESOURCE_TYPE``.
        """
        session = get_session()
        session.page  # ensure the session (and its listeners) has started

        lines = [
            f"{r['method']} {r['url']} {r['status']} {r['resource_type']}"
            for r in session.responses
        ]
        body = "\n".join(lines) if lines else "no responses captured yet."
        return ToolResult.ok(body, count=len(lines))
