"""Navigate tool: load a URL in the singleton page."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from runtime.browser import get_session
from tools._retry import retry_once_on_timeout
from tools.base import Tool
from tools.result import ToolResult


class Navigate(Tool):
    """Navigate the browser page to a URL and wait for the load event."""

    name = "navigate"
    description = (
        "Navigate the browser page to *url* and wait for the load event. Returns "
        "the final URL (after any redirects), the page title, and the HTTP status "
        "of the main document response."
    )
    action = "navigate to the page"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The absolute URL to navigate to.",
            },
        },
        "required": ["url"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the navigate tool.

        Args:
            url: The absolute URL to navigate to.

        Returns:
            A ``ToolResult`` with the final URL, title, and HTTP status, or an
            error when navigation fails or times out.
        """
        url = kwargs.get("url")
        if not isinstance(url, str) or not url:
            return ToolResult.err(
                "url is required and must be a non-empty string.", code="bad-arguments"
            )

        page = get_session().page

        try:
            response = retry_once_on_timeout(lambda: page.goto(url, wait_until="load"))
        except PlaywrightTimeoutError as exc:
            return ToolResult.err(
                f"navigation to {url!r} timed out: {exc}", code="navigation-timeout"
            )
        except Exception as exc:
            return ToolResult.err(
                f"navigation to {url!r} failed: {exc}", code="navigation-failed"
            )

        status = response.status if response is not None else None
        title = page.title()
        meta: dict[str, Any] = {"url": page.url, "title": title}
        if status is not None:
            meta["status"] = status

        return ToolResult.ok(
            f"navigated to {page.url!r} (status {status}), title: {title!r}",
            **meta,
        )
