"""Wait-for tool: block until a text/selector/URL condition is met."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from runtime.browser import get_session
from tools.base import Tool
from tools.result import ToolResult


class WaitFor(Tool):
    """Wait for at least one of text/selector/url_substring to appear."""

    name = "wait_for"
    description = (
        "Wait until at least one of the given conditions is met: *text* visible "
        "anywhere on the page, *selector* present, or the URL containing "
        "*url_substring*. At least one condition is required; when more than one "
        "is given, all of them must be met. *timeout_ms* bounds the wait for each "
        "condition (default 10000)."
    )
    action = "wait for the condition"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text that must become visible on the page.",
            },
            "selector": {
                "type": "string",
                "description": "A CSS selector that must become present on the page.",
            },
            "url_substring": {
                "type": "string",
                "description": "A substring the page URL must come to contain.",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Timeout in milliseconds for each condition. Defaults to 10000.",
            },
        },
        "required": [],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the wait_for tool.

        Args:
            text: Optional text that must become visible on the page.
            selector: Optional CSS selector that must become present.
            url_substring: Optional substring the page URL must come to contain.
            timeout_ms: Timeout in milliseconds for each condition (default 10000).

        Returns:
            A ``ToolResult`` confirming the condition(s) were met, or an error on
            bad arguments, timeout, or failure.
        """
        text = kwargs.get("text")
        selector = kwargs.get("selector")
        url_substring = kwargs.get("url_substring")
        timeout_ms = kwargs.get("timeout_ms", 10000)

        if text is None and selector is None and url_substring is None:
            return ToolResult.err(
                "at least one of text, selector, url_substring is required.",
                code="bad-arguments",
            )
        for name, value in (("text", text), ("selector", selector), ("url_substring", url_substring)):
            if value is not None and not isinstance(value, str):
                return ToolResult.err(f"{name} must be a string when provided.", code="bad-arguments")
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms <= 0:
            return ToolResult.err("timeout_ms must be a positive integer.", code="bad-arguments")

        page = get_session().page

        try:
            if url_substring is not None:
                page.wait_for_url(f"**{url_substring}**", timeout=timeout_ms)
            if selector is not None:
                page.wait_for_selector(selector, timeout=timeout_ms)
            if text is not None:
                page.get_by_text(text).first.wait_for(timeout=timeout_ms)
        except PlaywrightTimeoutError as exc:
            return ToolResult.err(
                f"wait_for timed out after {timeout_ms}ms: {exc}", code="wait-timeout"
            )
        except Exception as exc:
            return ToolResult.err(f"wait_for failed: {exc}", code="wait-failed")

        return ToolResult.ok("condition(s) met", url=page.url)
