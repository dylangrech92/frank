"""Fill tool: type text into a ref-tagged input/textarea."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from runtime.browser import get_session
from tools._ref import locator_for_ref
from tools._retry import retry_once_on_timeout
from tools.base import Tool
from tools.result import ToolResult


class Fill(Tool):
    """Fill the element tagged with a given ``data-qa-ref`` with text."""

    name = "fill"
    description = (
        "Fill the element tagged with *ref* (from the last snapshot) with *text*, "
        "replacing any existing value. Works on inputs, textareas, and "
        "contenteditable elements."
    )
    action = "fill the element"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "The data-qa-ref of the element to fill, from snapshot().",
            },
            "text": {
                "type": "string",
                "description": "The text to fill the element with.",
            },
        },
        "required": ["ref", "text"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the fill tool.

        Args:
            ref: The data-qa-ref of the element to fill.
            text: The text to fill the element with.

        Returns:
            A ``ToolResult`` confirming the fill, or an error when the ref is
            stale or the fill fails/times out.
        """
        ref = kwargs.get("ref")
        text = kwargs.get("text")
        if not isinstance(ref, str) or not ref:
            return ToolResult.err(
                "ref is required and must be a non-empty string.", code="bad-arguments"
            )
        if not isinstance(text, str):
            return ToolResult.err("text is required and must be a string.", code="bad-arguments")

        page = get_session().page
        locator = locator_for_ref(page, ref)
        if locator is None:
            return ToolResult.err(
                f"no element with ref {ref!r} on the current page.",
                code="stale-ref",
                hint="re-run snapshot to get fresh refs",
            )

        try:
            retry_once_on_timeout(lambda: locator.fill(text))
        except PlaywrightTimeoutError as exc:
            return ToolResult.err(f"fill on {ref!r} timed out: {exc}", code="action-timeout")
        except Exception as exc:
            return ToolResult.err(f"fill on {ref!r} failed: {exc}", code="action-failed")

        return ToolResult.ok(f"filled {ref!r} with {text!r}", url=page.url, ref=ref)
