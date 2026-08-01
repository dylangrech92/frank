"""Hover tool: move the mouse over a ref-tagged element."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from runtime.browser import get_session
from tools._ref import locator_for_ref
from tools._retry import retry_once_on_timeout
from tools.base import Tool
from tools.result import ToolResult


class Hover(Tool):
    """Hover the mouse over the element tagged with a given ``data-qa-ref``."""

    name = "hover_element"
    description = (
        "Move the mouse over the element tagged with *ref* (from the last "
        "snapshot), e.g. to reveal a tooltip or hover-triggered menu."
    )
    action = "hover the element"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "The data-qa-ref of the element to hover, from snapshot().",
            },
        },
        "required": ["ref"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the hover tool.

        Args:
            ref: The data-qa-ref of the element to hover.

        Returns:
            A ``ToolResult`` confirming the hover, or an error when the ref is
            stale or the hover fails/times out.
        """
        ref = kwargs.get("ref")
        if not isinstance(ref, str) or not ref:
            return ToolResult.err(
                "ref is required and must be a non-empty string.", code="bad-arguments"
            )

        page = get_session().page
        locator = locator_for_ref(page, ref)
        if locator is None:
            return ToolResult.err(
                f"no element with ref {ref!r} on the current page.",
                code="stale-ref",
                hint="re-run snapshot to get fresh refs",
            )

        try:
            retry_once_on_timeout(locator.hover)
        except PlaywrightTimeoutError as exc:
            return ToolResult.err(f"hover on {ref!r} timed out: {exc}", code="action-timeout")
        except Exception as exc:
            return ToolResult.err(f"hover on {ref!r} failed: {exc}", code="action-failed")

        return ToolResult.ok(f"hovered {ref!r}", url=page.url, ref=ref)
