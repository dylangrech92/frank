"""Click tool: click a ref-tagged element."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from runtime.browser import get_session
from tools._ref import locator_for_ref
from tools._retry import retry_once_on_timeout
from tools.base import Tool
from tools.result import ToolResult


class Click(Tool):
    """Click the element tagged with a given ``data-qa-ref``."""

    name = "click"
    description = (
        "Click the element tagged with *ref* (from the last snapshot). Returns a "
        "confirmation and the current URL, since a click may trigger a navigation."
    )
    action = "click the element"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "The data-qa-ref of the element to click, from snapshot().",
            },
        },
        "required": ["ref"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the click tool.

        Args:
            ref: The data-qa-ref of the element to click.

        Returns:
            A ``ToolResult`` confirming the click and reporting the current URL,
            or an error when the ref is stale or the click fails/times out.
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
            retry_once_on_timeout(locator.click)
        except PlaywrightTimeoutError as exc:
            return ToolResult.err(f"click on {ref!r} timed out: {exc}", code="action-timeout")
        except Exception as exc:
            return ToolResult.err(f"click on {ref!r} failed: {exc}", code="action-failed")

        return ToolResult.ok(f"clicked {ref!r}", url=page.url, ref=ref)
