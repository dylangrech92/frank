"""Select-option tool: choose an option in a ref-tagged <select>."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from runtime.browser import get_session
from tools._ref import locator_for_ref
from tools._retry import retry_once_on_timeout
from tools.base import Tool
from tools.result import ToolResult


class SelectOption(Tool):
    """Select an option (by value) in the element tagged with a given ``data-qa-ref``."""

    name = "select_option"
    description = (
        "Select the option whose value is *value* in the <select> element tagged "
        "with *ref* (from the last snapshot)."
    )
    action = "select the option"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "The data-qa-ref of the <select> element, from snapshot().",
            },
            "value": {
                "type": "string",
                "description": "The value attribute of the option to select.",
            },
        },
        "required": ["ref", "value"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the select_option tool.

        Args:
            ref: The data-qa-ref of the <select> element.
            value: The value attribute of the option to select.

        Returns:
            A ``ToolResult`` confirming the selection, or an error when the ref
            is stale or the selection fails/times out.
        """
        ref = kwargs.get("ref")
        value = kwargs.get("value")
        if not isinstance(ref, str) or not ref:
            return ToolResult.err(
                "ref is required and must be a non-empty string.", code="bad-arguments"
            )
        if not isinstance(value, str):
            return ToolResult.err("value is required and must be a string.", code="bad-arguments")

        page = get_session().page
        locator = locator_for_ref(page, ref)
        if locator is None:
            return ToolResult.err(
                f"no element with ref {ref!r} on the current page.",
                code="stale-ref",
                hint="re-run snapshot to get fresh refs",
            )

        try:
            retry_once_on_timeout(lambda: locator.select_option(value))
        except PlaywrightTimeoutError as exc:
            return ToolResult.err(
                f"select_option on {ref!r} timed out: {exc}", code="action-timeout"
            )
        except Exception as exc:
            return ToolResult.err(
                f"select_option on {ref!r} failed: {exc}", code="action-failed"
            )

        return ToolResult.ok(f"selected {value!r} on {ref!r}", url=page.url, ref=ref)
