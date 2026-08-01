"""Press tool: send a keyboard key press to whichever element has focus."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from runtime.browser import get_session
from tools._retry import retry_once_on_timeout
from tools.base import Tool
from tools.result import ToolResult


class Press(Tool):
    """Press a keyboard key on the currently focused element."""

    name = "press"
    description = (
        "Press *key* on the keyboard, targeting whichever element currently has "
        "focus (not ref-scoped). Accepts Playwright key names such as 'Enter', "
        "'Tab', 'Escape', 'ArrowDown', or a single character."
    )
    action = "press the key"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The key to press (Playwright key name, e.g. 'Enter').",
            },
        },
        "required": ["key"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the press tool.

        Args:
            key: The Playwright key name to press.

        Returns:
            A ``ToolResult`` confirming the key press, or an error when it
            fails/times out.
        """
        key = kwargs.get("key")
        if not isinstance(key, str) or not key:
            return ToolResult.err(
                "key is required and must be a non-empty string.", code="bad-arguments"
            )

        page = get_session().page

        try:
            retry_once_on_timeout(lambda: page.keyboard.press(key))
        except PlaywrightTimeoutError as exc:
            return ToolResult.err(f"press {key!r} timed out: {exc}", code="action-timeout")
        except Exception as exc:
            return ToolResult.err(f"press {key!r} failed: {exc}", code="action-failed")

        return ToolResult.ok(f"pressed {key!r}", url=page.url)
