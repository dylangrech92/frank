"""Scroll tool: scroll the page window in a direction."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from runtime.browser import get_session
from tools._retry import retry_once_on_timeout
from tools.base import Tool
from tools.result import ToolResult

_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

_SCROLL_SCRIPT = """
([dx, dy]) => {
  window.scrollBy(dx, dy);
  return {
    scrollX: window.scrollX,
    scrollY: window.scrollY,
    maxScrollX: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
    maxScrollY: Math.max(0, document.documentElement.scrollHeight - window.innerHeight),
  };
}
"""


class Scroll(Tool):
    """Scroll the page window by a pixel amount in a direction."""

    name = "scroll"
    description = (
        "Scroll the page window *pixels* pixels in *direction* ('up', 'down', "
        "'left', or 'right'). Returns the new scroll position and the maximum "
        "scrollable extent on each axis."
    )
    action = "scroll the page"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "The direction to scroll.",
            },
            "pixels": {
                "type": "integer",
                "description": "How many pixels to scroll. Defaults to 600.",
            },
        },
        "required": ["direction"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the scroll tool.

        Args:
            direction: One of 'up', 'down', 'left', 'right'.
            pixels: How many pixels to scroll (default 600).

        Returns:
            A ``ToolResult`` with the new scroll position and max scroll extent
            on each axis, or an error on bad arguments or failure/timeout.
        """
        direction = kwargs.get("direction")
        if direction not in _DELTAS:
            return ToolResult.err(
                f"direction must be one of {sorted(_DELTAS)}, got {direction!r}.",
                code="bad-arguments",
            )

        pixels = kwargs.get("pixels", 600)
        if not isinstance(pixels, int) or isinstance(pixels, bool) or pixels <= 0:
            return ToolResult.err("pixels must be a positive integer.", code="bad-arguments")

        unit_x, unit_y = _DELTAS[direction]
        dx, dy = unit_x * pixels, unit_y * pixels

        page = get_session().page

        try:
            result = retry_once_on_timeout(lambda: page.evaluate(_SCROLL_SCRIPT, [dx, dy]))
        except PlaywrightTimeoutError as exc:
            return ToolResult.err(f"scroll {direction} timed out: {exc}", code="action-timeout")
        except Exception as exc:
            return ToolResult.err(f"scroll {direction} failed: {exc}", code="action-failed")

        return ToolResult.ok(
            f"scrolled {direction} by {pixels}px",
            scroll_x=result["scrollX"],
            scroll_y=result["scrollY"],
            max_scroll_x=result["maxScrollX"],
            max_scroll_y=result["maxScrollY"],
        )
