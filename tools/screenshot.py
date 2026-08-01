"""Screenshot tool: capture the viewport (or a single element) to PNG."""

from __future__ import annotations

from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from runtime.browser import get_session
from tools._ref import locator_for_ref
from tools._retry import retry_once_on_timeout
from tools.base import Tool
from tools.result import ToolResult


class Screenshot(Tool):
    """Capture a PNG screenshot of the viewport, or of a single ref-tagged element."""

    name = "screenshot"
    description = (
        "Capture a PNG screenshot. With no *ref*, captures the current viewport "
        "(never the full scrollable page). With *ref*, captures just that element. "
        "Animations are disabled for a stable capture. The image is saved to disk "
        "and its path is returned in meta.image_path for the caller to attach to "
        "the model's context."
    )
    action = "take a screenshot"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Optional data-qa-ref of a single element to screenshot instead of the viewport.",
            },
        },
        "required": [],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the screenshot tool.

        Args:
            ref: Optional data-qa-ref of a single element to screenshot instead
                of the full viewport.

        Returns:
            A ``ToolResult`` whose body confirms the capture and whose meta
            carries ``image_path`` (absolute path to the saved PNG), or an error
            when the ref is stale or the capture fails/times out.
        """
        ref = kwargs.get("ref")
        if ref is not None and (not isinstance(ref, str) or not ref):
            return ToolResult.err("ref must be a non-empty string when provided.", code="bad-arguments")

        session = get_session()
        page = session.page
        session.screenshot_counter += 1
        n = session.screenshot_counter
        assert session.run_dir is not None
        path = session.run_dir / f"shot-{n}.png"

        try:
            if ref is not None:
                locator = locator_for_ref(page, ref)
                if locator is None:
                    return ToolResult.err(
                        f"no element with ref {ref!r} on the current page.",
                        code="stale-ref",
                        hint="re-run snapshot to get fresh refs",
                    )
                retry_once_on_timeout(
                    lambda: locator.screenshot(path=str(path), animations="disabled", scale="css")
                )
            else:
                retry_once_on_timeout(
                    lambda: page.screenshot(
                        path=str(path), animations="disabled", scale="css", full_page=False
                    )
                )
        except PlaywrightTimeoutError as exc:
            return ToolResult.err(f"screenshot timed out: {exc}", code="action-timeout")
        except Exception as exc:
            return ToolResult.err(f"screenshot failed: {exc}", code="action-failed")

        return ToolResult.ok(
            f"[screenshot shot-{n}.png captured — image attached]",
            image_path=str(path),
        )
