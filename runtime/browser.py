"""Singleton Playwright browser session shared by every verify-mode tool.

Sync API only — the agent loop is a plain synchronous process, no asyncio
anywhere. Call :func:`get_session` to obtain the process-wide
:class:`BrowserSession`; it starts the browser lazily on first access to
:attr:`BrowserSession.page` and tears down cleanly via
:meth:`BrowserSession.shutdown`.

``main.py`` owns the two pieces of state this module does not invent for
itself: :func:`configure` installs the ``browser`` config block, and
``run_dir`` is set to the project's ``.coding_agent/runs/<id>`` directory
before the run starts — so the transcript and the envelope's artifact paths
resolve to one real directory whether or not the browser ever starts (a pure
``http_request`` + ``report`` verification never touches Playwright).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from playwright.sync_api import (
    Browser,
    BrowserContext,
    ConsoleMessage,
    Dialog,
    Error as PlaywrightError,
    Page,
    Playwright,
    Response,
    sync_playwright,
)

# The agent's own install directory — where scripts/install_browsers.py puts
# the Chromium build. Not the target project root, which varies per run.
_INSTALL_ROOT = Path(__file__).resolve().parent.parent

# Defaults, overridden by main.py via configure() from the config's `browser`
# block. Module-level because the browser session is a process-wide singleton
# (same pattern as the LSP/DAP managers).
_headless: bool = True
_viewport_width: int = 1280
_viewport_height: int = 800


def configure(browser_cfg: dict[str, Any] | None) -> None:
    """Install the ``browser`` config block's settings for this process.

    Called once by ``main.py`` before the run starts. Unknown keys are ignored
    and a missing/empty block leaves the defaults in place, so verify mode
    works out of the box with no ``browser`` block in config.json at all.

    Args:
        browser_cfg: The config's ``browser`` block, or None when absent.
    """
    if not browser_cfg:
        return

    global _headless, _viewport_width, _viewport_height
    if isinstance(browser_cfg.get("headless"), bool):
        _headless = browser_cfg["headless"]
    if isinstance(browser_cfg.get("viewport_width"), int):
        _viewport_width = browser_cfg["viewport_width"]
    if isinstance(browser_cfg.get("viewport_height"), int):
        _viewport_height = browser_cfg["viewport_height"]


class BrowserSession:
    """The one browser session every verify-mode tool drives.

    Holds the Playwright process, browser, context, and single page, plus the
    unbounded event logs (console messages, page errors, responses, dialogs
    seen) that the read-only tools (``console_logs``, ``network_requests``,
    ``handle_dialog``) report back. Starts lazily on first access to
    :attr:`page`; :meth:`shutdown` is safe to call more than once.
    """

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._started = False

        self.run_dir: Path | None = None
        self.screenshot_counter = 0

        self.console_messages: list[dict[str, Any]] = []
        self.page_errors: list[str] = []
        self.responses: list[dict[str, Any]] = []
        self.dialogs_seen: list[dict[str, str]] = []

        # One-shot arm for the next dialog; set by the handle_dialog tool,
        # consumed (and reset) by _on_dialog the moment a dialog fires.
        self._pending_dialog_action: str | None = None
        self._pending_dialog_prompt_text: str | None = None

    @property
    def page(self) -> Page:
        """The singleton Page, starting the browser on first access."""
        if not self._started:
            self.start()
        assert self._page is not None
        return self._page

    def start(self) -> None:
        """Launch Playwright, the browser, context, and page. Idempotent.

        Raises:
            RuntimeError: If ``run_dir`` was never set — the caller (main.py)
                owns that directory, and starting without it would scatter
                traces and screenshots into an unknown location.
        """
        if self._started:
            return

        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_INSTALL_ROOT / ".browsers")

        if self.run_dir is None:
            raise RuntimeError(
                "BrowserSession.run_dir was never set — main.py must assign the "
                "run's artifact directory before any browser tool is dispatched"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=_headless)
        self._context = self._browser.new_context(
            viewport={"width": _viewport_width, "height": _viewport_height},
            device_scale_factor=1,
        )
        self._context.tracing.start(screenshots=True, snapshots=True)
        self._page = self._context.new_page()

        self._page.on("console", self._on_console)
        self._page.on("pageerror", self._on_page_error)
        self._page.on("response", self._on_response)
        self._page.on("dialog", self._on_dialog)

        self._started = True

    def arm_dialog(self, action: str, prompt_text: str | None = None) -> None:
        """Arm *action* (``"accept"`` or ``"dismiss"``) for the next dialog only."""
        self._pending_dialog_action = action
        self._pending_dialog_prompt_text = prompt_text

    def shutdown(self) -> None:
        """Stop tracing to ``run_dir/trace.zip``, close everything, stop Playwright.

        Idempotent — safe to call more than once (e.g. from a finally block
        after an earlier failure already tore things down).
        """
        if not self._started:
            return

        if self._context is not None and self.run_dir is not None:
            try:
                self._context.tracing.stop(path=str(self.run_dir / "trace.zip"))
            except Exception:
                pass

        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass

        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._started = False

    # -- event handlers, registered on the page in start() ------------------

    def _on_console(self, message: ConsoleMessage) -> None:
        self.console_messages.append(
            {"type": message.type, "text": message.text, "location": dict(message.location)}
        )

    def _on_page_error(self, error: PlaywrightError) -> None:
        message = getattr(error, "message", None) or str(error)
        stack = getattr(error, "stack", None)
        self.page_errors.append(f"{message}\n{stack}" if stack else message)

    def _on_response(self, response: Response) -> None:
        self.responses.append(
            {
                "method": response.request.method,
                "url": response.url,
                "status": response.status,
                "resource_type": response.request.resource_type,
            }
        )

    def _on_dialog(self, dialog: Dialog) -> None:
        self.dialogs_seen.append({"type": dialog.type, "message": dialog.message})

        action = self._pending_dialog_action
        prompt_text = self._pending_dialog_prompt_text
        self._pending_dialog_action = None
        self._pending_dialog_prompt_text = None

        if action == "accept":
            if prompt_text is not None:
                dialog.accept(prompt_text)
            else:
                dialog.accept()
        else:
            dialog.dismiss()


_session: BrowserSession | None = None


def get_session() -> BrowserSession:
    """Return the process-wide :class:`BrowserSession`, creating it on first call."""
    global _session
    if _session is None:
        _session = BrowserSession()
    return _session
