"""One-shot helper: install the Chromium build Playwright needs into ./.browsers.

Run with `uv run python scripts/install_browsers.py` whenever runtime/browser.py
reports a missing executable — sets the same PLAYWRIGHT_BROWSERS_PATH the browser
session launches from, so the download lands exactly where verify mode looks for
it rather than in the user-global Playwright cache.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_REPO_ROOT / ".browsers")
    result = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
