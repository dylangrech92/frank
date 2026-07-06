"""Web search tool: DuckDuckGo text search via the ddgs library with rate-limit backoff.

Searches the web using DuckDuckGo's ``ddgs`` CLI backend, enforces a minimum cooldown
between calls to avoid aggressive throttling, and returns up to eight results as a
readable numbered list of title, URL, and snippet blocks.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from tools.base import Tool
from tools.result import ToolResult

# DuckDuckGo throttles aggressively; enforce a minimum two-second gap between calls.
_DDG_COOLDOWN = 2.0
_ddg_last_call = 0.0
_ddg_lock = threading.Lock()


def _enforce_cooldown() -> None:
    """Sleep until at least ``_DDG_COOLDOWN`` seconds have passed since the last call."""
    global _ddg_last_call

    with _ddg_lock:
        elapsed = time.time() - _ddg_last_call
        if elapsed < _DDG_COOLDOWN:
            time.sleep(_DDG_COOLDOWN - elapsed)
        _ddg_last_call = time.time()


def _transform(raw: list[dict]) -> list[dict]:
    """Convert raw ``ddgs`` output dicts to standard result dicts, deduplicating by URL.

    Args:
        raw: The unprocessed list of dicts returned by ``DDGS().text()``.

    Returns:
        A deduplicated list with ``title``, ``snippet``, and ``url`` keys.
    """
    seen: set[str] = set()
    results: list[dict] = []

    for r in raw:
        url = (r.get("href") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = re.sub(r"\s{2,}", " ", (r.get("body") or "").strip())
        results.append({
            "title": (r.get("title") or "").strip(),
            "snippet": snippet,
            "url": url,
        })

    return results


class WebSearch(Tool):
    """Web search via DuckDuckGo using the ``ddgs`` library.

    Validates a non-empty query string, calls DDGS with rate-limit backoff
    (up to three attempts), and returns up to eight formatted result blocks.
    """

    name = "web_search"
    summary = 'Search the web (DuckDuckGo) for top results.'
    description = (
        "Search the web (DuckDuckGo) and return the top results as titles, URLs, and snippets."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return, 1-8. Default 5.",
            },
        },
        "required": ["query"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute a DuckDuckGo web search and return formatted results.

        Args:
            **kwargs: Parsed from LLM function-call payload. Requires ``query`` (str). \
Optional ``max_results`` (int, 1-8, default 5).

        Returns:
            A ``ToolResult`` with a numbered result list on success, or an error \
explaining why the search could not proceed.
        """
        # --- validate query --------------------------------------------------
        raw_query = kwargs.get("query")
        if not isinstance(raw_query, str) or not raw_query.strip():
            return ToolResult.err(
                "query is required and must be a non-empty string.",
                code="bad-arguments",
            )

        query: str = raw_query.strip()

        # --- resolve max_results ---------------------------------------------
        raw_max = kwargs.get("max_results")
        if isinstance(raw_max, int):
            limit = max(1, min(8, raw_max))
        else:
            limit = 5

        # --- lazy import -----------------------------------------------------
        try:
            from ddgs import DDGS  # pylint: disable=import-outside-toplevel
            from ddgs.exceptions import DDGSException, RatelimitException  # pylint: disable=import-outside-toplevel
        except ImportError:
            return ToolResult.err(
                "the ddgs library is not installed; web search is unavailable.",
                code="web-unavailable",
            )

        # --- attempt up to 3 calls with exponential backoff ------------------
        last_exc = "unknown"
        for attempt in range(3):
            _enforce_cooldown()

            try:
                raw = list(DDGS().text(query, max_results=limit))
                results = _transform(raw)
                break
            except RatelimitException as exc:
                last_exc = str(exc)
                time.sleep(2 ** attempt * 3)
            except DDGSException as exc:
                return ToolResult.err(f"web search failed: {exc}", code="web-error")
            except Exception as exc:
                return ToolResult.err(f"web search failed: {exc}", code="web-error")
        else:
            # all three attempts were rate-limited (or hit the same error class)
            return ToolResult.err(
                "web search rate-limited by DuckDuckGo; try again shortly.",
                code="web-error",
            )

        # --- handle empty results --------------------------------------------
        if not results:
            return ToolResult.ok(f"no results for '{query}'", count=0)

        # --- build formatted body --------------------------------------------
        body_lines: list[str] = []
        for i, r in enumerate(results, start=1):
            body_lines.append(f"{i}. {r['title']}")
            body_lines.append(f"   {r['url']}")
            body_lines.append(f"   {r['snippet']}")
        body_lines.append(f"-- {len(results)} result(s)")
        body: str = "\n".join(body_lines)

        return ToolResult.ok(body, count=len(results))
