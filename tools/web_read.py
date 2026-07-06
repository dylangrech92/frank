"""Web read tool: fetch a public http(s) URL behind the SSRF guard and return clean extracted text."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests

from runtime.web import FetchBlocked, fetch_page
from tools.base import Tool
from tools.result import ToolResult


class WebRead(Tool):
    """Fetch a public http(s) URL and return its readable text content.

    HTML is cleaned via ``trafilatura`` when available (lazily imported).  Private
    and internal addresses are blocked by the SSRF guard in :mod:`runtime.web`.
    """

    name = "web_read"
    summary = 'Fetch a public URL and return its readable text.'
    description = (
        "Fetch a public http(s) URL and return its readable text content "
        "(HTML is cleaned and extracted). Private and internal addresses are blocked."
    )
    action = 'fetch the page'
    oversize_hint = 'request a more specific URL or section'
    parallel_safe = True  # network only, no shared state
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "The http(s) URL to read.",
            },
            "max_chars": {
                "type": "integer",
                "description": (
                    "Optionally cap the extracted text to this many characters "
                    "(absolute cap 200000). When omitted, the full extracted text is "
                    "returned."
                ),
            },
        },
        "required": ["source"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the web-read tool.

        Args:
            source: The http(s) URL to fetch and extract text from.
            max_chars: Optional character cap (absolute cap 200000). When omitted,
                the full extracted text is returned.

        Returns:
            A ``ToolResult`` with the extracted text or an error description.
        """
        # -------------------------------------------------------------------
        # 1. Validate source
        # -------------------------------------------------------------------
        raw_source = kwargs.get("source") if isinstance(kwargs.get("source"), str) else ""

        if not raw_source:
            return ToolResult.err(
                "source is required and must be a non-empty string.",
                code="bad-arguments",
            )

        parsed = urlparse(raw_source)
        if parsed.scheme not in ("http", "https"):
            return ToolResult.err(
                f"unsupported URL scheme {parsed.scheme!r}; only http and https are allowed.",
                code="bad-arguments",
            )

        # -------------------------------------------------------------------
        # 2. max_chars — only caps when the caller explicitly asks for it
        # -------------------------------------------------------------------
        max_chars_raw = kwargs.get("max_chars")
        max_chars: int | None = None
        if isinstance(max_chars_raw, int) and max_chars_raw > 0:
            max_chars = min(max_chars_raw, 200000)

        # -------------------------------------------------------------------
        # 3. Fetch
        # -------------------------------------------------------------------
        try:
            text, content_type = fetch_page(raw_source)
        except FetchBlocked as exc:
            return ToolResult.err(str(exc), code="ssrf-blocked")
        except requests.exceptions.SSLError as exc:
            return ToolResult.err(
                f"TLS certificate verification failed: {exc}",
                code="tls-error",
            )
        except requests.exceptions.HTTPError as exc:
            return ToolResult.err(
                f"HTTP error fetching {raw_source}: {exc}",
                code="web-error",
            )
        except requests.exceptions.RequestException as exc:
            return ToolResult.err(
                f"failed to fetch {raw_source}: {exc}",
                code="web-error",
            )

        # -------------------------------------------------------------------
        # 4. Extraction
        # -------------------------------------------------------------------
        is_html = "html" in content_type if content_type else False
        if not is_html and text.strip():
            is_html = text.strip().startswith("<")

        extracted: str | None = None
        if is_html:
            try:
                # Lazy import to avoid hard dependency on trafilatura
                from trafilatura import extract as tra_extract  # type: ignore[import-not-found, no-redef]  # noqa: E501 pylint: disable=import-outside-toplevel
            except ImportError:
                extracted = None
            else:
                result = tra_extract(text, url=raw_source, include_comments=False, include_links=True)
                if result and result.strip():
                    extracted = result.strip()

        if extracted is not None:
            text = extracted

        # -------------------------------------------------------------------
        # 5. Normalize — collapse runs of 3+ newlines down to 2
        # -------------------------------------------------------------------
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # -------------------------------------------------------------------
        # 6. Empty result
        # -------------------------------------------------------------------
        if not text:
            return ToolResult.ok(
                f"no readable text content at {raw_source}",
                url=raw_source,
                content_type=content_type,
                chars=0,
            )

        # -------------------------------------------------------------------
        # 7. Cap only when the caller explicitly requested max_chars
        # -------------------------------------------------------------------
        truncation_marker = ""
        if max_chars is not None and len(text) > max_chars:
            text, actually_truncated = truncate_text_at(text, max_chars)
            if actually_truncated:
                truncation_marker = f"\n[truncated at {max_chars} characters]"

        final_body = text + truncation_marker

        return ToolResult.ok(
            final_body,
            url=raw_source,
            content_type=content_type,
            chars=len(final_body),
        )


def truncate_text_at(text: str, limit: int) -> tuple[str, bool]:
    """Return *(text[:limit], True)* if truncated, else *(text, False)*.

    Args:
        text: The text to potentially truncate.
        limit: Maximum allowed length.

    Returns:
        A ``(truncated_text, was_truncated)`` tuple.
    """
    if len(text) <= limit:
        return text, False
    return text[:limit], True
