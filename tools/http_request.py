"""HTTP-request tool: a plain out-of-browser HTTP call (e.g. hit an API directly)."""

from __future__ import annotations

from typing import Any

import requests

from tools.base import Tool
from tools.result import ToolResult

_TIMEOUT_SECONDS = 30


class HttpRequest(Tool):
    """Issue a plain HTTP request outside the browser and return the full response."""

    name = "http_request"
    description = (
        "Issue an HTTP request with *method* to *url*, with optional *headers* "
        "and *body*, using a 30s timeout. Returns the HTTP status, content type, "
        "and the FULL response body text — never truncated. Runs outside the "
        "browser (no cookies/session shared with the page); use this to probe an "
        "API directly rather than through the UI."
    )
    action = "issue the http request"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "HTTP method, e.g. 'GET', 'POST', 'PUT', 'DELETE'.",
            },
            "url": {
                "type": "string",
                "description": "The absolute URL to request.",
            },
            "headers": {
                "type": "object",
                "description": "Optional request headers, as a flat object of string values.",
            },
            "body": {
                "type": "string",
                "description": "Optional raw request body.",
            },
        },
        "required": ["method", "url"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the http_request tool.

        Args:
            method: HTTP method, e.g. 'GET', 'POST'.
            url: The absolute URL to request.
            headers: Optional request headers.
            body: Optional raw request body.

        Returns:
            A ``ToolResult`` with the full, untruncated response body as text,
            plus status and content type in meta, or an error when the request
            fails.
        """
        method = kwargs.get("method")
        url = kwargs.get("url")
        headers = kwargs.get("headers")
        body = kwargs.get("body")

        if not isinstance(method, str) or not method:
            return ToolResult.err(
                "method is required and must be a non-empty string.", code="bad-arguments"
            )
        if not isinstance(url, str) or not url:
            return ToolResult.err(
                "url is required and must be a non-empty string.", code="bad-arguments"
            )
        if headers is not None and not isinstance(headers, dict):
            return ToolResult.err("headers must be an object when provided.", code="bad-arguments")
        if body is not None and not isinstance(body, str):
            return ToolResult.err("body must be a string when provided.", code="bad-arguments")

        try:
            response = requests.request(
                method.upper(), url, headers=headers, data=body, timeout=_TIMEOUT_SECONDS
            )
        except requests.exceptions.RequestException as exc:
            return ToolResult.err(f"http request failed: {exc}", code="http-error")

        return ToolResult.ok(
            response.text,
            status=response.status_code,
            content_type=response.headers.get("Content-Type", ""),
            url=response.url,
        )
