"""OpenAI-compatible chat-completions client using only the Python standard library.

Provides ``LLMClient`` which POSTs to any OpenAI-compatible endpoint (e.g. vLLM,
Ollama with the openai suffix, LMStudio) and returns parsed ``ChatResponse`` objects.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from config import LLMConfig
from dataclasses import dataclass, field
from typing import Any, Dict, List


class OverCapError(Exception):
    """Raised when the provider rejects a request because the context is too long."""

    pass


@dataclass(frozen=True)
class ToolCall:
    """A single tool/function call returned by the model.

    Attributes:
        id: Opaque caller-assigned ID for the tool call.
        name: Name of the function to invoke.
        arguments: Parsed JSON dict from ``function.arguments`` on native success;
            falls back to an empty dict when the payload is not valid JSON.
    """

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResponse:
    """Parsed response from a single chat-completions call.

    Attributes:
        text: Concatenated content of ``choices[0].message.content``; empty string
            when the assistant returned no textual content (e.g. pure tool use).
        tool_calls: List of parsed ``ToolCall`` entries from
            ``choices[0].message.tool_calls``; empty list when none was returned.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)


def _strip_outer_braces(args_str: str) -> str:
    """Strip markdown code fences and any junk outside the outermost ``{...}``.

    Returns the substring from the first ``{`` to the last ``}`` inclusive, or
    the input unchanged when no ``{`` / ``}`` pair is found.
    """
    start = args_str.find("{")
    end = args_str.rfind("}")
    if start == -1 or end == -1 or end < start:
        return args_str
    return args_str[start:end + 1]


def _escape_raw_control_chars_in_strings(args_str: str) -> str:
    """Escape literal newlines/tabs that appear inside double-quoted string values.

    A small state-machine scan: outside of a string, characters pass through
    unchanged; inside a double-quoted string (tracking ``\\"`` escapes so a
    quote does not falsely end the string), a raw ``\\n`` becomes ``\\\\n`` and
    a raw ``\\t`` becomes ``\\\\t``.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in args_str:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _remove_trailing_commas(args_str: str) -> str:
    """Remove commas that appear immediately before a closing ``}`` or ``]``.

    Whitespace between the comma and the closing bracket is tolerated.
    """
    return re.sub(r",(\s*[}\]])", r"\1", args_str)


def _repair_json(args_str: str) -> str | None:
    """Attempt cheap, ordered repairs on a malformed tool-call arguments string.

    Each repair is applied in turn and ``json.loads`` is retried after every
    step; the first repair that yields valid JSON wins. Repairs are cumulative
    (each builds on the previous step's output) since malformed payloads often
    combine more than one issue (e.g. fenced *and* trailing-comma).

    Args:
        args_str: The raw, non-parsing arguments string.

    Returns:
        The repaired string (already verified to parse) on success, or
        ``None`` when no repair combination produces valid JSON.
    """
    candidate = args_str
    repairs = (
        _strip_outer_braces,
        _escape_raw_control_chars_in_strings,
        _remove_trailing_commas,
    )
    for repair in repairs:
        candidate = repair(candidate)
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _tool_call_from_dict(raw: Dict[str, Any]) -> ToolCall:
    """Convert a raw OpenAI-style tool-call dict into a ``ToolCall``.

    On a JSON parse failure, cheap repairs are attempted in order (see
    ``_repair_json``) and ``json.loads`` retried after each. If every repair
    fails, ``arguments`` is set to a small error-marker dict — never a silent
    ``{}`` — so ``dispatch()`` can report the malformed payload back to the
    model instead of it looking like a missing-parameter error.
    """
    func = raw.get("function", {})
    name = func.get("name", "")
    args_str = func.get("arguments", "null")

    if not isinstance(args_str, str):
        return ToolCall(id=raw["id"], name=name, arguments={"raw": args_str})

    try:
        arguments = json.loads(args_str)
    except json.JSONDecodeError as exc:
        repaired = _repair_json(args_str)
        if repaired is not None:
            arguments = json.loads(repaired)
        else:
            arguments = {
                "__json_error__": str(exc),
                "__raw__": args_str[:200],
            }

    return ToolCall(id=raw["id"], name=name, arguments=arguments)


class LLMClient:
    """Thin HTTP client for OpenAI-compatible chat-completions endpoints.

    Args:
        config: A frozen ``LLMConfig`` dataclass holding base_url, api_key, model,
            temperature, context_limit, and an optional max_tokens (None = do not
            cap output; the reactive compaction loop governs the context budget).

    Example:
        >>> from config import LLMConfig  # doctest: +SKIP
        >>> cfg = LLMConfig("https://llm.example.com/v1", "sk-xxx", "my-model")  # doctest: +SKIP
        >>> client = LLMClient(cfg)  # doctest: +SKIP
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send a chat request and return the parsed response.

    Args:
        messages: A list of message dicts as expected by the provider API.
            Each dict typically carries keys like ``role`` and ``content``.
        tools: Optional list of tool/function-schema dicts (as per the OpenAI
            format). Only included in the request body when non-empty.

    Returns:
        A ``ChatResponse`` with ``text`` and ``tool_calls`` populated from
        ``choices[0].message``.

    Raises:
        OverCapError: When the provider returns HTTP 400 with a body mentioning
            *context_length_exceeded* or *maximum context length*.
        RuntimeError: For all other non-2xx responses, including the HTTP status
            and response body text.
    """
        body: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens

        if tools:
            body["tools"] = tools

        payload = json.dumps(body).encode("utf-8")
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            detail = f"HTTP {exc.code}: {body_text}"
            if exc.code == 400 and (
                "context_length_exceeded" in body_text.lower()
                or "maximum context length" in body_text.lower()
            ):
                raise OverCapError(f"context too long: {detail}") from exc
            raise RuntimeError(detail) from exc

        choices = data.get("choices", [])
        if not choices:
            return ChatResponse()

        message = choices[0].get("message", {})
        text = message.get("content", "") or ""
        tool_calls_raw = message.get("tool_calls") or []
        tool_calls = [_tool_call_from_dict(tc) for tc in tool_calls_raw]

        return ChatResponse(text=text, tool_calls=tool_calls)
