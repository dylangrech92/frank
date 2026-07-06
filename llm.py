"""OpenAI-compatible chat-completions client using only the Python standard library.

Provides ``LLMClient`` which POSTs to any OpenAI-compatible endpoint (e.g. vLLM,
Ollama with the openai suffix, LMStudio) and returns parsed ``ChatResponse`` objects.
"""

from __future__ import annotations

import json
import re
import time
import requests
from config import LLMConfig
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

# One retry, ~1s backoff, on 5xx and connection-level errors (F5). Never
# retried: 4xx responses, and anything past the point a 2xx response has
# started streaming deltas to on_delta (see LLMClient._request_with_retry).
_MAX_ATTEMPTS = 2
_RETRY_BACKOFF_SECONDS = 1.0


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
        prompt_tokens: ``usage.prompt_tokens`` from the response when the
            provider reports it; ``None`` when absent (estimate-only fallback).
        completion_tokens: ``usage.completion_tokens`` from the response when
            the provider reports it; ``None`` when absent.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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


def _read_sse_response(resp: Any, on_delta: Callable[[str], None]) -> ChatResponse:
    """Consume a ``text/event-stream`` chat-completions response into a ChatResponse.

    Iterates ``data:`` lines until ``[DONE]``, forwarding every assistant-text
    fragment to *on_delta* as it arrives and accumulating tool-call fragments
    (OpenAI streaming format: index-keyed deltas whose ``arguments`` strings
    concatenate). Callback exceptions are swallowed — display must never kill
    the request. Unparseable data lines are skipped.

    A ``usage`` key is checked on every parsed chunk (not just ones carrying
    choices) since providers commonly send the final usage-bearing chunk with
    an empty ``choices`` list — checking it before the empty-choices bail-out
    is what actually captures it (F6).

    Args:
        resp: An iterable yielding raw bytes (or str) lines, one SSE line each
            (the open ``urlopen`` response object, or ``Response.iter_lines()``
            from ``requests``).
        on_delta: Called with each non-empty ``delta.content`` fragment.

    Returns:
        A ``ChatResponse`` identical in shape to the non-streaming parse, with
        ``prompt_tokens``/``completion_tokens`` populated when the stream
        carried a ``usage`` chunk.
    """
    text_parts: list[str] = []
    calls_by_index: dict[int, dict[str, Any]] = {}
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    for raw_line in resp:
        raw_text = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = raw_text.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        usage = chunk.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}

        piece = delta.get("content")
        if piece:
            text_parts.append(piece)
            try:
                on_delta(piece)
            except Exception:
                pass

        for frag in delta.get("tool_calls") or []:
            idx = frag.get("index", 0)
            slot = calls_by_index.setdefault(
                idx, {"id": None, "function": {"name": "", "arguments": ""}}
            )
            if frag.get("id"):
                slot["id"] = frag["id"]
            func = frag.get("function") or {}
            if func.get("name"):
                slot["function"]["name"] = func["name"]
            if func.get("arguments"):
                slot["function"]["arguments"] += func["arguments"]

    tool_calls: List[ToolCall] = []
    for idx in sorted(calls_by_index):
        slot = calls_by_index[idx]
        if slot["id"] is None:
            slot["id"] = f"call_{idx}"  # some providers omit ids on streamed calls
        tool_calls.append(_tool_call_from_dict(slot))

    return ChatResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class LLMClient:
    """Thin HTTP client for OpenAI-compatible chat-completions endpoints.

    Uses a ``requests.Session`` (rather than a bare ``urllib.request.urlopen``
    per call) so the underlying TCP/TLS connection is kept alive and reused
    across the several chat calls a single turn makes (F1). ``requests`` is
    already a project dependency (see ``tools/web_read.py``), so this avoids
    hand-rolling keep-alive management on top of ``http.client``.

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
        self._session = requests.Session()

    def _request_with_retry(
        self,
        url: str,
        body: Dict[str, Any],
        headers: Dict[str, str],
        stream: bool,
    ) -> requests.Response:
        """POST *body* to *url*, retrying once on 5xx / connection-level errors.

        Only the connect-and-status-check phase is ever retried: a 2xx
        response is returned immediately, before its body (or SSE stream) is
        consumed, so a retry here never re-sends a request whose deltas were
        already forwarded to ``on_delta`` (that consumption happens later, in
        ``chat()``/``_read_sse_response``, outside this method). 4xx
        responses are never retried — only 5xx and connection-level failures
        (``requests.exceptions.ConnectionError``/``Timeout``) are (F5).

        Args:
            url: Full chat-completions endpoint URL.
            body: JSON-serializable request body.
            headers: Request headers (Content-Type, Authorization).
            stream: Passed through to ``requests`` so a streaming response's
                body is not eagerly buffered.

        Returns:
            The ``requests.Response`` for a status code below 500 (the caller
            still checks for 4xx and raises the appropriate error contract).

        Raises:
            RuntimeError: On a 5xx or connection-level failure that persists
                through the retry.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            is_last_attempt = attempt == _MAX_ATTEMPTS - 1
            try:
                resp = self._session.post(url, json=body, headers=headers, stream=stream)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                if not is_last_attempt:
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                    continue
                raise RuntimeError(f"connection error: {exc}") from exc

            if resp.status_code >= 500:
                if not is_last_attempt:
                    resp.close()
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                    continue
                body_text = resp.text
                raise RuntimeError(f"HTTP {resp.status_code}: {body_text}")

            return resp

        # Unreachable: the loop above always either returns or raises.
        raise RuntimeError(f"request failed after retry: {last_exc}")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        """Send a chat request and return the parsed response.

    Args:
        messages: A list of message dicts as expected by the provider API.
            Each dict typically carries keys like ``role`` and ``content``.
        tools: Optional list of tool/function-schema dicts (as per the OpenAI
            format). Only included in the request body when non-empty.
        on_delta: Optional callback invoked with each assistant-text fragment as
            it streams in. Streaming (SSE) is requested only when this is set
            AND ``config.stream`` is true; the return value is identical either
            way. A provider that ignores ``stream`` and answers with plain JSON
            is handled transparently (``on_delta`` is then never called).

    Returns:
        A ``ChatResponse`` with ``text`` and ``tool_calls`` populated from
        ``choices[0].message``, plus ``prompt_tokens``/``completion_tokens``
        when the provider's response carried a ``usage`` field.

    Raises:
        OverCapError: When the provider returns HTTP 400 with a body mentioning
            *context_length_exceeded* or *maximum context length*.
        RuntimeError: For all other non-2xx responses, including the HTTP status
            and response body text.
    """
        want_stream = on_delta is not None and self.config.stream
        body: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        if want_stream:
            body["stream"] = True
            # Most OpenAI-compatible servers (this project's default Ollama
            # endpoint included) only emit the final usage-bearing chunk when
            # explicitly asked via stream_options — otherwise SSE responses
            # never carry usage at all, silently defeating F6 for the
            # streaming path (verified live: identical request without this
            # flag omits `usage` entirely; with it, a final chunk with empty
            # `choices` and populated `usage` is sent, as already handled by
            # ``_read_sse_response``).
            body["stream_options"] = {"include_usage": True}

        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens

        if tools:
            body["tools"] = tools

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        resp = self._request_with_retry(url, body, headers, want_stream)

        if resp.status_code >= 400:
            body_text = resp.text
            detail = f"HTTP {resp.status_code}: {body_text}"
            if resp.status_code == 400 and (
                "context_length_exceeded" in body_text.lower()
                or "maximum context length" in body_text.lower()
            ):
                raise OverCapError(f"context too long: {detail}")
            raise RuntimeError(detail)

        content_type = resp.headers.get("Content-Type", "")
        if on_delta is not None and want_stream and content_type.startswith("text/event-stream"):
            return _read_sse_response(resp.iter_lines(), on_delta)

        data = resp.json()

        choices = data.get("choices", [])
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if not choices:
            return ChatResponse(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

        message = choices[0].get("message", {})
        text = message.get("content", "") or ""
        tool_calls_raw = message.get("tool_calls") or []
        tool_calls = [_tool_call_from_dict(tc) for tc in tool_calls_raw]

        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
