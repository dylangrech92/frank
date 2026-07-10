"""Whole-call wall-clock ceiling on the streaming LLM path, driven directly.

``llm._read_sse_response`` consumes an SSE ``text/event-stream`` chat response.
Its ``requests`` timeouts bound the connect phase and the per-read *gap*, but a
stream that keeps trickling chunks resets the gap timer on every line — so a
pathologically slow-but-alive generation had no whole-call bound and could run
for the model's entire context window. This adds a wall-clock ceiling, anchored
when the attempt's request is issued and checked once per SSE line, that raises
a distinct ``ResponseCeilingError`` on breach.

This script drives the real production code — ``llm._read_sse_response`` directly
and ``LLMClient.chat`` end-to-end with a stubbed request seam (no network) — and
asserts:

a. A source that trickles SSE lines forever under a tiny ceiling raises
   ``ResponseCeilingError`` (the exact, distinct type) and does so within
   ceiling + 1s (the check is per-line, not a hang).
b. The exception message reports how much work is thrown away — elapsed seconds
   plus the accumulated assistant-text / tool-call-fragment counts.
c. A normal stream (content chunks + tool-call fragments split across chunks + a
   final usage chunk with empty choices + ``[DONE]``) parses unchanged: text
   joined, tool-call ``arguments`` merged across chunks and JSON-parsed, and
   ``prompt_tokens``/``completion_tokens`` captured. This is the regression
   guard for the pre-existing streaming behavior.
d. A breach does NOT retry: driving ``chat`` end-to-end with the request seam
   counted, the breach raises after exactly one request issue — the ceiling
   fires while draining an already-returned 2xx stream, past every retry point.

The ceiling is injected for determinism by monkeypatching the module constant
``llm._WALL_CLOCK_CEILING_SECONDS`` (read once at ``_read_sse_response`` entry),
always restored in a finally.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file); touches no repo files and makes no network calls.
"""

from __future__ import annotations

import json
import re
import sys
import time


def _trickle_source(sleep_s: float = 0.05):
    """Yield content-bearing SSE data lines forever, sleeping between each.

    Simulates a stream that stays alive (so the per-read-gap timeout never
    trips) but never terminates — the exact shape the wall-clock ceiling exists
    to bound.
    """
    while True:
        yield 'data: {"choices": [{"delta": {"content": "x"}}]}'
        time.sleep(sleep_s)


def _sse(obj) -> str:
    """Render one SSE ``data:`` line carrying *obj* as JSON."""
    return "data: " + json.dumps(obj)


def check_breach_raises_bounded() -> list[str]:
    """a. Trickle + tiny ceiling raises ResponseCeilingError within ceiling + 1s."""
    import llm

    failures: list[str] = []
    ceiling = 0.3
    saved = llm._WALL_CLOCK_CEILING_SECONDS
    llm._WALL_CLOCK_CEILING_SECONDS = ceiling
    try:
        started = time.monotonic()
        wall_start = time.monotonic()
        try:
            llm._read_sse_response(_trickle_source(), lambda _piece: None, started)
        except llm.ResponseCeilingError:
            pass
        except Exception as exc:  # any other type is a failure
            failures.append(
                f"trickle raised {type(exc).__name__}, expected ResponseCeilingError: {exc}"
            )
            return failures
        else:
            failures.append("trickle source did not raise — the ceiling never fired")
            return failures

        wall_elapsed = time.monotonic() - wall_start
        if wall_elapsed > ceiling + 1.0:
            failures.append(
                f"breach took {wall_elapsed:.2f}s, expected <= {ceiling + 1.0:.2f}s "
                f"(the ceiling must be checked per-line, not on a hang)"
            )
    finally:
        llm._WALL_CLOCK_CEILING_SECONDS = saved

    return failures


def check_breach_message_reports_discarded_work() -> list[str]:
    """b. The exception message reports elapsed seconds + accumulated-content info."""
    import llm

    failures: list[str] = []
    saved = llm._WALL_CLOCK_CEILING_SECONDS
    llm._WALL_CLOCK_CEILING_SECONDS = 0.3
    try:
        started = time.monotonic()
        try:
            llm._read_sse_response(_trickle_source(), lambda _piece: None, started)
        except llm.ResponseCeilingError as exc:
            msg = str(exc)
        else:
            failures.append("trickle source did not raise — cannot inspect the message")
            return failures

        if re.search(r"after \d+(\.\d+)?s", msg) is None:
            failures.append(f"message does not report elapsed seconds: {msg!r}")
        if "assistant-text" not in msg:
            failures.append(f"message does not report accumulated assistant-text: {msg!r}")
        if "tool-call fragment" not in msg:
            failures.append(f"message does not report tool-call fragment count: {msg!r}")
    finally:
        llm._WALL_CLOCK_CEILING_SECONDS = saved

    return failures


def check_normal_stream_parses_unchanged() -> list[str]:
    """c. A normal stream (usage chunk + [DONE]) parses exactly as before."""
    import llm

    failures: list[str] = []

    lines = [
        _sse({"choices": [{"delta": {"content": "Hello "}}]}),
        _sse({"choices": [{"delta": {"content": "world"}}]}),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {"name": "do_it", "arguments": '{"a"'},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}}
                ]
            }
        ),
        _sse({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}),
        "data: [DONE]",
    ]

    seen: list[str] = []
    # A far-future ceiling: this stream must complete without tripping it.
    started = time.monotonic()
    resp = llm._read_sse_response(iter(lines), seen.append, started)

    if resp.text != "Hello world":
        failures.append(f"text not joined correctly: {resp.text!r}")
    if seen != ["Hello ", "world"]:
        failures.append(f"on_delta did not receive each fragment once: {seen!r}")
    if len(resp.tool_calls) != 1:
        failures.append(f"expected exactly 1 tool call, got {len(resp.tool_calls)}")
    else:
        tc = resp.tool_calls[0]
        if tc.name != "do_it":
            failures.append(f"tool-call name wrong: {tc.name!r}")
        if tc.arguments != {"a": 1}:
            failures.append(
                f"tool-call arguments not merged/parsed across chunks: {tc.arguments!r}"
            )
    if resp.prompt_tokens != 11:
        failures.append(f"prompt_tokens not captured from usage chunk: {resp.prompt_tokens!r}")
    if resp.completion_tokens != 7:
        failures.append(
            f"completion_tokens not captured from usage chunk: {resp.completion_tokens!r}"
        )

    return failures


class _FakeStreamResponse:
    """Minimal stand-in for a streaming ``requests.Response`` (2xx SSE)."""

    def __init__(self, line_source) -> None:
        self.status_code = 200
        self.headers = {"Content-Type": "text/event-stream"}
        self._line_source = line_source

    def iter_lines(self):
        return self._line_source


def check_breach_does_not_retry() -> list[str]:
    """d. End-to-end via chat(): a breach raises after exactly one request issue."""
    import llm
    from config import LLMConfig

    failures: list[str] = []

    client = llm.LLMClient(LLMConfig(base_url="http://stub", api_key="k", model="m", stream=True))

    issues = {"n": 0}

    def _stub_request_with_retry(url, body, headers, stream):
        # Mirror the real return contract: (response, request_started_monotonic).
        # A 2xx streaming response is handed back BEFORE its body is consumed, so
        # the later ceiling breach happens past this seam and cannot re-enter it.
        issues["n"] += 1
        return _FakeStreamResponse(_trickle_source()), time.monotonic()

    client._request_with_retry = _stub_request_with_retry  # type: ignore[method-assign]

    saved = llm._WALL_CLOCK_CEILING_SECONDS
    llm._WALL_CLOCK_CEILING_SECONDS = 0.3
    try:
        try:
            client.chat([{"role": "user", "content": "hi"}], on_delta=lambda _piece: None)
        except llm.ResponseCeilingError:
            pass
        except Exception as exc:
            failures.append(
                f"chat raised {type(exc).__name__}, expected ResponseCeilingError: {exc}"
            )
            return failures
        else:
            failures.append("chat did not raise — the ceiling never fired end-to-end")
            return failures

        if issues["n"] != 1:
            failures.append(
                f"request was issued {issues['n']} times — a ceiling breach must NOT "
                f"retry (it is raised past the 2xx that already started streaming)"
            )
    finally:
        llm._WALL_CLOCK_CEILING_SECONDS = saved

    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("breach-raises-bounded", check_breach_raises_bounded),
        ("breach-message-reports-work", check_breach_message_reports_discarded_work),
        ("normal-stream-parses-unchanged", check_normal_stream_parses_unchanged),
        ("breach-does-not-retry", check_breach_does_not_retry),
    ):
        try:
            failures = fn()
        except Exception as exc:  # a raised exception is itself a failure
            failures = [f"{label} raised {type(exc).__name__}: {exc}"]
        for f in failures:
            print(f"FAIL [{label}]: {f}", file=sys.stderr)
        all_failures.extend(failures)

    if all_failures:
        return 1
    print(
        "PASS: a trickling stream trips the wall-clock ceiling with a distinct "
        "ResponseCeilingError (bounded, message reports discarded work); a normal "
        "stream parses unchanged; and a breach never retries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
