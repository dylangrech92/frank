"""Reasoning-trace capture and resend across turns, end to end.

The serving side renders each historical assistant turn as ``<think>trace
</think>`` only when the request resends the trace on that message — under the
field name ``reasoning`` (``reasoning_content`` is silently dropped inbound).
So the client must capture the trace it streams each round and carry it on the
assistant history row, or the model loses its own prior reasoning every turn.

Checks a–b prove the capture layer against a REAL local chat-completions
server (HTTP/1.1 chunked SSE — close-delimited framing cannot model
streaming):

a. Streaming: ``delta.reasoning`` pieces accumulate into
   ``ChatResponse.reasoning`` in order, and ``on_delta`` receives ONLY content
   fragments — the trace must never leak into the user-visible sink.
b. Non-streaming: ``message.reasoning`` lands in ``ChatResponse.reasoning``;
   a reply without the field yields ``""``.

Check c proves the resend through the REAL turn loop
(``agent.handle_user_message`` with a real ``LLMClient`` against the same
recording server; three user turns, four requests):

c1. Mid-turn (in-flight): the tool_calls-bearing assistant row in the next
    request carries that round's exact trace under ``"reasoning"``.
c2. Across turns: the completed turn's final-answer row still carries its
    trace after history pruning strips ``tool_calls``.
c3. A reasoning-free round adds NO ``"reasoning"`` key to its assistant row —
    absence, not an empty string.

Check d proves the token estimator counts a resent trace — it occupies real
wire context, so an uncounted trace would make compaction trigger late.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before
exec'ing this file); binds one ephemeral localhost port; touches no repo files.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

from evals._stub import disable_memory_hooks

_TRACE_ROUND1 = "First I inspect the fixtures."
_TRACE_ROUND2 = "Now I conclude."
_ANSWER_TURN1 = "The fixtures are fine."
_ANSWER_TURN2 = "Second answer."


class _RecordingServer(http.server.ThreadingHTTPServer):
    """Local chat-completions endpoint scripted per-request, recording bodies.

    ``script`` is a list of behavior dicts, one per incoming POST (the last
    entry repeats): ``reasoning`` (list of reasoning delta strings, may be
    empty), ``content`` (list of content delta strings), ``tool_call``
    (optional ``(name, arguments_json)`` pair). ``requests`` holds the parsed
    JSON body of every POST, in order — the only evidence check c trusts.
    """

    daemon_threads = True

    def __init__(self, script: list[dict]) -> None:
        super().__init__(("127.0.0.1", 0), _RecordingHandler)
        self.script = script
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def next_behavior(self, body: dict) -> dict:
        with self._lock:
            idx = len(self.requests)
            self.requests.append(body)
        return self.script[min(idx, len(self.script) - 1)]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    # Real SSE backends stream over HTTP/1.1 with chunked transfer encoding —
    # a close-delimited HTTP/1.0 body does NOT model streaming: the client's
    # buffered reader blocks until its read size fills or EOF, so a flushed
    # delta never surfaces.
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler dispatch name
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.loads(raw.decode("utf-8"))
        server: _RecordingServer = self.server  # type: ignore[assignment]
        behavior = server.next_behavior(body)
        if body.get("stream"):
            self._respond_sse(behavior)
        else:
            self._respond_json(behavior)

    def _respond_sse(self, behavior: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        for piece in behavior.get("reasoning", []):
            self._write_chunk(_sse({"choices": [{"delta": {"reasoning": piece}}]}))
        for piece in behavior.get("content", []):
            self._write_chunk(_sse({"choices": [{"delta": {"content": piece}}]}))
        tool_call = behavior.get("tool_call")
        if tool_call is not None:
            name, arguments_json = tool_call
            self._write_chunk(
                _sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-rt-1",
                                            "function": {
                                                "name": name,
                                                "arguments": arguments_json,
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            )
        self._write_chunk(
            _sse({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}})
        )
        self._write_chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _respond_json(self, behavior: dict) -> None:
        message: dict = {"role": "assistant", "content": "".join(behavior.get("content", []))}
        trace = "".join(behavior.get("reasoning", []))
        if trace:
            message["reasoning"] = trace
        tool_call = behavior.get("tool_call")
        if tool_call is not None:
            name, arguments_json = tool_call
            message["tool_calls"] = [
                {
                    "id": "call-rt-1",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_json},
                }
            ]
        payload = json.dumps(
            {
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 — base signature
        pass


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


@contextlib.contextmanager
def _serving(script: list[dict]):
    server = _RecordingServer(script)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _client(server: _RecordingServer, stream: bool):
    import llm
    from config import LLMConfig

    return llm.LLMClient(
        LLMConfig(base_url=server.base_url, api_key="k", model="m", stream=stream)
    )


def check_streaming_capture() -> list[str]:
    """a. Reasoning deltas accumulate on the response; on_delta stays content-only."""
    failures: list[str] = []
    seen: list[str] = []
    script = [{"reasoning": ["First I ", "inspect the fixtures."], "content": ["Hello ", "world"]}]

    with _serving(script) as server:
        resp = _client(server, stream=True).chat(
            [{"role": "user", "content": "hi"}], on_delta=seen.append
        )

    if resp.reasoning != _TRACE_ROUND1:
        failures.append(f"streamed trace not captured: {resp.reasoning!r}")
    if resp.text != "Hello world":
        failures.append(f"content wrong alongside reasoning: {resp.text!r}")
    if seen != ["Hello ", "world"]:
        failures.append(
            f"on_delta must receive only content fragments, exactly once each: {seen!r}"
        )
    return failures


def check_nonstreaming_capture() -> list[str]:
    """b. message.reasoning lands on the response; absence yields the empty string."""
    failures: list[str] = []
    script = [
        {"reasoning": ["Deliberation."], "content": ["Answer one."]},
        {"content": ["Answer two."]},
    ]

    with _serving(script) as server:
        client = _client(server, stream=False)
        with_trace = client.chat([{"role": "user", "content": "hi"}])
        without_trace = client.chat([{"role": "user", "content": "hi again"}])

    if with_trace.reasoning != "Deliberation.":
        failures.append(f"non-streaming trace not captured: {with_trace.reasoning!r}")
    if without_trace.reasoning != "":
        failures.append(
            f"a reply without the field must yield '': {without_trace.reasoning!r}"
        )
    return failures


def _assistant_rows(body: dict) -> list[dict]:
    return [m for m in body.get("messages", []) if m.get("role") == "assistant"]


def check_turn_loop_roundtrip() -> list[str]:
    """c. The real turn loop resends captured traces on assistant history rows."""
    disable_memory_hooks()
    import agent
    import tools.registry as registry
    from session import Session

    script = [
        # Turn 1, round 1: reasoning + a tool call — the in-flight resend path.
        {
            "reasoning": ["First I ", "inspect the fixtures."],
            "content": ["Checking the tree."],
            "tool_call": ("list_files", '{"path": "."}'),
        },
        # Turn 1, round 2: reasoning + the final answer — the completed-turn path.
        {"reasoning": ["Now I conclude."], "content": [_ANSWER_TURN1]},
        # Turn 2: a reasoning-free final answer — must add no key at all.
        {"content": [_ANSWER_TURN2]},
        # Turn 3: exists only so turn 2's history row appears in a request.
        {"content": ["Goodbye."]},
    ]

    failures: list[str] = []
    saved_mode = registry.current_mode()
    registry.activate_mode("research")
    original_cwd = os.getcwd()
    tmp = Path(tempfile.mkdtemp(prefix="llm-reasoning-roundtrip-"))
    os.chdir(tmp)
    try:
        with _serving(script) as server:
            session = Session(str(tmp), "test-model", "You are a test agent.")
            client = _client(server, stream=True)
            for prompt in ("Look around.", "Anything else?", "Thanks."):
                agent.handle_user_message(prompt, session, client)
            requests = list(server.requests)
    finally:
        os.chdir(original_cwd)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    if len(requests) != 4:
        return [f"expected exactly 4 chat requests, server saw {len(requests)}"]

    # c1 — the in-flight tool_calls row (request 2) carries round 1's trace.
    calling_rows = [r for r in _assistant_rows(requests[1]) if r.get("tool_calls")]
    if len(calling_rows) != 1:
        failures.append(
            f"request 2 should hold exactly one tool_calls assistant row, "
            f"found {len(calling_rows)}"
        )
    elif calling_rows[0].get("reasoning") != _TRACE_ROUND1:
        failures.append(
            f"in-flight tool_calls row lost its trace: "
            f"{calling_rows[0].get('reasoning')!r}"
        )

    # c2 — turn 1's final answer (request 3) keeps its trace after pruning.
    final_rows = [
        r for r in _assistant_rows(requests[2]) if r.get("content") == _ANSWER_TURN1
    ]
    if len(final_rows) != 1:
        failures.append(
            f"request 3 should hold turn 1's final answer exactly once, "
            f"found {len(final_rows)}"
        )
    elif final_rows[0].get("reasoning") != _TRACE_ROUND2:
        failures.append(
            f"completed-turn answer lost its trace across turns: "
            f"{final_rows[0].get('reasoning')!r}"
        )

    # c3 — turn 2's reasoning-free answer (request 4) has no key at all.
    plain_rows = [
        r for r in _assistant_rows(requests[3]) if r.get("content") == _ANSWER_TURN2
    ]
    if len(plain_rows) != 1:
        failures.append(
            f"request 4 should hold turn 2's answer exactly once, "
            f"found {len(plain_rows)}"
        )
    elif "reasoning" in plain_rows[0]:
        failures.append(
            f"a reasoning-free round must add NO 'reasoning' key, "
            f"found {plain_rows[0]['reasoning']!r}"
        )
    return failures


def check_estimator_counts_trace() -> list[str]:
    """d. The token estimator counts a resent trace — it occupies real context."""
    from compaction import estimate_tokens

    row = {"role": "assistant", "content": "Done."}
    with_trace = estimate_tokens([{**row, "reasoning": "word " * 400}])
    without_trace = estimate_tokens([row])
    if with_trace <= without_trace:
        return [
            f"a 400-word trace added nothing to the estimate "
            f"({without_trace} -> {with_trace}) — compaction would trigger late"
        ]
    return []


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("streaming-capture", check_streaming_capture),
        ("nonstreaming-capture", check_nonstreaming_capture),
        ("turn-loop-roundtrip", check_turn_loop_roundtrip),
        ("estimator-counts-trace", check_estimator_counts_trace),
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
        "PASS: reasoning deltas accumulate on ChatResponse without leaking into "
        "on_delta, non-streaming replies populate the same field, the turn loop "
        "resends each captured trace on its assistant history row (in-flight and "
        "across turns) while reasoning-free rounds add no key, and the token "
        "estimator counts resent traces"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
