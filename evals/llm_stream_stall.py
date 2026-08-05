"""Recovery from an SSE stream that stalls or dies mid-drain, end to end.

``LLMClient._request_with_retry`` retries the connect-and-status phase, but its
loop ends at the 2xx: a streaming response that goes silent AFTER headers (the
backend accepts the request, sends ``text/event-stream``, then never sends the
first chunk — or stops between chunks) used to raise a raw ``requests``
exception out of ``_read_sse_response`` after the per-read-gap timeout, which
nothing caught, killing the whole run. Measured live: a benchmark pilot lost
5/39 runs to exactly this shape.

The fix has two layers, split by what the sink has seen:

  * chat()-internal — a stall BEFORE the first text delta is invisible to the
    caller, so chat() re-issues the request once itself, announcing the retry
    on stderr (an unannounced ~10-minute silent retry reads as a hang to
    anything supervising the process by output liveness).
  * turn-level — once text has been forwarded to ``on_delta``, replaying inside
    chat() would deliver part of the answer twice, so the failure surfaces as a
    typed ``StreamStalledError`` and ``turn/llm_call.py`` (the one caller whose
    delivery contract tolerates re-delivery after an announced discard)
    re-issues the call, bounded by ``_MAX_STREAM_STALL_RETRIES``.

Checks a–c run against a REAL local HTTP server over real sockets with the
per-read-gap timeout patched small — no stubbed transport — so they prove the
actual exception types requests raises on a stalled/severed stream are the ones
chat() handles:

a. Headers-then-silence on the first request, a full SSE answer on the second:
   chat() returns the complete response, exactly two requests reach the server,
   each fragment hits on_delta exactly once, and the retry is announced on
   stderr.
b. A stream that dies after forwarding text raises ``StreamStalledError``
   mentioning the forwarded count, after exactly ONE request — chat() must not
   replay past-forwarded deltas itself.
c. Headers-then-silence on every request: ``StreamStalledError`` ("before the
   first token") after exactly two requests — the internal retry is bounded.

Checks d–e drive the REAL turn loop (``agent.handle_user_message``) with a
scripted stub client:

d. A chat call that raises ``StreamStalledError`` once and answers on re-issue:
   the turn completes with the scripted answer and the re-issue telemetry is on
   stderr.
e. A chat call that stalls on every attempt: the error propagates (loudly)
   after exactly ``1 + _MAX_STREAM_STALL_RETRIES`` chat calls — the retry
   budget is a bound, not a loop.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file); binds one ephemeral localhost port; touches no repo files.
"""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from evals._stub import _StubClient, _final_answer_response, disable_memory_hooks

# Per-read-gap timeout patched onto llm for the socket checks, and how long a
# "stalling" handler holds its silence. The stall must comfortably exceed the
# timeout; handler threads are daemonic so shutdown never waits out a sleeper.
_PATCHED_READ_TIMEOUT_S = 0.75
_STALL_HOLD_S = 3.0

_FINAL_TEXT = "Recovered analysis complete."


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


class _StallServer(http.server.ThreadingHTTPServer):
    """Local chat-completions endpoint scripted per-request.

    ``script`` is a list of behaviors, one per incoming POST (the last entry
    repeats): ``"stall"`` sends SSE headers then goes silent, ``"partial"``
    sends headers plus one text delta then goes silent, ``"ok"`` serves a
    complete two-delta stream with a usage chunk and ``[DONE]``.
    """

    daemon_threads = True

    def __init__(self, script: list[str]) -> None:
        super().__init__(("127.0.0.1", 0), _StallHandler)
        self.script = script
        self.request_count = 0
        self._lock = threading.Lock()

    def next_behavior(self) -> str:
        with self._lock:
            idx = self.request_count
            self.request_count += 1
        return self.script[min(idx, len(self.script) - 1)]


class _StallHandler(http.server.BaseHTTPRequestHandler):
    # Real SSE backends stream over HTTP/1.1 with chunked transfer encoding —
    # that framing is what makes each flushed delta reach the client
    # immediately. A close-delimited HTTP/1.0 body does NOT model streaming:
    # the client's buffered reader blocks until its read size fills or EOF, so
    # a flushed partial delta never surfaces (observed while building this
    # eval: the "partial" case looked pre-first-token to the client).
    protocol_version = "HTTP/1.1"

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler dispatch name
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        behavior = self.server.next_behavior()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        if behavior in ("stall", "partial"):
            if behavior == "partial":
                self._write_chunk(_sse({"choices": [{"delta": {"content": "Hel"}}]}))
            time.sleep(_STALL_HOLD_S)
            return
        self._write_chunk(_sse({"choices": [{"delta": {"content": "Hello "}}]}))
        self._write_chunk(_sse({"choices": [{"delta": {"content": "world"}}]}))
        self._write_chunk(_sse({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}))
        self._write_chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 — base signature
        pass


@contextlib.contextmanager
def _serving(script: list[str]):
    """Run a _StallServer with llm's read timeout patched small; always restore."""
    import llm

    server = _StallServer(script)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    saved = llm._READ_TIMEOUT_SECONDS
    llm._READ_TIMEOUT_SECONDS = _PATCHED_READ_TIMEOUT_S
    try:
        yield server
    finally:
        llm._READ_TIMEOUT_SECONDS = saved
        server.shutdown()
        server.server_close()


def _client(server):
    import llm
    from config import LLMConfig

    return llm.LLMClient(
        LLMConfig(
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            api_key="k",
            model="m",
            stream=True,
        )
    )


def check_prefirst_stall_recovers() -> list[str]:
    """a. Stall before the first token, then a good stream: one internal retry wins."""
    failures: list[str] = []
    seen: list[str] = []
    stderr = io.StringIO()

    with _serving(["stall", "ok"]) as server:
        with contextlib.redirect_stderr(stderr):
            resp = _client(server).chat([{"role": "user", "content": "hi"}], on_delta=seen.append)

        if resp.text != "Hello world":
            failures.append(f"recovered response text wrong: {resp.text!r}")
        if seen != ["Hello ", "world"]:
            failures.append(f"on_delta did not receive each fragment exactly once: {seen!r}")
        if resp.prompt_tokens != 11:
            failures.append(f"usage not captured on the retried stream: {resp.prompt_tokens!r}")
        if server.request_count != 2:
            failures.append(
                f"server saw {server.request_count} request(s), expected exactly 2 "
                f"(one stalled + one retry)"
            )
    if "retrying" not in stderr.getvalue():
        failures.append(
            f"internal retry was not announced on stderr (watchdogs key liveness "
            f"off output): {stderr.getvalue()!r}"
        )
    return failures


def check_midstream_stall_raises_typed() -> list[str]:
    """b. A stream that dies past the first delta raises StreamStalledError, no replay."""
    import llm

    failures: list[str] = []
    seen: list[str] = []

    with _serving(["partial"]) as server:
        try:
            _client(server).chat([{"role": "user", "content": "hi"}], on_delta=seen.append)
        except llm.StreamStalledError as exc:
            if "already forwarded" not in str(exc):
                failures.append(f"message does not report forwarded text: {exc}")
        except Exception as exc:
            failures.append(
                f"raised {type(exc).__name__}, expected StreamStalledError: {exc}"
            )
            return failures
        else:
            failures.append("mid-stream death did not raise — the stall went unnoticed")
            return failures

        if seen != ["Hel"]:
            failures.append(f"forwarded fragments before the death are wrong: {seen!r}")
        if server.request_count != 1:
            failures.append(
                f"server saw {server.request_count} request(s), expected exactly 1 — "
                f"chat() must never replay past-forwarded deltas itself"
            )
    return failures


def check_prefirst_stall_bounded() -> list[str]:
    """c. Stalling on every attempt raises StreamStalledError after exactly 2 requests."""
    import llm

    failures: list[str] = []
    stderr = io.StringIO()

    with _serving(["stall"]) as server:
        try:
            with contextlib.redirect_stderr(stderr):
                _client(server).chat(
                    [{"role": "user", "content": "hi"}], on_delta=lambda _piece: None
                )
        except llm.StreamStalledError as exc:
            if "before the first token" not in str(exc):
                failures.append(f"message does not name the pre-first-token shape: {exc}")
        except Exception as exc:
            failures.append(
                f"raised {type(exc).__name__}, expected StreamStalledError: {exc}"
            )
            return failures
        else:
            failures.append("perpetual stalling never raised — the retry is unbounded?")
            return failures

        if server.request_count != 2:
            failures.append(
                f"server saw {server.request_count} request(s), expected exactly 2 — "
                f"the internal retry must be bounded to one fresh attempt"
            )
    return failures


def _stalling_factory():
    """Script entry that raises StreamStalledError, as a dead stream would."""
    import llm

    def factory():
        raise llm.StreamStalledError(
            "stream died after 600.0s with 42 assistant-text character(s) "
            "already forwarded: simulated"
        )

    return factory


def _run_turn(script) -> tuple[str | None, BaseException | None, str, "_StubClient"]:
    """Drive one real handle_user_message turn against *script*; restore all state."""
    disable_memory_hooks()
    import agent
    import tools.registry as registry
    from session import Session

    saved_mode = registry.current_mode()
    registry.activate_mode("research")
    original_cwd = os.getcwd()
    tmp = Path(tempfile.mkdtemp(prefix="llm-stream-stall-"))
    os.chdir(tmp)
    stderr = io.StringIO()
    client = _StubClient(script, max_calls=10)
    answer: str | None = None
    raised: BaseException | None = None
    try:
        session = Session(str(tmp), "test-model", "You are a test agent.")
        with contextlib.redirect_stderr(stderr):
            try:
                answer = agent.handle_user_message("Summarize the fixtures.", session, client)  # pyright: ignore[reportArgumentType]
            except BaseException as exc:
                raised = exc
    finally:
        os.chdir(original_cwd)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)
    return answer, raised, stderr.getvalue(), client


def check_turn_reissues_after_stall() -> list[str]:
    """d. The turn loop re-issues a stalled call and completes with the real answer."""
    failures: list[str] = []
    answer, raised, err, client = _run_turn(
        [_stalling_factory(), _final_answer_response(_FINAL_TEXT)]
    )

    if raised is not None:
        failures.append(f"turn raised {type(raised).__name__} instead of recovering: {raised}")
        return failures
    if answer != _FINAL_TEXT:
        failures.append(f"turn answer is {answer!r}, expected the scripted final answer")
    if client.calls != 2:
        failures.append(f"stub saw {client.calls} chat call(s), expected 2 (stall + re-issue)")
    if "stream stalled" not in err or "re-issuing" not in err:
        failures.append(f"re-issue telemetry missing from stderr: {err!r}")
    return failures


def check_turn_retry_budget_bounds() -> list[str]:
    """e. Perpetual stalls propagate after exactly 1 + _MAX_STREAM_STALL_RETRIES calls."""
    import llm
    from turn.llm_call import _MAX_STREAM_STALL_RETRIES

    failures: list[str] = []
    answer, raised, _err, client = _run_turn([_stalling_factory()])

    expected_calls = 1 + _MAX_STREAM_STALL_RETRIES
    if raised is None:
        failures.append(
            f"a perpetually stalling backend produced answer {answer!r} — the "
            f"exhausted budget must fail loudly, not fabricate a turn"
        )
        return failures
    if not isinstance(raised, llm.StreamStalledError):
        failures.append(
            f"turn raised {type(raised).__name__}, expected StreamStalledError: {raised}"
        )
    if client.calls != expected_calls:
        failures.append(
            f"stub saw {client.calls} chat call(s), expected exactly {expected_calls} "
            f"(the retry budget is a bound, not a loop)"
        )
    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("prefirst-stall-recovers", check_prefirst_stall_recovers),
        ("midstream-stall-raises-typed", check_midstream_stall_raises_typed),
        ("prefirst-stall-bounded", check_prefirst_stall_bounded),
        ("turn-reissues-after-stall", check_turn_reissues_after_stall),
        ("turn-retry-budget-bounds", check_turn_retry_budget_bounds),
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
        "PASS: a pre-first-token stall retries once (announced, bounded, exactly-once "
        "on_delta preserved), a mid-stream death surfaces as StreamStalledError without "
        "a client-side replay, and the turn loop re-issues the call within its budget"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
