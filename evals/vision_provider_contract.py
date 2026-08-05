"""The optional ``vision`` provider: the image stops at one seam, as text.

A text-only main model cannot verify a web page it cannot see, and the backends
this project targets reject a prompt carrying an image outright when the served
model has no vision tower. The fix is a second, optional provider: when a
``vision`` block sits beside ``llm`` in the config, a screenshot captured in
verify mode is sent to THAT provider alone — one call, no tools, no history —
and only the returned TEXT is spliced into the transcript. The main model keeps
driving the loop and never receives pixels.

The whole guarantee is about which HTTP endpoint the base64 payload reaches, so
nothing here is asserted against a harness constant: two real local
chat-completions servers are stood up, each recording every request body it
receives, and every claim is read back off those recordings. The production
seam (``agent._attach_screenshot``), the real ``config.load``, the real
``llm.LLMClient``, the real ``Session`` and the real ``agent.handle_user_message``
turn loop all run unmodified — only the two endpoints are local.

Asserts:

a. Config parsing: a config with no ``vision`` key parses to ``vision=None``
   with its ``llm`` block untouched; a config with one parses both blocks
   independently; and the shared validator rejects a ``vision`` block missing a
   required key, naming ``vision.<key>`` (not ``llm.<key>``).

b. With a ``vision`` block: exactly ONE request reaches the vision stub, it
   carries exactly one image part whose data URI is the real PNG's bytes, and it
   asks for no tools. The transcript gets a text row naming the saved path and
   carrying the description — no image part anywhere — and the call is billed to
   the session's usage totals. Driving a real turn afterwards, the MAIN stub's
   recorded bodies carry zero image parts while carrying the description and the
   path.

c. With NO ``vision`` block: the vision stub receives nothing at all, the
   transcript holds a real image part, and a real turn puts that image on the
   MAIN provider's wire — the control that proves check b's image scan can see
   an image when one is there, rather than being vacuously satisfied.

d. Every vision failure is loud: an unreachable endpoint and an empty
   description (a reasoning model emitting only hidden reasoning) both produce a
   row stating the image is NOT attached, never a silent fallback to pixels and
   never a row that lets the model believe it saw something.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file); it also adds the repo root itself if not already present, so the
script can be invoked directly with
``PYTHONPATH=. .venv/bin/python evals/vision_provider_contract.py``. Binds two
ephemeral localhost ports, writes only inside throwaway temp directories, and
restores ``CODING_AGENT_CONFIG``, the active mode and the cwd in a ``finally``.
"""

from __future__ import annotations

import base64
import contextlib
import http.server
import io
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import zlib
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_VISION_TEXT = (
    "A checkout page. Heading reads 'Order summary'. A red banner states "
    "'Payment declined: card expired'. The Pay button is greyed out."
)
_MAIN_TEXT = "Answer from the main model."
_VISION_PROMPT_TOKENS = 1234
_VISION_COMPLETION_TOKENS = 56


def _png_bytes() -> bytes:
    """A genuinely valid 1x1 PNG — the seam reads real bytes and encodes them."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\xff\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


# =============================================================================
# Two recording chat-completions stubs
# =============================================================================


class _ProviderServer(http.server.ThreadingHTTPServer):
    """A local chat-completions endpoint that records every request body.

    ``reply_text`` is the assistant content each response carries and is
    reassigned between checks (an empty string models a provider that answers
    with no content at all). ``requests`` holds the parsed JSON body of every
    POST received, in order — the only evidence this eval trusts.
    """

    daemon_threads = True

    def __init__(self, reply_text: str) -> None:
        super().__init__(("127.0.0.1", 0), _ProviderHandler)
        self.reply_text = reply_text
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def record(self, body: dict) -> None:
        with self._lock:
            self.requests.append(body)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class _ProviderHandler(http.server.BaseHTTPRequestHandler):
    # HTTP/1.1 throughout, and the streaming branch frames its body in chunks.
    # A close-delimited HTTP/1.0 body cannot model SSE at all: the client's
    # buffered reader blocks until its read size fills or EOF, so a flushed
    # delta never surfaces. The vision describe call is non-streaming and the
    # turn below is driven with on_delta=None, so the JSON branch is what runs
    # today — but a stub that silently mis-frames the moment a caller starts
    # streaming is a trap, so both branches are correct here.
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler dispatch name
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = {"_unparseable": raw.decode("utf-8", "replace")}
        server: _ProviderServer = self.server  # type: ignore[assignment]
        server.record(body)

        if body.get("stream"):
            self._respond_sse(server.reply_text)
        else:
            self._respond_json(server.reply_text)

    def _respond_json(self, text: str) -> None:
        payload = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {
                    "prompt_tokens": _VISION_PROMPT_TOKENS,
                    "completion_tokens": _VISION_COMPLETION_TOKENS,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _respond_sse(self, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for obj in (
            {"choices": [{"delta": {"content": text}}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": _VISION_PROMPT_TOKENS,
                    "completion_tokens": _VISION_COMPLETION_TOKENS,
                },
            },
        ):
            self._write_chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self._write_chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 — base signature
        pass


@contextlib.contextmanager
def _serving(reply_text: str):
    server = _ProviderServer(reply_text)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _closed_port() -> int:
    """Bind and immediately release a port, so connecting to it is refused."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# =============================================================================
# Helpers over the recorded bodies
# =============================================================================


def _image_parts(body: dict) -> list[dict]:
    """Every ``image_url`` content part in one recorded request body."""
    out: list[dict] = []
    for message in body.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        out.extend(
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
    return out


def _body_text(body: dict) -> str:
    """All prose in one recorded body, flattened for substring assertions."""
    chunks: list[str] = []
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return "\n".join(chunks)


def _write_config(path: Path, main_url: str, vision_url: str | None) -> None:
    cfg: dict = {
        "llm": {
            "base_url": main_url,
            "api_key": "main-key",
            "model": "main-model",
            "context_limit": 200000,
            "stream": False,
        }
    }
    if vision_url is not None:
        cfg["vision"] = {
            "base_url": vision_url,
            "api_key": "vision-key",
            "model": "vision-model",
            "context_limit": 16000,
        }
    path.write_text(json.dumps(cfg), encoding="utf-8")


@contextlib.contextmanager
def _project(main_url: str, vision_url: str | None):
    """A temp project dir with a config on disk, exported the way main.py does.

    ``CODING_AGENT_CONFIG`` is the env var main.py sets at startup and the seam
    reads back, so pointing it at this config is what makes the run use these
    stub endpoints — and is also why the repo's own config.json can never be
    reached from here, whatever it happens to contain.
    """
    import tools.registry as registry
    from evals._stub import disable_memory_hooks

    disable_memory_hooks()
    saved_mode = registry.current_mode()
    saved_env = os.environ.get("CODING_AGENT_CONFIG")
    original_cwd = os.getcwd()
    tmp = Path(tempfile.mkdtemp(prefix="vision-provider-"))
    cfg_path = tmp / "config.json"
    _write_config(cfg_path, main_url, vision_url)
    os.environ["CODING_AGENT_CONFIG"] = str(cfg_path)
    registry.activate_mode("verify")
    os.chdir(tmp)
    try:
        yield tmp, cfg_path
    finally:
        os.chdir(original_cwd)
        if saved_env is None:
            os.environ.pop("CODING_AGENT_CONFIG", None)
        else:
            os.environ["CODING_AGENT_CONFIG"] = saved_env
        if saved_mode is not None:
            registry.activate_mode(saved_mode)


def _seed_verify_turn(session, tmp: Path) -> str:
    """Lay down a real verify turn's scaffolding and return the PNG's path."""
    from llm import ToolCall

    png = str(tmp / "shot.png")
    Path(png).write_bytes(_png_bytes())
    session.append_user("verify the checkout page")
    session.append_assistant(
        "taking a screenshot",
        tool_calls=[ToolCall(id="tc-1", name="screenshot", arguments={})],
    )
    session.append_tool_result("tc-1", "screenshot", f"screenshot saved to {png}")
    return png


def _run_main_turn(session, cfg_path: Path) -> str | None:
    """Drive one real turn against the MAIN provider built from the real config."""
    import agent
    from config import load as config_load
    from llm import LLMClient

    client = LLMClient(config_load(cfg_path).llm)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        return agent.handle_user_message(
            "Report what the screenshot showed.", session, client
        )


# =============================================================================
# Checks
# =============================================================================


def check_config_block_parses() -> list[str]:
    """a. vision is optional, independent of llm, and shares llm's validator."""
    from config import load as config_load

    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="vision-config-"))

    plain = tmp / "no-vision.json"
    _write_config(plain, "http://main-host:1/v1", None)
    cfg = config_load(plain)
    if cfg.vision is not None:
        failures.append(
            f"a config with no 'vision' key parsed to vision={cfg.vision!r}; "
            f"expected None — the block must be opt-in"
        )
    if cfg.llm.base_url != "http://main-host:1/v1" or cfg.llm.model != "main-model":
        failures.append(f"the llm block changed shape: {cfg.llm!r}")

    both = tmp / "with-vision.json"
    _write_config(both, "http://main-host:1/v1", "http://vision-host:2/v1")
    cfg = config_load(both)
    if cfg.vision is None:
        failures.append("a config WITH a 'vision' block parsed to vision=None")
    else:
        if cfg.vision.base_url != "http://vision-host:2/v1":
            failures.append(f"vision.base_url == {cfg.vision.base_url!r}")
        if cfg.vision.model != "vision-model":
            failures.append(f"vision.model == {cfg.vision.model!r}")
        if cfg.vision.api_key != "vision-key":
            failures.append(f"vision.api_key == {cfg.vision.api_key!r}")
        if cfg.vision.context_limit != 16000:
            failures.append(f"vision.context_limit == {cfg.vision.context_limit!r}")
    if cfg.llm.base_url != "http://main-host:1/v1" or cfg.llm.model != "main-model":
        failures.append(
            f"the llm block was altered by the presence of a vision block: {cfg.llm!r}"
        )

    broken = tmp / "broken-vision.json"
    broken.write_text(
        json.dumps(
            {
                "llm": {"base_url": "http://main-host:1/v1", "api_key": "k", "model": "m"},
                "vision": {"base_url": "http://vision-host:2/v1", "api_key": "k"},
            }
        ),
        encoding="utf-8",
    )
    try:
        config_load(broken)
    except ValueError as exc:
        if "vision.model" not in str(exc):
            failures.append(
                f"a vision block missing 'model' raised {exc!r}; the message must "
                f"name 'vision.model', not the llm block"
            )
    else:
        failures.append(
            "a vision block missing the required 'model' key parsed without error "
            "— a typo'd block must not degrade into 'no vision configured'"
        )

    return failures


def check_vision_takes_image_main_gets_text() -> list[str]:
    """b. The image reaches the vision stub only; the main stub sees text alone."""
    import agent
    import session_context
    from session import Session

    failures: list[str] = []
    png_b64 = base64.b64encode(_png_bytes()).decode("ascii")

    with _serving(_MAIN_TEXT) as main_stub, _serving(_VISION_TEXT) as vision_stub:
        with _project(main_stub.base_url, vision_stub.base_url) as (tmp, cfg_path):
            session = Session(str(tmp), "main-model", "You are a verification agent.")
            try:
                png = _seed_verify_turn(session, tmp)
                with contextlib.redirect_stderr(io.StringIO()):
                    agent._attach_screenshot(session, png, "tc-1")

                if len(vision_stub.requests) != 1:
                    failures.append(
                        f"the vision endpoint received {len(vision_stub.requests)} "
                        f"request(s), expected exactly 1 — one screenshot must cost "
                        f"one single-shot describe call"
                    )
                    return failures
                if main_stub.requests:
                    failures.append(
                        f"the main endpoint received {len(main_stub.requests)} "
                        f"request(s) while only a screenshot was attached"
                    )

                describe = vision_stub.requests[0]
                parts = _image_parts(describe)
                if len(parts) != 1:
                    failures.append(
                        f"the describe request carries {len(parts)} image part(s), "
                        f"expected exactly 1"
                    )
                else:
                    url = (parts[0].get("image_url") or {}).get("url", "")
                    if url != "data:image/png;base64," + png_b64:
                        failures.append(
                            "the describe request's data URI is not the captured "
                            f"PNG's bytes: {url[:64]!r}…"
                        )
                if describe.get("tools"):
                    failures.append(
                        f"the describe call offered tools: {describe['tools']!r} — it "
                        f"is a single-shot question, not a turn"
                    )
                if describe.get("model") != "vision-model":
                    failures.append(
                        f"the describe call asked for model {describe.get('model')!r}, "
                        f"not the configured vision model"
                    )

                if session._stats_input_tokens != _VISION_PROMPT_TOKENS:
                    failures.append(
                        f"the describe call billed {session._stats_input_tokens} prompt "
                        f"token(s) to the session, expected {_VISION_PROMPT_TOKENS} — a "
                        f"real provider call must reach the usage totals"
                    )
                if session._stats_output_tokens != _VISION_COMPLETION_TOKENS:
                    failures.append(
                        f"the describe call billed {session._stats_output_tokens} "
                        f"completion token(s), expected {_VISION_COMPLETION_TOKENS}"
                    )

                spliced = [
                    m
                    for m in session._messages
                    if isinstance(m.get("content"), str) and _VISION_TEXT in m["content"]
                ]
                if len(spliced) != 1:
                    failures.append(
                        f"{len(spliced)} transcript row(s) carry the description, "
                        f"expected exactly 1"
                    )
                elif png not in str(spliced[0]["content"]):
                    failures.append(
                        f"the spliced row does not name the saved screenshot, so the "
                        f"model cannot cite it: {spliced[0]['content']!r}"
                    )
                elif spliced[0].get("role") != "user":
                    failures.append(
                        f"the spliced row rides role {spliced[0].get('role')!r}, "
                        f"expected 'user'"
                    )
                if any(isinstance(m.get("content"), list) for m in session._messages):
                    failures.append(
                        "a transcript row still carries content parts — an image was "
                        "attached despite a configured vision provider"
                    )

                # The spliced row rides the user role, so the pruner's boundary
                # scan must be told it is not a turn start — otherwise every
                # screenshot folds away the tool scaffolding of the run that took
                # it, exactly the trap a real image attachment already avoids.
                pruned = session_context._prune_messages(list(session._messages))
                if not any(
                    m.get("role") == "assistant" and m.get("tool_calls") for m in pruned
                ):
                    failures.append(
                        "the in-flight assistant tool_calls row was folded away — the "
                        "spliced description moved the turn boundary past the run's "
                        "own evidence chain"
                    )
                if not any(m.get("role") == "tool" for m in pruned):
                    failures.append(
                        "the in-flight tool result was folded away — the spliced "
                        "description moved the turn boundary"
                    )

                answer = _run_main_turn(session, cfg_path)
                if answer != _MAIN_TEXT:
                    failures.append(
                        f"the main turn answered {answer!r}, expected {_MAIN_TEXT!r}"
                    )
                if not main_stub.requests:
                    failures.append("the main endpoint was never called by the turn")
                    return failures

                stray = sum(len(_image_parts(b)) for b in main_stub.requests)
                if stray:
                    failures.append(
                        f"{stray} image part(s) reached the MAIN provider across "
                        f"{len(main_stub.requests)} request(s) — the whole point of "
                        f"the vision block is that they never do"
                    )
                if len(vision_stub.requests) != 1:
                    failures.append(
                        f"the vision endpoint received "
                        f"{len(vision_stub.requests)} request(s) after the turn ran; "
                        f"the turn itself must not call it"
                    )
                prose = "\n".join(_body_text(b) for b in main_stub.requests)
                if _VISION_TEXT not in prose:
                    failures.append(
                        "the main provider never received the description — the "
                        "screenshot became nothing at all"
                    )
                if png not in prose:
                    failures.append(
                        "the main provider's request does not name the saved "
                        "screenshot path"
                    )
            finally:
                session.close()

    return failures


def check_no_vision_block_keeps_pixels() -> list[str]:
    """c. With no vision block the image goes to the main provider, as before."""
    import agent
    from session import Session

    failures: list[str] = []
    png_b64 = base64.b64encode(_png_bytes()).decode("ascii")

    with _serving(_MAIN_TEXT) as main_stub, _serving(_VISION_TEXT) as vision_stub:
        with _project(main_stub.base_url, None) as (tmp, cfg_path):
            session = Session(str(tmp), "main-model", "You are a verification agent.")
            try:
                png = _seed_verify_turn(session, tmp)
                with contextlib.redirect_stderr(io.StringIO()):
                    agent._attach_screenshot(session, png, "tc-1")

                if vision_stub.requests:
                    failures.append(
                        f"the vision endpoint received {len(vision_stub.requests)} "
                        f"request(s) with no 'vision' block configured"
                    )

                shots = [m for m in session._messages if m.get("screenshot")]
                if len(shots) != 1:
                    failures.append(
                        f"expected 1 screenshot row without a vision provider, got "
                        f"{len(shots)}"
                    )
                elif not isinstance(shots[0].get("content"), list):
                    failures.append(
                        f"the screenshot row lost its content parts: {shots[0]!r}"
                    )

                _run_main_turn(session, cfg_path)
                if not main_stub.requests:
                    failures.append("the main endpoint was never called by the turn")
                    return failures

                # The control for check b: the same scan that must find ZERO
                # images with a vision provider has to find the image when there
                # is one, or its zero proves nothing.
                parts = _image_parts(main_stub.requests[-1])
                if len(parts) != 1:
                    failures.append(
                        f"the main provider's request carries {len(parts)} image "
                        f"part(s) with no vision block configured, expected 1 — "
                        f"legacy behaviour changed, or the image scan is blind"
                    )
                else:
                    url = (parts[0].get("image_url") or {}).get("url", "")
                    if url != "data:image/png;base64," + png_b64:
                        failures.append(
                            "the main provider's image part is not the captured "
                            f"PNG's bytes: {url[:64]!r}…"
                        )
                if vision_stub.requests:
                    failures.append(
                        f"the vision endpoint received {len(vision_stub.requests)} "
                        f"request(s) during a run configured without it"
                    )
            finally:
                session.close()

    return failures


def check_vision_failures_are_loud() -> list[str]:
    """d. An unreachable provider and an empty description both fail loudly."""
    import agent
    from session import Session

    failures: list[str] = []

    with _serving(_MAIN_TEXT) as main_stub:
        # Unreachable endpoint: nothing is listening on a port that was bound and
        # released, so the connection is refused rather than hanging.
        dead_url = f"http://127.0.0.1:{_closed_port()}/v1"
        with _project(main_stub.base_url, dead_url) as (tmp, _cfg_path):
            session = Session(str(tmp), "main-model", "You are a verification agent.")
            try:
                png = _seed_verify_turn(session, tmp)
                before = len(session._messages)
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    agent._attach_screenshot(session, png, "tc-1")
                added = session._messages[before:]

                if len(added) != 1:
                    failures.append(
                        f"an unreachable vision provider appended {len(added)} row(s), "
                        f"expected exactly 1"
                    )
                else:
                    content = str(added[0].get("content", ""))
                    if "NOT attached" not in content:
                        failures.append(
                            f"an unreachable vision provider produced a row that does "
                            f"not say the image is NOT attached: {content!r}"
                        )
                    if png not in content:
                        failures.append(
                            f"the failure row does not name the screenshot: {content!r}"
                        )
                if any(isinstance(m.get("content"), list) for m in session._messages):
                    failures.append(
                        "a failed describe call fell back to attaching the image to "
                        "the main model — the fallback must be text, not pixels"
                    )
                if main_stub.requests:
                    failures.append(
                        f"the main endpoint received {len(main_stub.requests)} "
                        f"request(s) while the vision call was failing"
                    )
            finally:
                session.close()

        # Empty description: a reasoning model that emits only hidden reasoning
        # answers 200 with no content. Empty is a failure, not a description.
        with _serving("") as blank_stub:
            with _project(main_stub.base_url, blank_stub.base_url) as (tmp, _cfg_path):
                session = Session(str(tmp), "main-model", "You are a verification agent.")
                try:
                    png = _seed_verify_turn(session, tmp)
                    before = len(session._messages)
                    buf = io.StringIO()
                    with contextlib.redirect_stderr(buf):
                        agent._attach_screenshot(session, png, "tc-1")
                    added = session._messages[before:]

                    if len(blank_stub.requests) != 1:
                        failures.append(
                            f"the blank vision endpoint saw "
                            f"{len(blank_stub.requests)} request(s), expected 1"
                        )
                    if len(added) != 1:
                        failures.append(
                            f"an empty description appended {len(added)} row(s), "
                            f"expected exactly 1"
                        )
                    elif "NOT attached" not in str(added[0].get("content", "")):
                        failures.append(
                            f"an empty description was spliced in as if it were a "
                            f"description: {added[0].get('content')!r}"
                        )
                    if any(isinstance(m.get("content"), list) for m in session._messages):
                        failures.append(
                            "an empty description fell back to attaching the image to "
                            "the main model"
                        )
                finally:
                    session.close()

    return failures


CHECKS = [
    ("config-block-parses", check_config_block_parses),
    ("vision-takes-image-main-gets-text", check_vision_takes_image_main_gets_text),
    ("no-vision-block-keeps-pixels", check_no_vision_block_keeps_pixels),
    ("vision-failures-are-loud", check_vision_failures_are_loud),
]


def main() -> int:
    import tools.registry as registry

    registry.discover()

    all_failures: list[str] = []
    for name, check in CHECKS:
        try:
            failures = check()
        except Exception as exc:  # a raising check is a failing check, never a skip
            import traceback

            failures = [f"{name} raised {type(exc).__name__}: {exc}\n{traceback.format_exc()}"]

        if failures:
            for failure in failures:
                print(f"FAIL [{name}]: {failure}", file=sys.stderr)
            all_failures.extend(failures)
        else:
            print(f"PASS: {name}")

    if all_failures:
        return 1

    print(
        "PASS: a configured vision provider takes the only copy of the image and "
        "returns text the main provider receives as prose, an unconfigured one "
        "leaves the legacy pixel path byte-for-byte intact, and every vision "
        "failure tells the model the image was never attached"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
