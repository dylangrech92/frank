"""Verify-mode contract check: evidence gate, terminal exit, vision, artifacts.

Verify mode is the only mode with a *terminal* tool — an accepted ``report``
call ends the turn on the spot — and the only mode that puts images on the
wire. Both are new turn-loop behaviours that no other eval covers, so this
script exercises them against the real production path with zero mocks: the
real ``tools.registry.dispatch``, the real ``agent.handle_user_message`` (with
the shared scripted-LLM stub, no network), the real ``Session`` and its on-disk
transcript, the real ``session_context`` pruners, and the real
``main._browser_finish_run``. No browser is launched — the browser tools
themselves are proven by driving a real page, not by a unit check.

Asserts:

a. The evidence gate, through real dispatch:
   - ``report`` outside verify mode is refused with ``code="not-in-mode"``;
   - a 'pass' assertion with no evidence is refused with
     ``code="evidence-required"``;
   - a top-level 'pass' over a failed assertion is refused with
     ``code="incoherent-verdict"``;
   - an evidenced, coherent payload is accepted.

b. The terminal exit, through the real turn loop: a REJECTED report does not
   end the turn (the model keeps its turn and can fix the verdict — a gate
   that costs the run is a gate the model cannot survive), the next ACCEPTED
   report does end it (the trailing scripted response is never consumed), the
   returned answer is the rendered verdict, ``turn_report["report"]`` carries
   the structured payload, and ``verified`` stays ``None`` — it answers "did
   this turn verify the files it changed", and a verify run changes none.

c. The vision path, which touches three separate contracts that a screenshot
   would otherwise break silently:
   - the transcript ON DISK holds an ``image_ref`` path, never base64;
   - ``_prune_messages`` does not treat a screenshot as a turn boundary (if it
     did, a verify run would fold away its own evidence chain the moment it
     looked at anything);
   - ``_prune_images`` keeps only the newest ``MAX_IMAGES`` and degrades the
     rest to a path placeholder;
   - ``llm.to_wire_messages`` strips the ``screenshot`` flag but keeps the
     content parts intact;
   - ``compaction``'s token estimate for an image row is bounded, not
     proportional to the base64 length (an unbounded estimate would trip the
     compaction ladder on a context that was never over cap);
   - an unreadable image path produces an explicit "NOT attached" message
     rather than a crash or a silent gap.

d. Artifact truthfulness: ``main._browser_finish_run`` reports ``trace`` only
   when ``trace.zip`` actually exists on disk, and ``None`` when it does not.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before
exec'ing this file); it also adds the repo root itself if not already present,
so the script can be invoked directly with
``PYTHONPATH=. .venv/bin/python evals/verify_mode_contract.py``. Every check
restores whichever mode it found active in a ``finally``, and all writes land
in throwaway temp directories — no repo file is touched.
"""

from __future__ import annotations

import base64
import contextlib
import io
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# A minimal but genuinely valid 1x1 PNG, built here rather than committed as a
# binary fixture: the vision path reads real bytes off disk and base64-encodes
# them, so the test needs real bytes, not a stub.
def _png_bytes() -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


_GOOD_REPORT = {
    "verdict": "pass",
    "plan": ["the login form submits", "no console errors"],
    "assertions": [
        {
            "assertion": "the login form submits",
            "verdict": "pass",
            "evidence": {
                "kind": "network",
                "detail": "POST /login returned 302 to /dashboard",
            },
        },
        {
            "assertion": "no console errors",
            "verdict": "pass",
            "evidence": {"kind": "console", "detail": "console_logs returned 0 errors"},
        },
    ],
    "observations": "The dashboard rendered in under a second.",
}

_UNEVIDENCED_REPORT = {
    "verdict": "pass",
    "plan": ["the login form submits"],
    "assertions": [
        {
            "assertion": "the login form submits",
            "verdict": "pass",
            "evidence": {"kind": "network", "detail": ""},
        }
    ],
    "observations": "",
}

_INCOHERENT_REPORT = {
    "verdict": "pass",
    "plan": ["the login form submits"],
    "assertions": [
        {
            "assertion": "the login form submits",
            "verdict": "fail",
            "evidence": {"kind": "network", "detail": "POST /login returned 500"},
        }
    ],
    "observations": "",
}


def check_evidence_gate() -> list[str]:
    """a. The gate, through real dispatch: mode gate, evidence, coherence."""
    import tools.registry as registry

    failures: list[str] = []
    saved_mode = registry.current_mode()
    try:
        # The mode gate first: report must not be reachable from code mode, or
        # the terminal-exit path could fire in a mode that has no verdict to
        # give.
        registry.activate_mode("code")
        result = registry.dispatch("report", dict(_GOOD_REPORT))
        if result.status != "error" or result.code != "not-in-mode":
            failures.append(
                f"report dispatched from code mode returned "
                f"status={result.status!r} code={result.code!r}; expected an "
                f"error with code='not-in-mode'"
            )

        registry.activate_mode("verify")

        result = registry.dispatch("report", dict(_UNEVIDENCED_REPORT))
        if result.status != "error" or result.code != "evidence-required":
            failures.append(
                f"an unevidenced 'pass' assertion returned status="
                f"{result.status!r} code={result.code!r}; expected an error "
                f"with code='evidence-required' — the evidence gate did not hold"
            )

        result = registry.dispatch("report", dict(_INCOHERENT_REPORT))
        if result.status != "error" or result.code != "incoherent-verdict":
            failures.append(
                f"a top-level 'pass' over a failed assertion returned status="
                f"{result.status!r} code={result.code!r}; expected an error "
                f"with code='incoherent-verdict'"
            )

        result = registry.dispatch("report", dict(_GOOD_REPORT))
        if result.status != "success":
            failures.append(
                f"a fully evidenced, coherent report was refused: "
                f"status={result.status!r} code={result.code!r} "
                f"body={result.body!r} — the gate rejects valid input"
            )
    finally:
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    return failures


def check_terminal_exit_ends_turn() -> list[str]:
    """b. A rejected report keeps the turn; an accepted one ends it."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient, disable_memory_hooks

    disable_memory_hooks()
    from llm import ChatResponse, ToolCall
    from session import Session

    failures: list[str] = []
    saved_mode = registry.current_mode()
    registry.activate_mode("verify")

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="verify-terminal-")
    os.chdir(tmp)
    session = None
    try:
        session = Session(tmp, "test-model", "You are a verification agent.")

        counter = {"n": 0}

        def _report_call(payload: dict):
            def factory():
                counter["n"] += 1
                return ChatResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id=f"vm-call-{counter['n']}",
                            name="report",
                            arguments=dict(payload),
                        )
                    ],
                )

            return factory

        def _never_reached():
            return ChatResponse(text="THIS ANSWER MUST NEVER BE USED", tool_calls=[])

        # Round 1's report is REJECTED by the gate, so the turn must continue;
        # round 2's is ACCEPTED, so the turn must end there and the round-3
        # response must never be requested.
        client = _StubClient([
            _report_call(_UNEVIDENCED_REPORT),
            _report_call(_GOOD_REPORT),
            _never_reached,
        ])

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            answer = agent.handle_user_message(
                "verify the login flow at http://localhost/",
                session,
                client,  # pyright: ignore[reportArgumentType]
                on_delta=None,
            )

        if client.calls != 2:
            failures.append(
                f"the turn made {client.calls} chat call(s); expected exactly 2 "
                f"— a rejected report must keep the turn alive (call 2) and an "
                f"accepted one must end it before call 3"
            )

        if "THIS ANSWER MUST NEVER BE USED" in (answer or ""):
            failures.append(
                "the turn ran past the accepted report and used a later "
                "response as its answer — the terminal exit did not fire"
            )
        if not (answer or "").startswith("VERDICT: pass"):
            failures.append(
                f"the turn's answer does not start with the rendered verdict: "
                f"{(answer or '')[:120]!r}"
            )
        if "evidence (network): POST /login returned 302 to /dashboard" not in (answer or ""):
            failures.append(
                f"the rendered answer does not carry the assertion evidence: "
                f"{answer!r}"
            )

        report = session.turn_report.get("report")
        if not isinstance(report, dict):
            failures.append(f"turn_report['report'] is not a dict: {report!r}")
        else:
            if report.get("verdict") != "pass":
                failures.append(f"turn_report['report']['verdict'] == {report.get('verdict')!r}, expected 'pass'")
            if report.get("plan") != _GOOD_REPORT["plan"]:
                failures.append(f"turn_report['report']['plan'] == {report.get('plan')!r}")
            if len(report.get("assertions") or []) != 2:
                failures.append(
                    f"turn_report['report']['assertions'] has "
                    f"{len(report.get('assertions') or [])} entries, expected 2"
                )
            if report.get("observations") != _GOOD_REPORT["observations"]:
                failures.append(
                    f"turn_report['report']['observations'] == {report.get('observations')!r}"
                )

        # verified answers "did this turn verify the files it changed"; a verify
        # run changes no files, so None is the only honest value. Deriving it
        # from the verdict would silently overload one field with two meanings.
        if session.turn_report.get("verified") is not None:
            failures.append(
                f"turn_report['verified'] == "
                f"{session.turn_report.get('verified')!r}; expected None — a "
                f"verify run changes no files, so there is nothing to verify"
            )
        if session.turn_report.get("files_changed"):
            failures.append(
                f"turn_report['files_changed'] is non-empty after a verify "
                f"turn: {session.turn_report.get('files_changed')!r}"
            )
    finally:
        # Session ids are timestamp+pid, so a live lock from one check would
        # collide with the next check's session inside the same second.
        if session is not None:
            session.close()
        os.chdir(original_cwd)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    return failures


def check_screenshot_transcript_and_context() -> list[str]:
    """c. The vision path: on-disk transcript, pruners, wire, token estimate."""
    import agent
    import compaction
    import session as session_mod
    import session_context
    from evals._stub import disable_memory_hooks
    from llm import ToolCall, to_wire_messages

    disable_memory_hooks()
    from session import Session

    failures: list[str] = []
    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="verify-vision-")
    os.chdir(tmp)
    session = None
    try:
        png = os.path.join(tmp, "shot.png")
        with open(png, "wb") as fh:
            fh.write(_png_bytes())

        session = Session(tmp, "test-model", "You are a verification agent.")
        # A real verify turn's shape, not a simplified one: the scaffolding the
        # screenshot must not fold away is the assistant's tool_calls row and
        # its tool result. A fixture of bare text rows would survive a broken
        # boundary scan by accident and prove nothing.
        session.append_user("verify the dashboard")
        session.append_assistant(
            "navigating",
            tool_calls=[ToolCall(id="tc-1", name="navigate", arguments={"url": "http://localhost/"})],
        )
        session.append_tool_result("tc-1", "navigate", "navigated to http://localhost/")
        agent._attach_screenshot(session, png, "tc-1")

        shot_rows = [m for m in session._messages if m.get("screenshot")]
        if len(shot_rows) != 1:
            failures.append(f"expected 1 screenshot row in the session, got {len(shot_rows)}")
            return failures
        row = shot_rows[0]
        if row.get("role") != "user":
            failures.append(f"screenshot row role is {row.get('role')!r}, expected 'user'")
        parts = row.get("content")
        if not isinstance(parts, list) or not any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in parts
        ):
            failures.append(f"screenshot row carries no image_url part: {parts!r}")

        # The transcript on disk must never carry the base64 payload: it is
        # written on every append, and a 200KB PNG re-serialized per row turns
        # a session file into hundreds of megabytes.
        transcript = session.transcript_path
        if not transcript.exists():
            failures.append(f"the session transcript was never written to {transcript}")
        else:
            raw = transcript.read_text(encoding="utf-8")
            if "base64," in raw:
                failures.append(
                    f"the persisted transcript at {transcript} contains a base64 "
                    f"data URI — the image payload was written to disk"
                )
            if '"image_ref"' not in raw:
                failures.append(
                    f"the persisted transcript at {transcript} carries no "
                    f"image_ref part — the path was not recorded"
                )
            if png not in raw:
                failures.append(
                    f"the persisted transcript does not name the image path {png}"
                )

        # The boundary scan: the screenshot must NOT be read as a turn start,
        # or the in-flight turn's own scaffolding folds away behind it.
        pruned = session_context._prune_messages(list(session._messages))
        if not any(
            m.get("role") == "user" and m.get("content") == "verify the dashboard"
            for m in pruned
        ):
            failures.append(
                "the real user message did not survive pruning — the screenshot "
                "was treated as the turn boundary"
            )
        if not any(m.get("role") == "assistant" and m.get("tool_calls") for m in pruned):
            failures.append(
                "the in-flight assistant tool_calls row was folded away — the "
                "screenshot moved the turn boundary past the run's own evidence "
                "chain"
            )
        if not any(m.get("role") == "tool" and m.get("name") == "navigate" for m in pruned):
            failures.append(
                "the in-flight tool result was folded away — the screenshot "
                "moved the turn boundary past the run's own evidence chain"
            )

        # The wire boundary: the flag is harness bookkeeping and must not reach
        # the provider, but the content parts must survive untouched.
        wire = to_wire_messages([dict(row)])
        if "screenshot" in wire[0]:
            failures.append(f"the 'screenshot' flag reached the wire: {wire[0].keys()}")
        if not isinstance(wire[0].get("content"), list):
            failures.append(f"the wire message lost its content parts: {wire[0]!r}")

        # The image budget, asserted through the REAL assembly entry point that
        # the turn loop calls — not by invoking _prune_images directly, which
        # would still pass if the pruner were never wired into assemble_context.
        # ATTACHED is a fixed count, deliberately NOT derived from MAX_IMAGES:
        # a bound checked against itself can never catch the bound growing.
        ATTACHED = 5
        if session_mod.MAX_IMAGES >= ATTACHED:
            failures.append(
                f"MAX_IMAGES is {session_mod.MAX_IMAGES}, which is no longer a "
                f"small budget — a long verify run would carry that many full "
                f"images in every request"
            )
        for i in range(ATTACHED):
            extra = os.path.join(tmp, f"shot{i}.png")
            with open(extra, "wb") as fh:
                fh.write(_png_bytes())
            agent._attach_screenshot(session, extra, f"call-extra-{i}")

        assembled = session_context.assemble_context(session, [])
        with_pixels = [
            m
            for m in assembled
            if isinstance(m.get("content"), list)
            and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])
        ]
        if len(with_pixels) > session_mod.MAX_IMAGES:
            failures.append(
                f"{len(with_pixels)} screenshots still carry pixels in the "
                f"assembled context; MAX_IMAGES is {session_mod.MAX_IMAGES} — "
                f"the pruner is not reached by assemble_context"
            )

        # The BACKEND ceiling, counted on the wire the way the server counts it:
        # image parts, not messages carrying them. This is deliberately a hard 1
        # rather than MAX_IMAGES — the check above only proves the pruner ran,
        # and it passed for the whole time MAX_IMAGES was 3, while the llama.cpp
        # server rejected every prompt past the first image ("At most 1 image(s)
        # may be provided in one prompt", HTTP 400, not retried) and killed the
        # run on the second screenshot. A bound checked against the constant it
        # is bounding cannot catch that; only a fact from outside can.
        wire_images = sum(
            1
            for m in to_wire_messages(assembled)
            if isinstance(m.get("content"), list)
            for p in m["content"]
            if isinstance(p, dict) and p.get("type") == "image_url"
        )
        if wire_images > 1:
            failures.append(
                f"the wire payload carries {wire_images} image parts after "
                f"{ATTACHED} screenshots; every OpenAI-compatible backend this "
                f"project targets accepts at most 1 per prompt, so this request "
                f"would 400 and end the run. If a backend that accepts more is "
                f"now in use, raise MAX_IMAGES and this bound together."
            )

        placeholders = [
            m for m in assembled if isinstance(m.get("content"), str) and "pruned" in m["content"]
        ]
        if not placeholders:
            failures.append("no pruned screenshot collapsed to a path placeholder")
        elif png not in " ".join(str(m["content"]) for m in placeholders):
            failures.append(
                "the oldest screenshot's placeholder does not name its file — "
                "the model can no longer cite the artifact"
            )

        # The token estimate must be bounded by the image constant, not by the
        # length of the base64 payload: a proportional estimate would trip the
        # compaction ladder on a context that was never over cap.
        big = "data:image/png;base64," + base64.b64encode(b"\x00" * 200_000).decode("ascii")
        big_row = {
            "role": "user",
            "content": [
                {"type": "text", "text": "shot"},
                {"type": "image_url", "image_url": {"url": big}},
            ],
            "screenshot": True,
        }
        estimated = compaction.estimate_tokens([big_row])
        if estimated > compaction._IMAGE_PART_TOKENS * 2:
            failures.append(
                f"an image row estimated at {estimated} tokens against an "
                f"_IMAGE_PART_TOKENS budget of {compaction._IMAGE_PART_TOKENS} "
                f"— the base64 payload is being counted as prose"
            )

        # An unreadable path is reported to the model, not swallowed.
        before = len(session._messages)
        agent._attach_screenshot(session, os.path.join(tmp, "does-not-exist.png"), "call-missing")
        added = session._messages[before:]
        if len(added) != 1:
            failures.append(f"a failed screenshot read appended {len(added)} rows, expected 1")
        elif "NOT attached" not in str(added[0].get("content", "")):
            failures.append(
                f"a failed screenshot read did not tell the model the image is "
                f"missing: {added[0]!r}"
            )
        elif any(m.get("screenshot") for m in added):
            failures.append(
                "a failed screenshot read was flagged as a screenshot row — it "
                "would be counted against the image budget it never used"
            )
    finally:
        if session is not None:
            session.close()
        os.chdir(original_cwd)

    return failures


def check_artifacts_never_guessed() -> list[str]:
    """d. trace is reported only when trace.zip actually exists on disk."""
    import main
    import tools.registry as registry
    from evals._stub import disable_memory_hooks

    disable_memory_hooks()
    from runtime import browser
    from session import Session

    failures: list[str] = []
    saved_mode = registry.current_mode()
    registry.activate_mode("verify")

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="verify-artifacts-")
    os.chdir(tmp)
    try:
        run_dir = Path(tmp) / "runs" / "session-1"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Never started, so shutdown() is a no-op and no trace.zip is written —
        # exactly the state of a verify run that only ever called http_request.
        bs = browser.get_session()
        bs.run_dir = run_dir

        session = Session(tmp, "test-model", "You are a verification agent.")
        main._browser_finish_run(session)

        if session.turn_report.get("artifacts_dir") != str(run_dir):
            failures.append(
                f"artifacts_dir == {session.turn_report.get('artifacts_dir')!r}, "
                f"expected {str(run_dir)!r}"
            )
        if session.turn_report.get("trace") is not None:
            failures.append(
                f"trace == {session.turn_report.get('trace')!r} with no "
                f"trace.zip on disk — the envelope guessed a path instead of "
                f"reporting None"
            )

        (run_dir / "trace.zip").write_bytes(b"PK\x03\x04")
        main._browser_finish_run(session)
        if session.turn_report.get("trace") != str(run_dir / "trace.zip"):
            failures.append(
                f"trace == {session.turn_report.get('trace')!r} with a real "
                f"trace.zip present; expected {str(run_dir / 'trace.zip')!r}"
            )

        # And the whole thing is inert outside verify mode: sentinels the
        # function would have to overwrite must survive it untouched.
        registry.activate_mode("code")
        session.turn_report["artifacts_dir"] = "SENTINEL"
        session.turn_report["trace"] = "SENTINEL"
        main._browser_finish_run(session)
        if (
            session.turn_report["artifacts_dir"] != "SENTINEL"
            or session.turn_report["trace"] != "SENTINEL"
        ):
            failures.append(
                f"_browser_finish_run wrote artifacts in code mode: "
                f"artifacts_dir={session.turn_report['artifacts_dir']!r} "
                f"trace={session.turn_report['trace']!r}"
            )
        session.close()
    finally:
        os.chdir(original_cwd)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    return failures


CHECKS = [
    ("evidence-gate", check_evidence_gate),
    ("terminal-exit-ends-turn", check_terminal_exit_ends_turn),
    ("screenshot-transcript-and-context", check_screenshot_transcript_and_context),
    ("artifacts-never-guessed", check_artifacts_never_guessed),
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
        "PASS: verify mode's evidence gate holds, an accepted report ends the "
        "turn while a rejected one does not, screenshots stay off disk and "
        "inside the image budget without moving the turn boundary, and "
        "artifacts are reported only when they exist"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
