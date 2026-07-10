"""Repeat-render dedup check, end-to-end through the real turn loop.

``_repeat_call_check`` owns repeated identical *successes*. Before this change it
appended a no-progress suffix to the FULL re-rendered body of every repeat #2..#cap
— so a model that re-issues an identical read-only call is handed a byte-identical
body twice more, wasting context and rewarding the re-issue. This change replaces
that body with a short stub for a read-only tool whose result has provably not
changed, gated by two safety conditions that each guard a distinct hazard:

(d) identical arguments do NOT imply an identical result — a re-read of a file the
    model just edited MUST get the fresh body — so a content fingerprint gates the
    stub; on mismatch the full fresh body is returned.
(e) a compaction may have folded the earlier result out of context — a stub that
    points at a vanished result strands the model — so the stub is withheld unless
    the turn's compaction count is unchanged since the last full render.

This script drives the *real* production hot path (``agent.handle_user_message``
with a stub LLM client, no network) and asserts:

a. Identical successful read twice (``read_file`` on an unchanged file) → the first
   tool-result row carries the body; the second is the stub (body text ABSENT,
   stub marker present).
b. Read → edit the same file (``update_file``) → identical re-read → the re-read
   renders the full fresh body reflecting the NEW content and is NOT stubbed
   (fingerprint mismatch defeats the dedup — condition (d)).
c. An exempt / non-parallel_safe tool (``run_command`` with an identical cmd twice)
   is never stubbed — verification tools are exempt from the whole repeat machinery.
d. The extracted ``_repeat_render`` is driven directly with a fabricated stamp
   dict: same key and fingerprint but a bumped compaction count → full render (not
   the stub) is returned and the stamp is re-stamped at the new count (condition
   (e)); the matching-count control returns the stub.

Registry-reset mechanism: checks (b) and (c) activate catalog-gated write /
verification tools (``update_file`` / ``run_command``). The tools registry is
module-global with no public deactivate API, so each such check snapshots
``tools.registry._active`` and restores it in a finally (mirroring verify_tool_
wiring.py's ``_active`` snapshot pattern). ``read_file`` is PINNED and needs no
activation.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file); each e2e check chdir's into its own throwaway temp project dir
(read_file / update_file / run_command resolve against cwd) and restores the cwd
afterward, disables memory side effects, and touches no repo files.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile


def _tool_call_response(name: str, arguments: dict):
    """Return a factory yielding a ChatResponse that issues one tool call.

    Each invocation gets a fresh call id (the wire protocol pairs a tool result
    to its call id) but the same name+arguments.
    """
    from llm import ChatResponse, ToolCall

    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        tc = ToolCall(id=f"call-{counter['n']}", name=name, arguments=dict(arguments))
        return ChatResponse(text="", tool_calls=[tc])

    return factory


def _final_answer_response(text: str):
    """Return a factory yielding a no-tool-call ChatResponse (turn ends)."""
    from llm import ChatResponse

    def factory():
        return ChatResponse(text=text, tool_calls=[])

    return factory


_STUB_MARKER = "output omitted"  # appears only in _REPEAT_DEDUP_STUB, never the suffix


def _tool_rows(session, name: str) -> list[str]:
    """Rendered content of every tool-result row for *name*, in dispatch order."""
    return [
        str(m.get("content", ""))
        for m in session._messages
        if m.get("role") == "tool" and m.get("name") == name
    ]


def check_identical_read_stubbed() -> list[str]:
    """a. Two identical successful reads: first body shown, second stubbed."""
    import agent
    from evals._stub import _StubClient
    from session import Session

    failures: list[str] = []

    original_cwd = os.getcwd()
    original_memory = agent.MEMORY_ENABLED
    tmp = tempfile.mkdtemp(prefix="repeat-dedup-read-")
    os.chdir(tmp)
    agent.MEMORY_ENABLED = False
    try:
        with open(os.path.join(tmp, "readme.txt"), "w", encoding="utf-8") as fh:
            fh.write("hello world\n")

        session = Session(tmp, "test-model", "You are a test agent.")
        script = [
            _tool_call_response("read_file", {"path": "readme.txt"}),
            _tool_call_response("read_file", {"path": "readme.txt"}),
            _final_answer_response("Read it twice."),
        ]
        client = _StubClient(script)

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                "read readme.txt twice", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )

        rows = _tool_rows(session, "read_file")
        if len(rows) != 2:
            failures.append(f"expected 2 read_file result rows, got {len(rows)}")
            return failures

        if "hello world" not in rows[0]:
            failures.append(f"first read did not render the file body: {rows[0]!r}")
        if _STUB_MARKER in rows[0]:
            failures.append(f"first read was already stubbed: {rows[0]!r}")
        if "hello world" in rows[1]:
            failures.append(
                f"second identical read re-emitted the full body instead of a "
                f"stub: {rows[1]!r}"
            )
        if _STUB_MARKER not in rows[1]:
            failures.append(
                f"second identical read was not deduped to a stub: {rows[1]!r}"
            )
    finally:
        agent.MEMORY_ENABLED = original_memory
        os.chdir(original_cwd)

    return failures


def check_reread_after_edit_full_body() -> list[str]:
    """b. Read → edit → identical re-read renders the fresh body, not a stub."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient
    from session import Session

    failures: list[str] = []

    saved_active = set(registry._active)
    registry.activate("update_file")  # catalog-gated write tool

    original_cwd = os.getcwd()
    original_memory = agent.MEMORY_ENABLED
    tmp = tempfile.mkdtemp(prefix="repeat-dedup-edit-")
    os.chdir(tmp)
    agent.MEMORY_ENABLED = False
    try:
        with open(os.path.join(tmp, "conf.txt"), "w", encoding="utf-8") as fh:
            fh.write("old_marker = 1\n")

        session = Session(tmp, "test-model", "You are a test agent.")
        # read (stamps the read registry so update_file's freshness gate passes),
        # overwrite with new content, then re-read the SAME path with identical args.
        script = [
            _tool_call_response("read_file", {"path": "conf.txt"}),
            _tool_call_response("update_file", {"path": "conf.txt", "content": "new_marker = 2\n"}),
            _tool_call_response("read_file", {"path": "conf.txt"}),
            _final_answer_response("Edited then re-read."),
        ]
        client = _StubClient(script)

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                "update conf.txt then re-read it", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )

        rows = _tool_rows(session, "read_file")
        if len(rows) != 2:
            failures.append(f"expected 2 read_file result rows, got {len(rows)}")
            return failures

        # The re-read is the SECOND identical read_file call — condition (d) must
        # defeat the dedup because the file changed between the two reads.
        if _STUB_MARKER in rows[1]:
            failures.append(
                f"re-read after an edit was wrongly stubbed — a changed body MUST "
                f"be re-rendered (condition d): {rows[1]!r}"
            )
        if "new_marker" not in rows[1]:
            failures.append(
                f"re-read did not render the NEW file content: {rows[1]!r}"
            )
        if "old_marker" in rows[1]:
            failures.append(
                f"re-read rendered stale content instead of the edit: {rows[1]!r}"
            )
    finally:
        agent.MEMORY_ENABLED = original_memory
        os.chdir(original_cwd)
        registry._active.clear()
        registry._active.update(saved_active)

    return failures


def check_exempt_tool_never_stubbed() -> list[str]:
    """c. An identical run_command twice is never stubbed (exempt, non-parallel_safe)."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient
    from session import Session

    failures: list[str] = []

    saved_active = set(registry._active)
    registry.activate("run_command")

    original_cwd = os.getcwd()
    original_memory = agent.MEMORY_ENABLED
    tmp = tempfile.mkdtemp(prefix="repeat-dedup-cmd-")
    os.chdir(tmp)
    agent.MEMORY_ENABLED = False
    try:
        session = Session(tmp, "test-model", "You are a test agent.")
        script = [
            _tool_call_response("run_command", {"cmd": "echo dedup_probe"}),
            _tool_call_response("run_command", {"cmd": "echo dedup_probe"}),
            _final_answer_response("Ran it twice."),
        ]
        client = _StubClient(script)

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                "run echo twice", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )

        rows = _tool_rows(session, "run_command")
        if len(rows) != 2:
            failures.append(f"expected 2 run_command result rows, got {len(rows)}")
            return failures
        for i, row in enumerate(rows):
            if _STUB_MARKER in row:
                failures.append(
                    f"run_command row {i} was stubbed — exempt / non-parallel_safe "
                    f"tools must never be deduped: {row!r}"
                )
    finally:
        agent.MEMORY_ENABLED = original_memory
        os.chdir(original_cwd)
        registry._active.clear()
        registry._active.update(saved_active)

    return failures


def check_compaction_defeats_dedup() -> list[str]:
    """d. Direct-drive _repeat_render: a bumped compaction count forces a full body."""
    import agent

    failures: list[str] = []

    key: tuple[str, str] = ("read_file", '{"path": "a.txt"}')
    rendered = "[read_file(success)]\n     1\tstable body\ntotal_lines: 1"
    fingerprint = agent._render_fingerprint(rendered)

    # (e) same key + same fingerprint, but a compaction happened since the last
    # full render (stamp says 0, current turn is at 1) -> the earlier result may
    # have been folded away, so the body must be re-emitted and re-stamped.
    stamps: dict[tuple[str, str], tuple[str, int]] = {key: (fingerprint, 0)}
    out = agent._repeat_render("read_file", key, rendered, stamps, 1)
    if _STUB_MARKER in out:
        failures.append(
            f"a compaction since the last full render did NOT defeat the stub — "
            f"the model would be stranded (condition e): {out!r}"
        )
    if "stable body" not in out:
        failures.append(f"full body was not re-emitted after a compaction: {out!r}")
    if stamps[key] != (fingerprint, 1):
        failures.append(
            f"stamp was not re-stamped at the new compaction count: {stamps[key]!r}"
        )

    # Positive control: identical fingerprint AND unchanged compaction count -> stub.
    stamps2: dict[tuple[str, str], tuple[str, int]] = {key: (fingerprint, 3)}
    out2 = agent._repeat_render("read_file", key, rendered, stamps2, 3)
    if _STUB_MARKER not in out2:
        failures.append(
            f"an unchanged body at an unchanged compaction count was not "
            f"deduped to a stub: {out2!r}"
        )
    if "stable body" in out2:
        failures.append(f"the stub still carried the full body: {out2!r}")

    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("identical-read-stubbed", check_identical_read_stubbed),
        ("reread-after-edit-full-body", check_reread_after_edit_full_body),
        ("exempt-tool-never-stubbed", check_exempt_tool_never_stubbed),
        ("compaction-defeats-dedup", check_compaction_defeats_dedup),
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
        "PASS: an identical successful read-only repeat is deduped to a stub; a "
        "re-read after an edit and a post-compaction repeat both re-emit the full "
        "body; and exempt / non-parallel_safe tools are never stubbed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
