"""Net-change annotation on the result envelope, end-to-end through the turn loop.

``turn_report['files_changed']`` is an activity log: a file edited and then
reverted to its original bytes still appears in it. A consumer reading that log
alone would conclude the tree changed when the net state did not — a real hazard
now that a harness steer can direct the agent to revert an edit whose reported
failure it could not reproduce. This script drives the *real* production hot path
(``agent.handle_user_message`` with a stub LLM client, no network) and asserts
the harness annotates each entry truthfully via ``entry['reverted']``:

a. Edit-then-revert — a turn that changes a file's bytes then writes the original
   bytes back, with a passing run_command so the verify gate clears, marks that
   entry ``reverted: True`` while ``verified`` stays True (semantics unchanged).

b. Edit-no-revert — a file changed and left changed carries NO ``reverted`` key.

c. Create-then-delete — a file created this turn then removed via a run_command
   ``rm`` carries ``reverted: True`` (its pre-image was "did not exist" and it no
   longer exists).

d. Unknown pre-image — a file whose FIRST mutation comes from a run_command shell
   side effect has no capturable pre-image, so it carries NO ``reverted`` key even
   when a later scripted call restores its original content: unknown is never
   guessed.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file); it chdir's into its own throwaway temp project dir (the edit tools
and run_command resolve against cwd) and restores the cwd afterward, disables
memory side effects for the run, and touches no repo files.
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


def _drive_turn(task: str, script: list, tmp_prefix: str, seed: dict[str, str]):
    """Drive one scripted turn through the real turn loop; return the session.

    Materializes *seed* (relative path -> content) in a fresh temp project dir,
    activates the catalog tools the scripts use, isolates cwd + memory, and
    returns the session so the caller can read ``session.turn_report``. Restores
    cwd/memory afterward. Touches no repo files.
    """
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient
    from session import Session

    for name in ("read_file", "create_file", "update_file", "run_command"):
        registry.activate(name)

    original_cwd = os.getcwd()
    original_memory = agent.MEMORY_ENABLED
    tmp = tempfile.mkdtemp(prefix=tmp_prefix)
    os.chdir(tmp)
    agent.MEMORY_ENABLED = False
    try:
        for rel, content in seed.items():
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as fh:
                fh.write(content)
        session = Session(tmp, "test-model", "You are a test agent.")
        client = _StubClient(script)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                task, session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )
        return session
    finally:
        agent.MEMORY_ENABLED = original_memory
        os.chdir(original_cwd)


def _entry_for(session, suffix: str) -> dict | None:
    """The files_changed entry whose resolved path ends with *suffix*, or None."""
    for entry in session.turn_report["files_changed"]:
        if str(entry.get("path", "")).endswith(suffix):
            return entry
    return None


def check_edit_then_revert() -> list[str]:
    """a. A byte-for-byte revert (plus a passing run) is flagged reverted, verified."""
    original = "value = 1\n"
    script = [
        # Read first — update_file gates on a prior read of the file.
        _tool_call_response("read_file", {"path": "mod.py"}),
        # Change the file's bytes...
        _tool_call_response("update_file", {"path": "mod.py", "content": "value = 2\n"}),
        # ...then write the ORIGINAL bytes back — a net no-op on disk.
        _tool_call_response("update_file", {"path": "mod.py", "content": original}),
        # A passing run clears the verify gate so verified can be True.
        _tool_call_response("run_command", {"cmd": "exit 0"}),
        _final_answer_response("Reverted the speculative edit; nothing changed."),
    ]
    session = _drive_turn(
        "mod.py mishandles the empty case; reproduce and fix",
        script,
        "envelope-revert-",
        {"mod.py": original},
    )

    failures: list[str] = []
    entry = _entry_for(session, "mod.py")
    if entry is None:
        failures.append("edit-then-revert: mod.py missing from files_changed")
        return failures
    if entry.get("reverted") is not True:
        failures.append(
            f"edit-then-revert: expected reverted=True on mod.py, entry={entry!r}"
        )
    if session.turn_report["verified"] is not True:
        failures.append(
            f"edit-then-revert: expected verified=True (semantics unchanged), "
            f"got {session.turn_report['verified']!r}"
        )
    return failures


def check_edit_no_revert() -> list[str]:
    """b. A file left changed carries no reverted key."""
    script = [
        _tool_call_response("read_file", {"path": "mod.py"}),
        _tool_call_response("update_file", {"path": "mod.py", "content": "value = 2\n"}),
        _tool_call_response("run_command", {"cmd": "exit 0"}),
        _final_answer_response("Applied the fix."),
    ]
    session = _drive_turn(
        "bump the value in mod.py",
        script,
        "envelope-changed-",
        {"mod.py": "value = 1\n"},
    )

    failures: list[str] = []
    entry = _entry_for(session, "mod.py")
    if entry is None:
        failures.append("edit-no-revert: mod.py missing from files_changed")
        return failures
    if "reverted" in entry:
        failures.append(
            f"edit-no-revert: mod.py was left changed but carries a reverted key, "
            f"entry={entry!r}"
        )
    return failures


def check_create_then_delete() -> list[str]:
    """c. A file created then rm'd via run_command is flagged reverted."""
    script = [
        _tool_call_response("create_file", {"path": "scratch.py", "content": "tmp = 1\n"}),
        _tool_call_response("run_command", {"cmd": "rm scratch.py"}),
        _final_answer_response("Created a scratch file then removed it."),
    ]
    session = _drive_turn(
        "add a throwaway scratch.py then clean it up",
        script,
        "envelope-create-delete-",
        {},
    )

    failures: list[str] = []
    entry = _entry_for(session, "scratch.py")
    if entry is None:
        failures.append("create-then-delete: scratch.py missing from files_changed")
        return failures
    if entry.get("reverted") is not True:
        failures.append(
            f"create-then-delete: expected reverted=True on scratch.py "
            f"(created then deleted), entry={entry!r}"
        )
    return failures


def check_unknown_preimage() -> list[str]:
    """d. A file first mutated by run_command is never flagged, even if restored."""
    original = "line = 1\n"
    script = [
        # FIRST mutation is a shell side effect — no capturable pre-image.
        _tool_call_response("run_command", {"cmd": "echo 'extra = 2' >> notes.py"}),
        # Read the post-shell state (update_file gates on a prior read), then a
        # later scripted call restores the original bytes; the harness must NOT
        # infer a revert, because it never knew this file's turn-start state.
        _tool_call_response("read_file", {"path": "notes.py"}),
        _tool_call_response("update_file", {"path": "notes.py", "content": original}),
        _tool_call_response("run_command", {"cmd": "exit 0"}),
        _final_answer_response("Touched notes.py via the shell then rewrote it."),
    ]
    session = _drive_turn(
        "append a note to notes.py",
        script,
        "envelope-unknown-",
        {"notes.py": original},
    )

    failures: list[str] = []
    entry = _entry_for(session, "notes.py")
    if entry is None:
        failures.append("unknown-preimage: notes.py missing from files_changed")
        return failures
    if "reverted" in entry:
        failures.append(
            f"unknown-preimage: notes.py's first mutation was a shell side effect "
            f"(no pre-image) but it carries a reverted key — unknown was guessed. "
            f"entry={entry!r}"
        )
    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("edit-then-revert", check_edit_then_revert),
        ("edit-no-revert", check_edit_no_revert),
        ("create-then-delete", check_create_then_delete),
        ("unknown-preimage", check_unknown_preimage),
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
        "PASS: files_changed entries are annotated reverted only on a known "
        "net no-op — a byte-for-byte revert (verified still True) and a "
        "create-then-delete are flagged, a left-changed file is not, and a file "
        "first mutated by run_command is never flagged even when later restored"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
