"""E8 loop-guard escalation check, end-to-end through the real turn loop.

The repeat-call hard cap refuses an identical tool call once it has run
``_REPEAT_CALL_CAP`` times this turn, but a determined model can keep re-issuing
the blocked call — each round a full LLM round-trip that dispatches nothing —
until an external timeout. Worse, when compaction folds the block error and the
tool result out of context, the model loses even the feedback that it is stuck.
E8 adds two things this script exercises against the *real* production hot path
(``agent.handle_user_message`` with a stub LLM client, no network):

    A. Escalation — after ``_BLOCKED_STREAK_CAP`` consecutive blocked calls with
       no dispatch in between, the turn is force-finalized with a plain-text
       give-up answer instead of looping forever. Asserts the turn returns a
       non-empty string that names the harness / repeated-and-blocked cause, that
       the total number of stub chat calls stays within
       ``_REPEAT_CALL_CAP + _BLOCKED_STREAK_CAP + 2``, that at least one
       fold-surviving steer row (user role, STEER_PREFIX / steer flag) was
       emitted this turn, and that ``turn_report["answer"]`` equals the returned
       give-up text.

    B. Reset — a real dispatch between blocked calls clears the streak, so a run
       that blocks twice, then dispatches a genuinely different call, then ends
       with a plain answer terminates NORMALLY (the model's own final text), never
       via the escalation give-up.

    C. Fold-survival — a message slice with no plain user row but a steer user-row
       ahead of assistant(tool_calls)+tool scaffolding, run through
       ``session._prune_messages``, keeps the steer row (it rides the user role,
       so everything from the steer onward is the in-flight turn) while the tool
       scaffolding *before* it is dropped. This is why the blocked-round steer
       survives a compaction fold when the block error behind it does not.

Exits 0 on success, 1 on any assertion failure or unexpected exception. Runs
with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing this
file); it chdir's into its own throwaway temp project dir (list_files resolves
against cwd) and restores the cwd afterward, and touches no repo files.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from evals._stub import _final_answer_response, _same_call_response, _StubClient


def check_escalation() -> list[str]:
    """A. Same blocked call forever -> harness force-finalizes the turn."""
    import agent
    from session import STEER_PREFIX, Session

    failures: list[str] = []
    cap = agent._REPEAT_CALL_CAP + agent._BLOCKED_STREAK_CAP + 2

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="loop-guard-escalation-")
    os.chdir(tmp)
    try:
        session = Session(tmp, "test-model", "You are a test agent.")
        # A single always-identical read-only call (list_files is PINNED and not
        # in _REPEAT_CAP_EXEMPT, so it is subject to the hard cap).
        client = _StubClient([_same_call_response("list_files", {"path": "."})])

        answer = agent.handle_user_message(
            "list the files", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
        )

        if not isinstance(answer, str) or not answer.strip():
            failures.append(f"turn did not return a non-empty string, got {answer!r}")
        else:
            low = answer.lower()
            if not ("harness" in low or "repeated" in low or "blocked" in low):
                failures.append(
                    f"give-up answer names neither harness nor repeated/blocked "
                    f"cause: {answer!r}"
                )

        if client.calls > cap:
            failures.append(
                f"stub chat called {client.calls} times, expected <= {cap}"
            )

        steer_rows = [
            m
            for m in session._messages
            if m.get("role") == "user"
            and (m.get("steer") or str(m.get("content", "")).startswith(STEER_PREFIX))
        ]
        if not steer_rows:
            failures.append("no fold-surviving steer row was emitted this turn")

        if session.turn_report.get("answer") != answer:
            failures.append(
                f"turn_report['answer']={session.turn_report.get('answer')!r} "
                f"does not equal the returned answer {answer!r}"
            )
    finally:
        os.chdir(original_cwd)

    return failures


def check_reset_no_premature_escalation() -> list[str]:
    """B. A real dispatch between blocks resets the streak -> normal termination."""
    import agent
    from session import Session

    failures: list[str] = []
    final_text = "Done — I listed the files and read the notes."

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="loop-guard-reset-")
    os.chdir(tmp)
    # A second valid list_files target so a genuinely DIFFERENT call can dispatch.
    (Path(tmp) / "sub").mkdir()
    try:
        session = Session(tmp, "test-model", "You are a test agent.")
        block_a = _same_call_response("list_files", {"path": "."})
        dispatch_b = _same_call_response("list_files", {"path": "sub"})
        # Blocks the streak up to 2 (below the escalation cap), then a different
        # call dispatches (resetting the streak to 0), then a plain final answer
        # ends the turn. Escalation must NOT fire.
        script = (
            [block_a] * (agent._REPEAT_CALL_CAP + 2)  # 3 dispatch + 2 blocked
            + [dispatch_b]                            # real dispatch -> streak = 0
            + [_final_answer_response(final_text)]
        )
        client = _StubClient(script)

        answer = agent.handle_user_message(
            "list the files", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
        )

        if answer != final_text:
            failures.append(
                f"expected the model's own final answer {final_text!r}, got "
                f"{answer!r} (escalation fired prematurely or the reset failed)"
            )
    finally:
        os.chdir(original_cwd)

    return failures


def check_fold_survival() -> list[str]:
    """C. A steer user-row survives _prune_messages; tool scaffolding before it drops."""
    from session import STEER_PREFIX, _prune_messages

    failures: list[str] = []

    steer_content = STEER_PREFIX + "The last tool call was blocked."
    slice_ = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "list_files", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "list_files",
         "content": "BLOCKED RESULT"},
        {"role": "user", "content": steer_content, "steer": True},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c2", "type": "function",
                 "function": {"name": "list_files", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "list_files",
         "content": "BLOCKED RESULT AGAIN"},
    ]
    pruned = _prune_messages(slice_)

    kept_steer = next((m for m in pruned if m.get("steer")), None)
    if kept_steer is None:
        failures.append("the steer row did not survive _prune_messages")
    elif str(kept_steer.get("content", "")) != steer_content:
        failures.append(
            f"the surviving steer content was altered: {kept_steer.get('content')!r}"
        )

    # The tool scaffolding that sat BEFORE the steer (completed prior turns) must
    # have been collapsed away — its result content must not appear.
    if any("BLOCKED RESULT" == str(m.get("content", "")) for m in pruned):
        failures.append("pre-steer tool scaffolding leaked past the prune")

    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("escalation", check_escalation),
        ("reset", check_reset_no_premature_escalation),
        ("fold-survival", check_fold_survival),
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
        "PASS: escalation force-finalizes a blocked loop; a dispatch resets the "
        "streak; the blocked-round steer survives _prune_messages"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
