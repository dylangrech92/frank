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
       emitted this turn, that ``turn_report["answer"]`` equals the returned
       give-up text, and (E11) that the end-of-turn consolidation hook fired
       exactly once even on this force-finalized path — so the real work the
       turn did is mined into memory rather than silently dropped.

    A2. Truthful give-up envelope (E14) — the force-finalized answer is
       synthesized from ``turn_report`` (no extra LLM call), not a static string.
       A turn that mutated a file and ran a successful ``run_command`` yields an
       answer that names the file and the run, never says "unavailable", and
       stamps ``verified == True``; a mutation with no verification run keeps the
       cut-short caveat and ``verified == False``; and a turn that did no work
       stays close to the plain give-up with no "Work already applied" section.

    B. Reset — a real dispatch between blocked calls clears the streak, so a run
       that blocks twice, then dispatches a genuinely different call, then ends
       with a plain answer terminates NORMALLY (the model's own final text), never
       via the escalation give-up. Also asserts (E11) the consolidation hook
       fires exactly once on this normal finalize path — guarding against a
       double-enqueue after the hook moved out of ``_finalize_answer`` into the
       single choke point.

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
    original_consolidate = agent.consolidation_maybe_extract
    consolidate_calls: list[tuple] = []
    try:
        session = Session(tmp, "test-model", "You are a test agent.")
        # A single always-identical read-only call (list_files is PINNED and not
        # in _REPEAT_CAP_EXEMPT, so it is subject to the hard cap).
        client = _StubClient([_same_call_response("list_files", {"path": "."})])

        # E11 — record the end-of-turn consolidation hook at the module seam
        # (it is called unconditionally; MEMORY_ENABLED only gates its body), so
        # the check is independent of memory config and does not touch memory.
        # This is a force-finalized turn: it must still reach the hook exactly
        # once, proving the escalation give-up path routes through the choke
        # point rather than returning without consolidating the work it did.
        agent.consolidation_maybe_extract = (
            lambda *a, **kw: consolidate_calls.append((a, kw))
        )

        answer = agent.handle_user_message(
            "list the files", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
        )

        if len(consolidate_calls) != 1:
            failures.append(
                f"consolidation hook fired {len(consolidate_calls)} times on the "
                f"force-finalized turn, expected exactly 1"
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
        agent.consolidation_maybe_extract = original_consolidate
        os.chdir(original_cwd)

    return failures


def _drive_giveup(tmp_prefix: str, prefix_script: list):
    """Drive one force-finalized turn and return ``(answer, session)``.

    Runs *prefix_script* (real create_file/run_command calls that leave facts in
    turn_report), then spirals into an identical blocked call until the E8
    escalation give-up force-finalizes the turn. Consolidation is stubbed to a
    no-op so the check never touches memory. chdir's into a throwaway temp dir
    (create_file / list_files resolve against cwd) and restores cwd afterward.
    """
    import agent
    from session import Session
    from tools import registry

    # create_file / run_command are catalog tools (not PINNED), so dispatch would
    # reject them not-loaded — activate them exactly as a real session would after
    # a load_tool call, so the prefix genuinely mutates and verifies.
    registry.activate("create_file")
    registry.activate("run_command")

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix=tmp_prefix)
    os.chdir(tmp)
    original_consolidate = agent.consolidation_maybe_extract
    try:
        agent.consolidation_maybe_extract = lambda *a, **kw: None
        session = Session(tmp, "test-model", "You are a test agent.")
        # The trailing always-identical read-only call eventually blocks at the
        # hard cap and escalates; the stub reuses it for every surplus round.
        script = list(prefix_script) + [_same_call_response("list_files", {"path": "."})]
        client = _StubClient(script)
        answer = agent.handle_user_message(
            "do the multi-file work", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
        )
        return answer, session
    finally:
        agent.consolidation_maybe_extract = original_consolidate
        os.chdir(original_cwd)


def check_giveup_verified_work() -> list[str]:
    """E14 (a). Force-finalize after a mutation + successful run_command.

    The synthesized give-up answer must name the mutated file and the verification
    run, must NOT claim the report is 'unavailable', and turn_report['verified']
    must be True (real work landed and was verified before the stall).
    """
    failures: list[str] = []
    prefix = [
        _same_call_response("create_file", {"path": "feature.py", "content": "x = 1\n"}),
        _same_call_response("run_command", {"cmd": "python3 -c \"print('ok')\""}),
    ]
    answer, session = _drive_giveup("giveup-verified-", prefix)
    report = session.turn_report

    files = report["files_changed"]
    if not files:
        failures.append("no files_changed recorded despite a create_file mutation")
    else:
        path = files[0]["path"]
        if path not in answer:
            failures.append(
                f"give-up answer does not name the mutated file path {path!r}: {answer!r}"
            )
    if "work already applied" not in answer.lower():
        failures.append(f"give-up answer omits the work-applied marker: {answer!r}")
    if "unavailable" in answer.lower():
        failures.append(f"give-up answer still claims work is 'unavailable': {answer!r}")
    if "run_command" not in answer:
        failures.append(f"give-up answer omits the verification run: {answer!r}")
    if report["verified"] is not True:
        failures.append(
            f"turn_report['verified']={report['verified']!r}, expected True "
            f"(mutation + successful run_command)"
        )
    if answer != report.get("answer"):
        failures.append("turn_report['answer'] does not equal the returned answer")
    return failures


def check_giveup_unverified_work() -> list[str]:
    """E14 (b). Force-finalize after a mutation with NO verification run.

    The answer must still name the file and carry the cut-short caveat, but
    turn_report['verified'] must be False (nothing ran to verify the change).
    """
    failures: list[str] = []
    prefix = [
        _same_call_response("create_file", {"path": "widget.py", "content": "y = 2\n"}),
    ]
    answer, session = _drive_giveup("giveup-unverified-", prefix)
    report = session.turn_report

    files = report["files_changed"]
    if not files:
        failures.append("no files_changed recorded despite a create_file mutation")
    else:
        path = files[0]["path"]
        if path not in answer:
            failures.append(
                f"give-up answer does not name the mutated file path {path!r}: {answer!r}"
            )
    if "review the applied changes before retrying" not in answer.lower():
        failures.append(f"give-up answer omits the cut-short caveat: {answer!r}")
    if report["verified"] is not False:
        failures.append(
            f"turn_report['verified']={report['verified']!r}, expected False "
            f"(mutation with no verification run)"
        )
    return failures


def check_giveup_no_work() -> list[str]:
    """E14 (c). Force-finalize a turn that did no work at all.

    With no mutation and no run, the answer stays close to the plain give-up (no
    'Work already applied' section) and verified is False.
    """
    failures: list[str] = []
    answer, session = _drive_giveup("giveup-nowork-", [])
    report = session.turn_report

    if "work already applied" in answer.lower():
        failures.append(
            f"give-up answer claims work applied on a no-work turn: {answer!r}"
        )
    low = answer.lower()
    if not ("harness" in low or "repeated" in low or "blocked" in low):
        failures.append(f"give-up answer names no harness/blocked cause: {answer!r}")
    if report["verified"] is not False:
        failures.append(
            f"turn_report['verified']={report['verified']!r}, expected False "
            f"(no work this turn)"
        )
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
    original_consolidate = agent.consolidation_maybe_extract
    consolidate_calls: list[tuple] = []
    # A second valid list_files target so a genuinely DIFFERENT call can dispatch.
    (Path(tmp) / "sub").mkdir()
    try:
        # E11 — a normal (non-escalated) turn must fire the consolidation hook
        # exactly once too: the restructure removed the inner call from
        # _finalize_answer and moved it to the single choke point, so this guards
        # against a double-enqueue regression on the normal finalize path.
        agent.consolidation_maybe_extract = (
            lambda *a, **kw: consolidate_calls.append((a, kw))
        )
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

        if len(consolidate_calls) != 1:
            failures.append(
                f"consolidation hook fired {len(consolidate_calls)} times on the "
                f"normal finalize turn, expected exactly 1 (double-enqueue "
                f"regression?)"
            )
    finally:
        agent.consolidation_maybe_extract = original_consolidate
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
        ("giveup-verified-work", check_giveup_verified_work),
        ("giveup-unverified-work", check_giveup_unverified_work),
        ("giveup-no-work", check_giveup_no_work),
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
        "PASS: escalation force-finalizes a blocked loop with a truthful give-up "
        "envelope (files + runs + verified stamped from turn_report); a dispatch "
        "resets the streak; the blocked-round steer survives _prune_messages"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
