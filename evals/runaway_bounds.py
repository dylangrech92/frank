"""Runaway-loop bounds, end-to-end through the real turn loop.

``loop_guard_escalation.py`` covers the ladder that ends a turn *between* rounds.
This covers the three ways a turn ran away regardless, each reproduced from a
shape measured in this harness's own session logs and each driven through the
real ``agent.handle_user_message`` with a stub LLM client (no network):

    A. **Bounded round** — one assistant message carrying a burst of identical
       calls. Measured live at 2,555 calls in a single message, 2,500 of them
       byte-identical: the hard cap refused them one at a time (2,497 blocked
       rows) but the round walked every single call, because the escalation
       ladder is only consulted once the round returns. Asserts the round stops
       dispatching at ``_BLOCKED_STREAK_CAP``, that the turn force-finalizes, and
       that the transcript stays wire-legal — one tool result row per call id,
       including for the calls that were never dispatched.

    B. **Parallel batch hole** — the repeat cap used to live inline on the
       sequential path only, so an all-``parallel_safe`` batch skipped it
       entirely and then cleared ``blocked_streak`` as proof of progress.
       Measured live at 102 byte-identical ``read_file`` calls in one turn
       against a cap of 3. Asserts a batch holding a capped call is blocked and
       the turn terminates.

    C. **Narration runaway** — a model that re-emits identical prose while
       varying its calls enough to stay under the identical-call tally. Measured
       in three sessions (3x, 4x and 6x identical narration in one turn, every
       one carrying tool calls). Asserts the turn force-finalizes, and — the
       other half of the contract — that a *terminal* answer echoing earlier
       narration is left alone, since an ending is not a loop.

Exits 0 on success, 1 on any assertion failure or unexpected exception. Runs
with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing this
file); each check chdir's into its own throwaway temp project dir and restores
the cwd afterward, and touches no repo files.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from evals._stub import (
    _burst_call_response,
    _final_answer_response,
    _same_call_response,
    _StubClient,
    disable_memory_hooks,
)

BURST = 500


def _tool_rows(session) -> list[dict]:
    return [m for m in session._messages if m.get("role") == "tool"]


def _count_rows(session, marker: str) -> int:
    return sum(1 for m in _tool_rows(session) if marker in str(m.get("content", "")))


def _protocol_failures(session) -> list[str]:
    """Every tool_call id in an assistant row must have exactly one tool row.

    The OpenAI wire protocol rejects a request whose assistant ``tool_calls``
    entry has no matching tool message, and sessions are resumable — so a round
    that stops early without emitting results for the calls it skipped leaves a
    transcript that can never be sent again.
    """
    failures: list[str] = []
    answered: dict[str, int] = {}
    for m in _tool_rows(session):
        cid = str(m.get("tool_call_id", ""))
        answered[cid] = answered.get(cid, 0) + 1
    for m in session._messages:
        for tc in m.get("tool_calls") or []:
            cid = str(tc.get("id", ""))
            n = answered.get(cid, 0)
            if n != 1:
                failures.append(
                    f"tool_call id {cid!r} has {n} matching tool result rows, expected 1"
                )
    return failures


def _drive(prefix: str, script: list, max_calls: int = 30, mode: str = "research"):
    """Run one turn against a stub client in a throwaway project dir.

    Returns ``(answer, session, client)``. Consolidation and orientation are
    disabled so the check stays offline and touches no memory.
    """
    disable_memory_hooks()
    import agent
    from session import Session
    from tools import registry

    saved_mode = registry.current_mode()
    registry.activate_mode(mode)
    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix=prefix)
    (Path(tmp) / "sub").mkdir()
    os.chdir(tmp)
    original_consolidate = agent.consolidation_maybe_extract
    try:
        agent.consolidation_maybe_extract = lambda *a, **kw: None
        session = Session(tmp, "test-model", "You are a test agent.")
        client = _StubClient(script, max_calls=max_calls)
        answer = agent.handle_user_message(
            "do the work", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
        )
        return answer, session, client
    finally:
        agent.consolidation_maybe_extract = original_consolidate
        os.chdir(original_cwd)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)


def check_bounded_round() -> list[str]:
    """A. A burst of identical calls in ONE message stops at the streak cap."""
    import agent

    failures: list[str] = []
    script = [_burst_call_response("list_files", {"path": "."}, BURST)]
    answer, session, client = _drive("runaway-burst-", script)

    blocked = _count_rows(session, "loop-guard-blocked")
    abandoned = _count_rows(session, "round-abandoned")
    rows = len(_tool_rows(session))
    # Counted by elimination: an identical repeat that dispatches is rendered as
    # a dedup stub, not a success envelope, so there is no one marker to match.
    dispatched = rows - blocked - abandoned

    if dispatched != agent._REPEAT_CALL_CAP:
        failures.append(
            f"{dispatched} calls dispatched, expected exactly "
            f"{agent._REPEAT_CALL_CAP} (the hard cap)"
        )
    if blocked != agent._BLOCKED_STREAK_CAP:
        failures.append(
            f"{blocked} calls blocked, expected exactly "
            f"{agent._BLOCKED_STREAK_CAP} (the round must stop at the streak cap)"
        )
    expected_abandoned = BURST - agent._REPEAT_CALL_CAP - agent._BLOCKED_STREAK_CAP
    if abandoned != expected_abandoned:
        failures.append(
            f"{abandoned} calls abandoned undispatched, expected "
            f"{expected_abandoned} — the round walked the whole burst instead of "
            f"stopping at the streak cap"
        )
    if rows != BURST:
        failures.append(f"{rows} tool result rows for {BURST} calls, expected {BURST}")
    failures.extend(_protocol_failures(session))

    if client.calls != 1:
        failures.append(
            f"stub chat called {client.calls} times, expected 1 (the burst is one round)"
        )
    low = (answer or "").lower()
    if not ("harness" in low and ("blocked" in low or "repeated" in low)):
        failures.append(f"turn did not force-finalize with a give-up answer: {answer!r}")
    if session.turn_report.get("answer") != answer:
        failures.append("turn_report['answer'] does not equal the returned answer")
    return failures


def check_parallel_batch_capped() -> list[str]:
    """B. An all-parallel_safe batch is subject to the same repeat cap."""
    import agent

    failures: list[str] = []
    from evals._stub import _next_call_id
    from llm import ChatResponse, ToolCall

    def batch():
        """One round: two identical-per-round parallel_safe calls."""
        return ChatResponse(
            text="",
            tool_calls=[
                ToolCall(id=_next_call_id("p"), name="list_files", arguments={"path": "."}),
                ToolCall(id=_next_call_id("p"), name="list_files", arguments={"path": "sub"}),
            ],
        )

    answer, session, client = _drive("runaway-parallel-", [batch], max_calls=12)

    blocked = _count_rows(session, "loop-guard-blocked")
    if blocked < 1:
        failures.append(
            "no call in the parallel batch was ever blocked — the repeat cap is "
            "not enforced on the concurrent dispatch path"
        )
    if client.calls > agent._REPEAT_CALL_CAP + agent._BLOCKED_STREAK_CAP + 2:
        failures.append(
            f"stub chat called {client.calls} times — the batch loop did not terminate"
        )
    low = (answer or "").lower()
    if not ("harness" in low and ("blocked" in low or "repeated" in low)):
        failures.append(f"turn did not force-finalize with a give-up answer: {answer!r}")
    failures.extend(_protocol_failures(session))
    return failures


def check_narration_runaway() -> list[str]:
    """C1. Identical prose on tool-bearing rounds force-finalizes the turn."""
    from turn.guards import _TEXT_RUNAWAY_CAP

    failures: list[str] = []
    narration = "Let me re-check the project layout before continuing."
    # Every round calls a DIFFERENT path, so the identical-call tally never
    # trips and only the repeated narration can end this turn.
    script = [
        _same_call_response("list_files", {"path": f"d{i}"}, text=narration)
        for i in range(12)
    ]
    answer, session, client = _drive("runaway-narration-", script, max_calls=12)

    if client.calls != _TEXT_RUNAWAY_CAP:
        failures.append(
            f"stub chat called {client.calls} times, expected {_TEXT_RUNAWAY_CAP} — "
            f"the narration loop was not caught at the cap"
        )
    low = (answer or "").lower()
    if not ("harness" in low and "same message" in low):
        failures.append(f"turn did not force-finalize on the narration loop: {answer!r}")
    if session.turn_report.get("answer") != answer:
        failures.append("turn_report['answer'] does not equal the returned answer")
    # The looping round's assistant row must never reach the transcript: it
    # carries tool_calls that were never dispatched, so leaving it would break a
    # resumed session's next request.
    failures.extend(_protocol_failures(session))
    tool_rounds = sum(
        1 for m in session._messages if m.get("role") == "assistant" and m.get("tool_calls")
    )
    if tool_rounds != _TEXT_RUNAWAY_CAP - 1:
        failures.append(
            f"{tool_rounds} assistant tool_calls rows in the transcript, expected "
            f"{_TEXT_RUNAWAY_CAP - 1} (the looping round must be dropped, not stored)"
        )
    return failures


def check_terminal_answer_not_killed() -> list[str]:
    """C2. A final answer echoing earlier narration ends the turn normally.

    The other half of the narration contract: the guard counts tool-BEARING
    rounds only. A model that narrates a plan twice and then repeats that same
    sentence as its final answer has finished, not looped, and must get its own
    text back rather than a give-up.
    """
    failures: list[str] = []
    narration = "I checked the layout and everything is where it should be."
    script = [
        _same_call_response("list_files", {"path": "d1"}, text=narration),
        _same_call_response("list_files", {"path": "d2"}, text=narration),
        _final_answer_response(narration),
    ]
    answer, session, client = _drive("runaway-terminal-", script, max_calls=8)

    if answer != narration:
        failures.append(
            f"expected the model's own final answer {narration!r}, got {answer!r} — "
            f"the narration guard fired on the terminal round"
        )
    if client.calls != 3:
        failures.append(f"stub chat called {client.calls} times, expected 3")
    failures.extend(_protocol_failures(session))
    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("bounded-round", check_bounded_round),
        ("parallel-batch-capped", check_parallel_batch_capped),
        ("narration-runaway", check_narration_runaway),
        ("terminal-answer-spared", check_terminal_answer_not_killed),
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
        "PASS: a burst round stops at the streak cap with a wire-legal transcript, "
        "the repeat cap applies to parallel batches, a narration loop force-"
        "finalizes the turn, and a terminal answer echoing earlier narration does not"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
