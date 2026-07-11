"""In-turn steers must not collapse the turn's own scaffolding.

``_prune_messages`` shapes the sent view: it keeps the in-flight turn verbatim and
collapses every completed prior turn to ``[user, final answer]``. The in-flight
boundary is the LAST user-role row. But harness steers ride the ``user`` role
(``append_steer``, flagged ``steer: True``) — so a mid-turn steer used to move the
boundary forward, folding ALL of the turn's real work BEFORE the steer (the
assistant ``tool_calls`` rows, their tool results, observed outputs) into the
completed-slice collapse. The model lost the record of its own work mid-task and
re-read files it already read; the loop-guard then blocked the repeats and the
blocked steer wiped the scaffolding again — a spiral into force-finalize.

The fix anchors the boundary on the last REAL user row (``role == "user"`` without
the ``steer`` flag). Steers stay embedded in intact scaffolding, in order. This
script asserts three things:

a. End-to-end through the real ``agent.handle_user_message`` (stub LLM, no
   network): an edit with no prior verification run (create_file — the
   reproduce-before-edit steer fires and appends after the round), then a round
   that reads a file, then a final answer. After the turn ``assemble_context``
   still carries the FIRST round's assistant ``tool_calls`` row and its tool
   result (the steer did not collapse pre-steer scaffolding), and the repro steer
   row sits AFTER that scaffolding.

b. Direct ``_prune_messages`` drive: a list whose only in-turn steer sits amid
   real scaffolding keeps EVERY row (boundary anchored at the real user); and a
   two-turn list still collapses the completed turn to ``[user, final answer]``
   exactly as before, keeping the in-flight turn verbatim.

c. A post-compaction tail — NO real user, but a steer row plus tool scaffolding —
   fully collapses (scaffolding dropped) while the steer row rides through (the
   fold-survival property).

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file); scenario (a) chdir's into its own throwaway temp project dir
(create_file/read_file resolve against cwd) and restores the cwd afterward,
disables memory side effects for the run, and touches no repo files.
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


def _is_tool_calls_row(m: dict, tool_name: str) -> bool:
    """True when *m* is an assistant row carrying a tool_calls entry for *tool_name*."""
    if m.get("role") != "assistant":
        return False
    for tc in m.get("tool_calls") or []:
        if tc.get("function", {}).get("name") == tool_name:
            return True
    return False


def _is_tool_result_row(m: dict, tool_name: str) -> bool:
    """True when *m* is a tool-result row produced by *tool_name*."""
    return m.get("role") == "tool" and m.get("name") == tool_name


def _repro_steer_index(messages: list[dict]) -> int:
    """Index of the reproduce-before-edit steer row, or -1 if absent."""
    import agent
    from session import STEER_PREFIX

    for i, m in enumerate(messages):
        if (
            m.get("role") == "user"
            and m.get("steer")
            and str(m.get("content", "")).startswith(STEER_PREFIX)
            and agent._REPRO_BEFORE_EDIT_STEER in str(m.get("content", ""))
        ):
            return i
    return -1


def check_e2e_scaffolding_survives() -> list[str]:
    """a. A mid-turn steer leaves the pre-steer scaffolding intact in the sent view."""
    import agent
    import tools.registry as registry
    from evals._stub import _StubClient, disable_memory_hooks
    disable_memory_hooks()
    from session import Session

    # create_file is a catalog tool (read_file is PINNED); a live model activates
    # it via load_tool before use, so do the same for dispatch to run it.
    registry.activate("create_file")

    failures: list[str] = []

    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="steer-scaffolding-e2e-")
    os.chdir(tmp)
    try:
        session = Session(tmp, "test-model", "You are a test agent.")
        # Round 1 edits with no verification run this turn -> the repro steer
        # fires and appends after the round. Round 2 reads the file it created.
        # Then a plain final answer ends the turn.
        script = [
            _tool_call_response("create_file", {"path": "buggy_a.py", "content": "x = 1\n"}),
            _tool_call_response("read_file", {"path": "buggy_a.py"}),
            _final_answer_response("Edited and read the file."),
        ]
        client = _StubClient(script)

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.handle_user_message(
                "fix the wrong output", session, client, on_delta=None  # pyright: ignore[reportArgumentType]
            )

        ctx = session.assemble_context()
        # assemble_context (no summary) emits [system, <pruned messages...>].
        if not ctx or ctx[0].get("role") != "system":
            failures.append("assembled context did not open with a system message")
            return failures
        sent = ctx[1:]

        create_tc_idx = next(
            (i for i, m in enumerate(sent) if _is_tool_calls_row(m, "create_file")), -1
        )
        create_result_idx = next(
            (i for i, m in enumerate(sent) if _is_tool_result_row(m, "create_file")), -1
        )
        steer_idx = _repro_steer_index(sent)

        if create_tc_idx < 0:
            failures.append(
                "the first round's create_file assistant tool_calls row was "
                "collapsed out of the sent view (the steer moved the boundary)"
            )
        if create_result_idx < 0:
            failures.append(
                "the first round's create_file tool result was collapsed out of "
                "the sent view (the steer moved the boundary)"
            )
        if steer_idx < 0:
            failures.append("the reproduce-before-edit steer row is absent from the sent view")

        # The steer must sit AFTER the round-1 scaffolding it followed, not before
        # it or in its place.
        if create_tc_idx >= 0 and create_result_idx >= 0 and steer_idx >= 0:
            if not (create_tc_idx < create_result_idx < steer_idx):
                failures.append(
                    f"steer not positioned after the first round's scaffolding: "
                    f"create_tc={create_tc_idx}, create_result={create_result_idx}, "
                    f"steer={steer_idx}"
                )
    finally:
        os.chdir(original_cwd)

    return failures


def check_prune_keeps_in_turn_steer() -> list[str]:
    """b. In-turn steer keeps all rows; a completed turn still collapses as before."""
    from session import _prune_messages

    failures: list[str] = []

    # A single in-flight turn: a real user, then create_file scaffolding, a
    # mid-turn steer, and a second round of scaffolding. The boundary is the real
    # user at index 0, so every row must survive verbatim, in order.
    in_turn = [
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "create_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "create_file", "content": "wrote file"},
        {"role": "user", "content": "[harness] reproduce before editing on", "steer": True},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c2", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "read_file", "content": "x = 1"},
    ]
    pruned = _prune_messages(in_turn)
    if pruned != in_turn:
        failures.append(
            "an in-flight turn carrying a mid-turn steer lost rows: expected all "
            f"{len(in_turn)} rows verbatim, got {len(pruned)} "
            f"(roles {[m.get('role') for m in pruned]})"
        )

    # Two turns: a completed turn (with tool scaffolding and a final answer),
    # then a fresh in-flight turn. The completed turn must collapse to
    # [user, final answer]; the in-flight turn is kept verbatim.
    two_turn = [
        {"role": "user", "content": "first task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "d1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "d1", "name": "read_file", "content": "old content"},
        {"role": "assistant", "content": "First task done."},
        {"role": "user", "content": "second task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "d2", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "d2", "name": "read_file", "content": "new content"},
    ]
    expected = [
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "First task done."},
        {"role": "user", "content": "second task"},
        two_turn[5],
        two_turn[6],
    ]
    pruned2 = _prune_messages(two_turn)
    if pruned2 != expected:
        failures.append(
            "completed-turn collapse regressed: expected the first turn folded to "
            f"[user, final answer] with the in-flight turn verbatim, got "
            f"{[m.get('role') for m in pruned2]}"
        )

    return failures


def check_post_compaction_tail_folds_steer_survives() -> list[str]:
    """c. A tail with no real user collapses; the steer rides through."""
    from session import _prune_messages

    failures: list[str] = []

    # Post-compaction tail: the originating user request sits behind the
    # watermark (re-injected as the task anchor by assemble_context), so this
    # slice has NO real user — only a steer plus tool scaffolding. The whole
    # slice must collapse (scaffolding dropped) while the steer row survives.
    steer_row = {"role": "user", "content": "[harness] you were blocked", "steer": True}
    tail = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "e1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "e1", "name": "read_file", "content": "BIG FILE"},
        steer_row,
        {"role": "assistant", "content": "Progress so far."},
    ]
    pruned = _prune_messages(tail)

    if any(m.get("role") == "tool" for m in pruned):
        failures.append("a tool result survived the compaction boundary in a no-real-user tail")
    if any("tool_calls" in m for m in pruned):
        failures.append("a tool_calls row survived the compaction boundary in a no-real-user tail")
    if steer_row not in pruned:
        failures.append("the steer row did not survive the compaction fold (fold-survival broken)")
    if not any(
        m.get("role") == "assistant" and m.get("content") == "Progress so far." for m in pruned
    ):
        failures.append("the retained final assistant answer was dropped from the tail")

    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("e2e-scaffolding-survives", check_e2e_scaffolding_survives),
        ("prune-keeps-in-turn-steer", check_prune_keeps_in_turn_steer),
        ("post-compaction-tail", check_post_compaction_tail_folds_steer_survives),
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
        "PASS: a mid-turn steer never moves the in-flight boundary — pre-steer "
        "scaffolding survives verbatim, completed turns still collapse, and a "
        "post-compaction tail folds while the steer rides through"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
