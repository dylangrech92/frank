"""Dispatch-level check for post-compaction tail pruning (no LLM required).

After compaction folds into the in-flight turn, the originating user message
sits behind the watermark (re-injected verbatim as the task anchor), so the
assembled tail has no ``user`` message. Before the fix, ``_prune_messages``
returned such a tail unchanged — letting tool results and the ``tool_calls``
that spawned them survive the compaction boundary verbatim. That both hogged
tokens (tool results are the largest rows) and was inconsistent with the
cross-turn drop rule applied to completed prior turns, and is the single
biggest reason a post-compaction context could still exceed the cap.

Drives a real ``Session`` in a throwaway temp dir and asserts:

    Case A (the fix) — fold advanced past the last user; the assembled tail
        contains ZERO ``tool`` messages and ZERO ``tool_calls`` keys, while the
        final assistant text answer is retained.
    Case B (regression guard) — a NEW user message arrives after compaction;
        its in-flight tool exchange is kept verbatim (the carve-out still
        works once a user is present in the tail).

Exits 0 on success, 1 on any assertion failure or unexpected exception. Runs
with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file) and touches no repo files.
"""

from __future__ import annotations

import sys
import tempfile


def _tail(ctx):
    """Return the assembled messages after the system + summary blocks."""
    # assemble_context emits: [system, user(summary+anchor), <tail...>].
    return ctx[2:]


def _has_role(tail, role):
    return any(m.get("role") == role for m in tail)


def _has_tool_calls(tail):
    return any("tool_calls" in m for m in tail)


def _has_content(tail, needle):
    return any(needle in str(m.get("content", "")) for m in tail)


def main() -> int:
    from session import Session

    ok = True
    tmp = tempfile.mkdtemp(prefix="compaction_tail_prune_")

    # ---- Case A: fold advanced past the last user; tail has no user ----
    seed_a = [
        {"role": "user", "content": "original task: read and summarize"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file",
         "content": "BIG FILE CONTENT"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c2", "type": "function",
                 "function": {"name": "grep", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "grep",
         "content": "GREP RESULTS"},
        {"role": "assistant", "content": "Here is my summary so far."},
    ]
    sa = Session(tmp, "test-model", "you are a helpful agent", messages=list(seed_a))
    # Summary covers index 0 (the user); tail = seed_a[1:] has NO user.
    sa.set_summary("## Task\nread and summarize", 1)
    tail_a = _tail(sa.assemble_context())

    if _has_role(tail_a, "tool"):
        print("FAIL A: a tool message survived the compaction boundary", file=sys.stderr)
        for i, m in enumerate(tail_a):
            print(f"  tail_a[{i}] role={m.get('role')!r}", file=sys.stderr)
        ok = False
    if _has_tool_calls(tail_a):
        print("FAIL A: a tool_calls key survived the compaction boundary", file=sys.stderr)
        ok = False
    if not _has_content(tail_a, "Here is my summary so far."):
        print("FAIL A: the retained final assistant answer was dropped", file=sys.stderr)
        ok = False

    # ---- Case B: a new user arrives after compaction; in-flight survives ----
    seed_b = [
        {"role": "user", "content": "old task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file",
         "content": "old content"},
        {"role": "user", "content": "new task after compaction"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c2", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "read_file",
         "content": "new content"},
    ]
    # A fresh temp dir per Session: the transcript lock is keyed on a
    # timestamp+pid session id, so two Sessions in one dir in the same second
    # collide on the lock.
    tmp_b = tempfile.mkdtemp(prefix="compaction_tail_prune_b_")
    sb = Session(tmp_b, "test-model", "you are a helpful agent", messages=list(seed_b))
    # Summary covers indices 0..2 (past the old user, up to before the new user
    # at index 3); tail = seed_b[3:] contains the new in-flight turn.
    sb.set_summary("## Task\nold then new task", 3)
    tail_b = _tail(sb.assemble_context())

    if not _has_content(tail_b, "new task after compaction"):
        print("FAIL B: the new post-compaction user message was dropped", file=sys.stderr)
        ok = False
    if not any(m.get("tool_call_id") == "c2" for m in tail_b):
        print("FAIL B: the new turn's in-flight tool result did not survive verbatim", file=sys.stderr)
        ok = False
    if not _has_tool_calls(tail_b):
        print("FAIL B: the new turn's in-flight tool_calls did not survive verbatim", file=sys.stderr)
        ok = False
    # The folded (pre-watermark) tool result must NOT leak into the tail.
    if _has_content(tail_b, "old content"):
        print("FAIL B: a folded (behind-watermark) tool result leaked into the tail", file=sys.stderr)
        ok = False

    if ok:
        print("PASS: post-compaction tail has no tool scaffolding; new turn kept verbatim")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
