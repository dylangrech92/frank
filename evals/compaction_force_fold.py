"""Dispatch-level check for the overflow-ladder force_fold (no LLM required).

When ``compact`` cannot get the context under cap — typically a session with a
few very large messages where nothing is foldable at ``keep_recent``=8, so
``compact`` returns False on the first try — the turn used to dead-end in
"Context is over the model's token budget." The overflow ladder's
``force_fold`` advances the watermark past all but the most recent messages
with NO summarizer call, so the turn survives instead.

Drives a real ``Session`` (no LLM — ``force_fold`` makes no provider call) and
asserts:

    - force_fold advances the watermark so the assembled tail is just the last
      force_keep_recent messages, and a marker summary is seeded when none
      existed (so assemble_context's tail-slice engages).
    - an existing summary is preserved (only the watermark advances).
    - the kept tail never begins on an orphaned ``tool`` row (the boundary is
      walked back off one).
    - force_fold returns False when nothing remains to fold (too few messages).

Exits 0 on success, 1 on any failure. Repo root on ``sys.path``; touches no
repo files.
"""

from __future__ import annotations

import sys
import tempfile


def _tail(ctx):
    return ctx[2:]


def main() -> int:
    import compaction
    from session import Session

    ok = True

    # ---- Case 1: no summary, force_fold seeds the marker and keeps the tail ----
    seed1 = [
        {"role": "user", "content": "task one"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file",
         "content": "BIG CONTENT ONE"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "read_file",
         "content": "BIG CONTENT TWO"},
        {"role": "assistant", "content": "working answer"},
    ]
    tmp1 = tempfile.mkdtemp(prefix="force_fold_1_")
    s1 = Session(tmp1, "test-model", "sys", messages=list(seed1))
    advanced = compaction.force_fold(s1, {"force_keep_recent_messages": 2})

    if not advanced:
        print("FAIL 1: force_fold returned False with 6 messages present", file=sys.stderr)
        ok = False
    if not s1._summary:
        print("FAIL 1: no marker summary was seeded", file=sys.stderr)
        ok = False
    if compaction._FORCE_FOLD_MARKER not in (s1._summary or ""):
        print("FAIL 1: the marker summary was not the seeded text", file=sys.stderr)
        ok = False
    tail1 = _tail(s1.assemble_context())
    if any(m.get("role") == "tool" for m in tail1):
        # The kept tail after force_fold + S1 pruning has no tool scaffolding.
        print("FAIL 1: tool scaffolding survived in the force-fold tail", file=sys.stderr)
        ok = False
    if not any("working answer" in str(m.get("content", "")) for m in tail1):
        print("FAIL 1: the most recent assistant text was dropped", file=sys.stderr)
        ok = False

    # ---- Case 2: an existing summary is preserved; only the watermark advances ----
    tmp2 = tempfile.mkdtemp(prefix="force_fold_2_")
    s2 = Session(tmp2, "test-model", "sys", messages=list(seed1))
    s2.set_summary("## Task\npreviously summarized task", 2)
    compaction.force_fold(s2, {"force_keep_recent_messages": 2})
    if "previously summarized task" not in (s2._summary or ""):
        print("FAIL 2: force_fold clobbered an existing summary", file=sys.stderr)
        ok = False
    if s2._summary_covers <= 2:
        print(f"FAIL 2: watermark did not advance (covers={s2._summary_covers})", file=sys.stderr)
        ok = False

    # ---- Case 3: too few messages -> force_fold returns False (nothing to fold) ----
    tmp3 = tempfile.mkdtemp(prefix="force_fold_3_")
    s3 = Session(tmp3, "test-model", "sys", messages=[
        {"role": "user", "content": "only one"},
    ])
    if compaction.force_fold(s3, {"force_keep_recent_messages": 2}):
        print("FAIL 3: force_fold advanced with too few messages to fold", file=sys.stderr)
        ok = False

    # ---- Case 4: kept tail never begins on an orphaned tool row ----
    # msgs[len-2] is a tool result -> the boundary must walk back so the tail
    # does not start on a `tool` message whose originating assistant call was
    # folded. Assembled via force_fold then assemble_context (which re-prunes).
    seed4 = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file",
         "content": "X"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "grep", "content": "Y"},
        {"role": "assistant", "content": "final answer text"},
    ]
    tmp4 = tempfile.mkdtemp(prefix="force_fold_4_")
    s4 = Session(tmp4, "test-model", "sys", messages=list(seed4))
    compaction.force_fold(s4, {"force_keep_recent_messages": 2})
    # Watermark cut must not land such that _messages[cut] is a bare tool row.
    cut = s4._summary_covers
    if cut < len(seed4) and seed4[cut].get("role") == "tool":
        print(f"FAIL 4: fold boundary landed on a tool row (cut={cut})", file=sys.stderr)
        ok = False

    if ok:
        print("PASS: force_fold seeds marker / preserves summary / walks off tool / refuses when empty")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
