"""Message-list shaping helpers used when assembling assembled context."""

from __future__ import annotations

from typing import Any, Dict, List


def _prune_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse completed prior turns to ``[user, final answer]`` for the sent view.

    Three kinds of row drive the shaping:

    * **Real user message** (``role == "user"`` without the ``steer`` flag) — a
      genuine turn start. The LAST one marks the boundary: everything from it
      onward is the in-flight turn, returned unchanged (full native scaffolding:
      assistant ``tool_calls`` rows and their tool-result messages).
    * **Steer** (a ``user``-role row flagged ``steer`` — harness guidance
      injected *inside* a turn, never a turn start). It NEVER defines the
      in-flight boundary: treating a mid-turn steer as a turn start would fold
      all of that turn's own pre-steer scaffolding — the assistant's ``tool_calls``
      rows and their results — into the completed-slice collapse, so the model
      loses the record of its own work the moment a steer is appended. Steers
      stay embedded in intact scaffolding, in order.
    * **Assistant answer** (``assistant`` with content and no ``tool_calls``) —
      the final answer of a completed turn; the only assistant row kept from the
      completed slice.

    Every completed turn before the boundary keeps only its user messages (real
    user rows AND any steers — see below) and its final assistant answer;
    tool-result messages and mid-chain assistant messages that carried tool calls
    are dropped, and any ``tool_calls`` field is stripped from kept messages so no
    dangling tool-call ids remain.

    The completed slice deliberately keeps steer rows via the user branch so the
    fold-survival property holds: when the slice has NO real user message — the
    post-compaction tail, whose originating request sits behind the watermark and
    is re-injected as the task anchor by ``assemble_context`` — the whole slice
    collapses like completed prior turns (tool scaffolding does not survive the
    compaction boundary) while any steer rides through, preserving the
    harness-guidance channel across the fold.

    The input list is never mutated; kept messages are shallow-copied when a
    field must be stripped.
    """
    if not messages:
        return []

    # Boundary scan: the last REAL user row (steers are in-turn guidance, not
    # turn starts, so they never move the boundary). ``last_user < 0`` means the
    # slice has no real user at all — the post-compaction tail — so there is no
    # in-flight turn to carve out and the whole slice collapses. A user arriving
    # after compaction lands in this tail and is caught here, so its in-flight
    # turn is still carved out and kept verbatim.
    last_user = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user" and not m.get("steer"):
            last_user = i

    completed = messages if last_user < 0 else messages[:last_user]
    in_flight = [] if last_user < 0 else messages[last_user:]

    pruned: List[Dict[str, Any]] = []
    for m in completed:
        role = m.get("role")
        if role == "user":
            # Real user rows AND steers: both ride the user role, and keeping
            # steers is what lets harness guidance survive a compaction fold.
            pruned.append(m)
        elif role == "assistant" and not m.get("tool_calls") and m.get("content"):
            # Final answer of a completed turn — keep without any tool_calls key.
            if "tool_calls" in m:
                m = {k: v for k, v in m.items() if k != "tool_calls"}
            pruned.append(m)
        # else: tool results and mid-chain (tool-calling / empty) assistants dropped.

    return pruned + list(in_flight)
