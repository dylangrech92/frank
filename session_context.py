"""Assembling the message list sent to the LLM: providers, folding, pruning."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

_HANDOVER_FRAME = (
    "You hit your context limit mid-task and are continuing the same task "
    "— this is not a new conversation. The summary below is the hand-over "
    "from your earlier work. Trust it: do not re-verify or repeat completed "
    "work; resume from the ## Last and ## Open sections.\n\n"
)


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
    * **Screenshot** (a ``user``-role row flagged ``screenshot`` — a vision
      attachment produced by a tool call mid-turn). Like a steer it is not a turn
      start, and for a sharper reason: treating one as the boundary would fold
      away the tool scaffolding of the very run that took the screenshot, so a
      verify run would lose its own evidence chain the moment it looked at
      anything.
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
        if m.get("role") == "user" and not m.get("steer") and not m.get("screenshot"):
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


def assemble_context(
    session: Any,
    providers: List[Callable[[Any], str]],
) -> List[Dict[str, str]]:
    """Return the list of message dicts to send to the LLM.

    The first element is a system message whose content is the session's ``system_prompt``
    followed by every non-empty block returned by registered context providers,
    each separated by a blank line.  All stored messages follow as pass-through.

    Returns:
        The assembled message list ready for the provider API.
    """
    blocks = [session.system_prompt]

    for provider in providers:
        block = provider(session)
        if block:
            blocks.append(block)

    system_content = "\n\n".join(blocks)

    result: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content}
    ]

    if session._summary:
        content = (
            _HANDOVER_FRAME
            + "Summary of the earlier conversation "
            "(older turns compacted to fit context):\n\n" + session._summary
        )
        # When compaction has folded the current turn's own user message into
        # the summary (its index is now behind the watermark), re-show it
        # verbatim so the model never loses the literal task it is working on
        # — the summary's paraphrase is a safety net, not a replacement.
        anchor = _folded_task_anchor(session._messages, session._summary_covers)
        if anchor is not None:
            content += (
                "\n\n---\n\nYour current task (original request, shown "
                "verbatim):\n\n" + anchor
            )
        result.append({"role": "user", "content": content})
        tail = session._messages[session._summary_covers:]
    else:
        tail = session._messages

    result.extend(_prune_images(session, _prune_messages(tail)))
    return result


def _prune_images(session: Any, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the most recent ``MAX_IMAGES`` screenshots as actual image data.

    Runs on the assembled view, after ``_prune_messages``, so it prunes what is
    actually being sent rather than what happens to be stored. Older screenshots
    collapse to a one-line placeholder naming the file — the model can still
    cite the artifact as evidence, it just no longer carries the pixels.

    ``session._image_refs`` indexes into ``session._messages``, but this list has
    been folded and pruned, so the two cannot be joined by position. They are
    matched by identity instead: the shared message dicts are the same objects,
    so ``id()`` says exactly which assembled row is which screenshot.

    The input list and its messages are never mutated — a pruned screenshot is
    emitted as a fresh text-only dict, leaving the session's own record intact.
    """
    refs = getattr(session, "_image_refs", None)
    if not refs:
        return messages

    # Imported here, not at module scope: session.py imports this module, so a
    # top-level import back into it would close the cycle at import time.
    from session import MAX_IMAGES, _PRUNED_IMAGE

    path_by_id = {id(session._messages[ref.index]): ref.path for ref in refs}
    present = [m for m in messages if id(m) in path_by_id]
    keep = {id(m) for m in present[-MAX_IMAGES:]} if MAX_IMAGES > 0 else set()

    out: List[Dict[str, Any]] = []
    for m in messages:
        key = id(m)
        if key not in path_by_id or key in keep:
            out.append(m)
            continue
        out.append({
            "role": m.get("role", "user"),
            "content": _PRUNED_IMAGE.format(path=path_by_id[key]),
            "screenshot": True,
        })
    return out


def _folded_task_anchor(
    messages: List[Dict[str, Any]],
    summary_covers: int,
) -> str | None:
    """Return the current task's user text when it sits behind the watermark.

    The "current task" is the most recent ``user`` message. When compaction
    has advanced *summary_covers* past it (its index < the watermark), it
    no longer appears in the assembled tail, so ``assemble_context`` re-shows
    it verbatim. Returns ``None`` when the latest user message is still
    visible in the tail (nothing to re-inject).
    """
    last_user = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user = i
    if last_user < 0 or last_user >= summary_covers:
        return None
    content = messages[last_user].get("content", "")
    return content if isinstance(content, str) else str(content)
