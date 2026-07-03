"""Token estimation and cap computation for reactive context compaction.

Pure helpers with no side effects: an estimator that mirrors what the provider
will actually count (assembled messages *and* tool schemas), and the single
shared cap definition used by both the pre-flight overflow check and the
compaction retry loop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from session import Session, _prune_messages  # noqa: E402


def _serialize_for_estimate(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
) -> str:
    """Flatten messages and tool schemas into one text blob for counting.

    Everything that occupies real context is included: each message's role and
    content, any native ``tool_calls`` (name + arguments), tool-result linkage
    fields, and the full JSON of every tool schema. Non-string content is
    JSON-encoded so structured payloads are counted, not skipped.
    """
    parts: List[str] = []

    for m in messages:
        parts.append(str(m.get("role", "")))
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content))
        name = m.get("name")
        if name:
            parts.append(str(name))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            parts.append(str(fn.get("name", "")))
            parts.append(str(fn.get("arguments", "")))

    if tools:
        for t in tools:
            parts.append(json.dumps(t))

    return "\n".join(parts)


def estimate_tokens(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
) -> int:
    """Estimate the token count of an assembled request.

    Uses ``tiktoken`` (``cl100k_base``) when it can be imported; otherwise falls
    back to the spec's chars/4 heuristic. Tool schemas are always counted — they
    occupy real context on every call.

    Args:
        messages: The assembled message list that would be sent to the provider.
        tools: Optional tool/function schema list included in the request.

    Returns:
        An integer token estimate (at least 1 for any non-empty input).
    """
    text = _serialize_for_estimate(messages, tools)
    if not text:
        return 0

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def compute_cap(window: int, compaction_cfg: Dict[str, Any] | None = None) -> int:
    """Return the usable token cap below the model's context window.

    The single shared definition::

        cap = window - max(reserve_ratio * window, reserve_min_tokens)

    ``reserve_ratio`` (default 0.10) and ``reserve_min_tokens`` (default 8000)
    are read from the ``compaction`` config block when present.

    Args:
        window: The model's full context window (``llm.context_limit``).
        compaction_cfg: The ``compaction`` config block, or None for defaults.

    Returns:
        The integer token cap; assembled requests should stay at or below it.
    """
    cfg = compaction_cfg or {}
    reserve_ratio = cfg.get("reserve_ratio", 0.10)
    reserve_min = cfg.get("reserve_min_tokens", 8000)
    reserve = max(reserve_ratio * window, reserve_min)
    return int(window - reserve)


# ---------------------------------------------------------------------------
# Reactive compaction: summarize older turns into a fixed six-section digest.
# ---------------------------------------------------------------------------


COMPACTION_SYSTEM_PROMPT = """You are compacting a long software-engineering conversation so it fits the model's context window. Produce a faithful, information-dense summary of everything so far, organized into EXACTLY these six sections, each under its exact markdown heading and in this order:

## Task
The user's overall goal(s) and any explicit requirements or constraints.

## State
What has been accomplished so far and the current state of the work.

## Files-touched
Every file created, edited, or examined, each with a one-line note on what changed or why it mattered. Preserve exact paths.

## Open
Unresolved problems, TODOs, failing tests, or questions still outstanding.

## Decisions
Key technical decisions taken and the reasoning behind them.

## Last
What was happening most recently, in enough detail that work can resume seamlessly.

Rules: keep exact file paths, function/class names, error messages, and concrete values. Drop pleasantries and repetition. Never invent anything that is not present in the conversation. Output only the six sections."""


def _render_messages_for_summary(messages):
    """Flatten messages into readable text for the summarizer's input."""
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        calls = m.get("tool_calls") or []
        if calls:
            names = ", ".join(
                (c.get("function", {}) or {}).get("name", "") for c in calls if isinstance(c, dict)
            )
            content = (content + f"  [tool_calls: {names}]").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def compact(session: "Session", client, window: int, compaction_cfg=None) -> bool:
    """Summarize the completed turns before the in-flight turn and set the watermark.

    Finds the in-flight turn (everything from the last ``user`` message onward),
    summarizes the pruned completed region before it (folding in any existing
    summary), and installs the result via ``session.set_summary``. The on-disk
    transcript is never touched.

    Args:
        session: The active Session.
        client: An LLMClient used to produce the summary.
        window: The model's context window (unused for now beyond signalling
            intent; kept for signature stability with the caller).
        compaction_cfg: The compaction config block (reserved for future tuning).

    Returns:
        True if a new summary was produced and installed; False if there was
        nothing new to summarize beyond the current watermark (caller should
        treat False as "cannot compact further").
    """
    msgs = session._messages

    cut = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            cut = i

    # Nothing new to fold in beyond what the watermark already covers.
    if cut <= session._summary_covers:
        return False

    region = _prune_messages(msgs[session._summary_covers:cut])
    material = _render_messages_for_summary(region)
    if session._summary:
        material = (
            "Previous summary of even earlier turns:\n"
            + session._summary
            + "\n\nConversation since that summary:\n"
            + material
        )

    request = [
        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
        {"role": "user", "content": material},
    ]
    response = client.chat(request, None)
    summary_text = response.text or ""

    session.set_summary(summary_text, cut)
    return True
