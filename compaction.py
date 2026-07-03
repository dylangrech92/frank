"""Token estimation and cap computation for reactive context compaction.

Pure helpers with no side effects: an estimator that mirrors what the provider
will actually count (assembled messages *and* tool schemas), and the single
shared cap definition used by both the pre-flight overflow check and the
compaction retry loop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


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
