from __future__ import annotations

import json

import compaction

from tools.registry import get_tool
from tools.result import ToolResult


def render_tool_result(name: str, result: ToolResult) -> str:
    """Render a ``ToolResult`` value as plain text for a tool message content slot.

    The rendered string follows this shape::

        [name(STATUS code=CODE)]       # error status with code
        [name(STATUS)]                  # success status, no code
        <body>                          # only when body is non-empty
        meta_key: value                 # one line per non-empty meta entry
        hint: <hint_text>               # only when a hint is present

    The bracket header always contains the tool name and status.  When status
    is ``'error'`` and ``result.code`` is set, the kebab-case code is appended
    inside the brackets as ``code=THECODE``.

    Args:
        name: Registered tool name (used in the header).
        result: A ``ToolResult`` instance to render.

    Returns:
        A plain-text string suitable for ``content`` on a ``tool`` role message.
    """
    lines: list[str] = []

    if result.status == "error" and result.code is not None:
        header = f"[{name}({result.status} code={result.code})]"
    else:
        header = f"[{name}({result.status})]"

    lines.append(header)

    body = result.body
    if body:
        if isinstance(body, (dict, list)):
            body_str_val = json.dumps(result.body)
        else:
            body_str_val = str(result.body)
        lines.append(body_str_val)

    for meta_key, meta_value in result.meta.items():
        if meta_value is None or meta_value == "":
            continue
        lines.append(f"{meta_key}: {meta_value}")

    hint_text = result.hint
    if hint_text is not None and hint_text:
        lines.append(f"hint: {hint_text}")

    return "\n".join(lines)


def _guard_oversize_result(
    name: str,
    result: ToolResult,
    rendered: str,
    cap: int,
    est_context_before_result: int,
) -> tuple[ToolResult, str]:
    """Discard an oversized rendered tool result and substitute a clean error.

    The design spec: never silently truncate a tool result — if it would
    consume more than 75% of the remaining token budget, discard the body
    entirely and return a ``result-too-large`` error instead, so the oversized
    raw body never enters the session transcript.

    Args:
        name: Registered tool name.
        result: The ``ToolResult`` returned by ``dispatch()``.
        rendered: The already-rendered text for *result* (see ``render_tool_result``).
        cap: The shared token cap for this turn's context (see ``compaction.compute_cap``).
        est_context_before_result: Token estimate of the assembled context before
            this result is appended.

    Returns:
        ``(result, rendered)`` unchanged when the result fits, or a substituted
        ``(ToolResult.err(...), rendered_error_text)`` pair when it does not.
    """
    result_tokens = compaction.estimate_tokens([{"role": "tool", "content": rendered}])
    budget = 0.75 * max(cap - est_context_before_result, 1)
    if result_tokens <= budget:
        return result, rendered

    tool = get_tool(name)
    action = getattr(tool, "action", "complete the operation") if tool else "complete the operation"
    oversize_hint = (
        getattr(tool, "oversize_hint", "narrow the request or use a more specific tool")
        if tool
        else "narrow the request or use a more specific tool"
    )
    substituted = ToolResult.err(
        f"{name} failed to {action} — result is too large to fit in context — {oversize_hint}",
        code="result-too-large",
    )
    return substituted, render_tool_result(name, substituted)
