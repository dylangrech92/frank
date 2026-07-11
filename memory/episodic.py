"""Shared LLM-JSON-array parsing helper.

The episodic memory layer this module used to hold (turn-end gist encoder,
salience/novelty scoring, clock-based erosion + eviction, and a recall/forget
layer over an ``episodes`` table) was retired in M7 -- "what happened this session," eroded by a clock, is conversational
continuity, not codebase knowledge. ``memory.consolidation`` replaced it --
mining the accepted turn's diff + transcript tail directly into anchored
knowledge atoms, off the hot path.

What remains is ``_safe_json_array``, a tolerant parser for an LLM's JSON-array
response (reasoning-model think-blocks, markdown fences, truncated output).
It has nothing to do with episodes specifically -- ``memory.consolidation``
imports it verbatim to parse its own ADD/UPDATE/DELETE/NOOP/DECISION/PIVOT
op list, exactly as the old gist encoder and fact-extractor did.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Safety-net JSON parser for LLM output
# ---------------------------------------------------------------------------


def _repair_json(text: str) -> str | None:
    """Repair a truncated JSON array/object by closing brackets opened outside of
    strings (after dropping a dangling trailing comma). Returns None when nothing
    needs closing."""
    stack: list[str] = []
    in_string = False
    escape = False
    backslash = chr(92)
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == backslash:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "[{":
            stack.append(ch)
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
    if not stack:
        return None
    repaired = text.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1].rstrip()
    closers = {"[": "]", "{": "}"}
    for opener in reversed(stack):
        repaired += closers[opener]
    return repaired


def _safe_json_array(raw: str) -> list:
    """Best-effort parse of an LLM JSON array response.

    Tolerant of reasoning-model output: drops a leading think block, strips prose
    and markdown fences around the JSON, isolates the first array/object, and
    repairs a truncated (unclosed) array before parsing.
    """
    text = raw.strip()

    # Drop a reasoning-model think block: keep only what follows the closing tag.
    close_tag = chr(60)+"/think"+chr(62)
    if close_tag in text:
        text = text.rsplit(close_tag, 1)[-1].strip()

    # Strip ```json / ``` fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        newline = chr(10)
        for i in range(1, len(lines)):
            if lines[i].strip().startswith("```"):
                text = newline.join(lines[1:i])
                break
        else:
            text = newline.join(lines[1:])
        text = text.strip()

    # Isolate the JSON payload from the first '[' or '{' onward (also drops any
    # unclosed think block or prose preamble).
    start = -1
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start == -1:
        return []
    text = text[start:].strip()

    # Try a strict parse first, then a bracket-repaired variant.
    for candidate in (text, _repair_json(text)):
        if candidate is None:
            continue
        try:
            result = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return result
        return []
    return []
