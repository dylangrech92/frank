"""On-disk conversation transcript storage for coding-agent sessions.

Serialises a list of OpenAI-format messages with YAML frontmatter to
``project_root/.coding_agent/sessions/<session_id>.json`` so every turn is
persisted without loss.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

from llm import ToolCall


# Module-level extension seam: add callables that accept a ``Session`` and
# return a non-empty string block.  Each provider renders context blocks that
# are appended to the system message during assemble_context().

CONTEXT_PROVIDERS: List[Callable[[Any], str]] = []


def _format_json(obj: Any) -> str:
    """Pretty-print *obj* as JSON with two-space indentation and trailing newline."""
    return json.dumps(obj, indent=2) + "\n"


def _prune_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse completed prior turns to ``[user, final answer]`` for the sent view.

    The in-flight turn — everything from the last ``user`` message onward — is
    returned unchanged (full native scaffolding: assistant tool_calls and their
    tool-result messages). Every completed turn before it keeps only its user
    message and its final assistant answer (the assistant message with no tool
    calls); tool-result messages and mid-chain assistant messages that carried
    tool calls are dropped, and any ``tool_calls`` field is stripped from kept
    messages so no dangling tool-call ids remain.

    The input list is never mutated; kept messages are shallow-copied when a
    field must be stripped.
    """
    if not messages:
        return []

    last_user = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user = i

    # No user message yet: treat the whole thing as in-flight.
    if last_user < 0:
        return list(messages)

    completed = messages[:last_user]
    in_flight = messages[last_user:]

    pruned: List[Dict[str, Any]] = []
    for m in completed:
        role = m.get("role")
        if role == "user":
            pruned.append(m)
        elif role == "assistant" and not m.get("tool_calls") and m.get("content"):
            # Final answer of a completed turn — keep without any tool_calls key.
            if "tool_calls" in m:
                m = {k: v for k, v in m.items() if k != "tool_calls"}
            pruned.append(m)
        # else: tool results and mid-chain (tool-calling / empty) assistants dropped.

    return pruned + list(in_flight)


class Session:
    """A conversation transcript stored with full fidelity on disk.

    Every launch creates a fresh file; there is no resume of prior transcripts.

    Args:
        project_root: Root directory of the project (str or Path).
        model: Model identifier for this session.
        system_prompt: Base system prompt for the assistant.
    """

    def __init__(self, project_root: str | Path, model: str, system_prompt: str) -> None:
        self.project_root = Path(project_root).resolve()
        self.model = model
        self.system_prompt = system_prompt
        self._messages: List[Dict[str, Any]] = []

        now_local = datetime.now()
        now_utc = now_local.astimezone(timezone.utc)
        self.created_at = now_utc
        self.session_id = now_local.strftime("%Y-%m-%dT%H:%M:%S").replace(":", "-")

        session_dir = self.project_root / ".coding_agent" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path: Path = session_dir / f"{self.session_id}.json"

    def append_user(self, text: str) -> None:
        """Append a user message and persist.

        Args:
            text: The user's message content.
        """
        self._messages.append({"role": "user", "content": text})
        self._persist()

    def append_assistant(self, text: str, tool_calls: List[ToolCall] | None = None) -> None:
        """Append an assistant message and persist.

        Args:
            text: The assistant's textual content.
            tool_calls: Optional list of parsed ``ToolCall`` objects. When non-empty
                the native OpenAI tool_calls key is added to the message dict.
        """
        entry: Dict[str, Any] = {"role": "assistant", "content": text}

        if tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ]

        self._messages.append(entry)
        self._persist()

    def append_tool_result(self, call_id: str, name: str, content: str) -> None:
        """Append a tool result message and persist.

        Args:
            call_id: The tool call ID to link this result to.
            name: The function/tool name that produced the result.
            content: The textual result content.
        """
        self._messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        })
        self._persist()

    def amend_last_tool_result(self, extra_text: str) -> None:
        """Append *extra_text* to the last tool message content if one exists.

        Only modifies the transcript when the final message in ``_messages`` has
        role ``"tool"``.  No-op otherwise.

        Args:
            extra_text: Text to append (prefixed by a newline).
        """
        if not self._messages or self._messages[-1]["role"] != "tool":
            return

        existing = self._messages[-1]["content"]
        self._messages[-1]["content"] = existing + "\n" + extra_text
        self._persist()

    def assemble_context(self) -> List[Dict[str, str]]:
        """Return the list of message dicts to send to the LLM.

        The first element is a system message whose content is ``system_prompt``
        followed by every non-empty block returned by registered context providers,
        each separated by a blank line.  All stored messages follow as pass-through.

        Returns:
            The assembled message list ready for the provider API.
        """
        blocks = [self.system_prompt]

        for provider in CONTEXT_PROVIDERS:
            block = provider(self)
            if block:
                blocks.append(block)

        system_content = "\n\n".join(blocks)

        return [
            {"role": "system", "content": system_content},
            *_prune_messages(self._messages),
        ]

    # ------------------------------------------------------------------ private

    def _persist(self) -> None:
        """Write YAML frontmatter followed by the JSON messages array to disk."""
        lines = [
            "---",
            f"session_id: {self.session_id}",
            f"cwd: {str(self.project_root)}",
            f"model: {self.model}",
            f"created_at: {self.created_at.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "---",
        ]

        body = _format_json(self._messages)
        file_contents = "\n".join(lines) + "\n" + body

        with open(self.transcript_path, "w", encoding="utf-8") as f:
            f.write(file_contents)
