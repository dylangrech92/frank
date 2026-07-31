"""On-disk conversation transcript storage for coding-agent sessions.

Serialises a list of OpenAI-format messages with YAML frontmatter to
``project_root/.coding_agent/sessions/<session_id>.json`` so every turn is
persisted without loss.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import session_lock
import stats
from llm import ToolCall

from session_context import _prune_messages
from session_store import _format_json, _load_transcript, _parse_frontmatter, list_sessions  # noqa: F401


# Module-level extension seam: add callables that accept a ``Session`` and
# return a non-empty string block.  Each provider renders context blocks that
# are appended to the system message during assemble_context().

CONTEXT_PROVIDERS: List[Callable[[Any], str]] = []

# Content prefix for harness steer messages (see ``Session.append_steer``).
# Spelled out rather than a bare tag: models — small local ones especially —
# treat anything on the user role as the human speaking unless the message
# itself says otherwise in plain words.
STEER_PREFIX = (
    "[automated message from the harness, NOT from the user — do not count "
    "this as a user request] "
)


class Session:
    """A conversation transcript stored with full fidelity on disk.

    A fresh launch creates a new file whose session id is unique even across
    concurrent same-second launches (timestamp + pid). ``Session.resume`` loads
    an existing transcript by session id and continues appending to it.

    Args:
        project_root: Root directory of the project (str or Path).
        model: Model identifier for this session.
        system_prompt: Base system prompt for the assistant.
        session_id: When resuming, the existing session id to reuse. Leave unset
            to mint a fresh id for a new session.
        messages: When resuming, the prior message list to seed the transcript with.
        summary: When resuming, a previously persisted compaction summary, if any.
        summary_covers: When resuming, how many leading messages ``summary`` covers.
        created_at: When resuming, the original creation timestamp to preserve.
        episodic_watermark: When resuming, the row count already handed to the
            episodic encoder, so a resumed session does not re-mine old history.

    Raises:
        RuntimeError: If another live process already holds the lock on this
            session's transcript (see ``session_lock.acquire``).
    """

    def __init__(
        self,
        project_root: str | Path,
        model: str,
        system_prompt: str,
        *,
        session_id: str | None = None,
        messages: List[Dict[str, Any]] | None = None,
        summary: str | None = None,
        summary_covers: int = 0,
        created_at: datetime | None = None,
        episodic_watermark: int = 0,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.model = model
        self.system_prompt = system_prompt
        self._messages: List[Dict[str, Any]] = list(messages) if messages is not None else []

        if session_id is not None:
            self.session_id = session_id
            self.created_at = created_at or datetime.now(timezone.utc)
        else:
            now_local = datetime.now()
            now_utc = now_local.astimezone(timezone.utc)
            self.created_at = now_utc
            self.session_id = (
                f"{now_local.strftime('%Y-%m-%dT%H:%M:%S').replace(':', '-')}-{os.getpid()}"
            )

        session_dir = self.project_root / ".coding_agent" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path: Path = session_dir / f"{self.session_id}.json"
        self._lock_path: Path = self.transcript_path.with_suffix(
            self.transcript_path.suffix + ".lock"
        )

        # Compaction watermark: when a summary is set, the first ``_summary_covers``
        # entries of ``_messages`` are replaced by ``_summary`` in the ASSEMBLED
        # view only. The on-disk transcript (_messages) always stays full-fidelity.
        self._summary: str | None = summary
        self._summary_covers: int = summary_covers

        # Episodic-extraction watermark: row count already handed to the encoder.
        # Persisted so a resumed session does not re-mine already-mined history.
        self.episodic_watermark: int = episodic_watermark

        # S2 — usage-driven compaction trigger state (in-memory only; not
        # persisted, since it is re-derived from the next real response).
        # ``last_prompt_tokens`` is the real ``prompt_tokens`` usage reported by
        # the most recent provider response, or None until one arrives.
        # ``last_prompt_context_len`` is the length of the assembled context
        # list that request was built from, so callers can estimate only what
        # has been appended since (that response's own text plus any new tool
        # results) instead of re-estimating the whole transcript. Reset to
        # None by the caller whenever compaction reshapes the assembled
        # context, since the recorded length no longer lines up.
        # ``token_estimate_ratio`` is an EMA (alpha 0.3) of observed
        # real/estimated ratios, seeded neutral at 1.0, that calibrates the
        # fallback chars/4-or-tiktoken estimator toward the provider's actual
        # tokenizer over the life of the session.
        self.last_prompt_tokens: int | None = None
        self.last_prompt_context_len: int = 0
        self.token_estimate_ratio: float = 1.0
        # Per-turn execution report (files changed, verification runs, gate
        # outcome, usage) written by the agent loop at the start of every
        # turn; consumed by one-shot ``--json`` mode to build the S4 result
        # envelope.
        self.turn_report: dict = {}

        # Usage-stats accumulators (stats.json). Seeded from any existing row so
        # a resumed session's totals continue rather than reset; run_time is the
        # prior wall-clock plus this process's elapsed time since construction.
        prior = stats.read_row(self.session_id)
        self._stats_run_base: float = float(prior.get("run_time", 0.0)) if prior else 0.0
        self._stats_tool_calls: int = int(prior.get("tool_calls", 0)) if prior else 0
        self._stats_input_tokens: int = int(prior.get("input", 0)) if prior else 0
        self._stats_output_tokens: int = int(prior.get("output", 0)) if prior else 0
        self._stats_run_started: float = time.monotonic()

        session_lock.acquire(self._lock_path, self.session_id)

    @classmethod
    def resume(
        cls, project_root: str | Path, model: str, system_prompt: str, session_id: str
    ) -> "Session":
        """Load an existing transcript by *session_id* and return a resumable Session.

        Any previously persisted compaction summary is restored so the assembled
        context picks up exactly where the prior run left off. Legacy transcripts
        that predate the episodic watermark field seed it to the full message
        count, so old history is never re-mined.

        Args:
            project_root: Root directory of the project (str or Path).
            model: Model identifier to use going forward (may differ from the
                model recorded in the transcript's frontmatter).
            system_prompt: Base system prompt for the assistant.
            session_id: The session id to resume (matches an existing
                ``<session_id>.json`` transcript file).

        Returns:
            A ``Session`` seeded with the prior transcript's messages and summary.

        Raises:
            FileNotFoundError: If no transcript exists for *session_id*, listing
                the available session ids for the project.
            ValueError: If the transcript file exists but its JSON body is
                corrupt or not shaped like a transcript.
            RuntimeError: If another live process already holds the lock on
                this session.
        """
        root = Path(project_root).resolve()
        session_dir = root / ".coding_agent" / "sessions"
        path = session_dir / f"{session_id}.json"

        if not path.exists():
            available = [sid for sid, _ in list_sessions(root)]
            available_str = ", ".join(available) if available else "(none found)"
            raise FileNotFoundError(
                f"session '{session_id}' not found at {path}. "
                f"Available sessions: {available_str}"
            )

        messages, summary, summary_covers, created_at, episodic_watermark = _load_transcript(path)
        if episodic_watermark is None:
            # Legacy transcript with no persisted watermark: skip re-mining old history.
            episodic_watermark = len(messages)

        return cls(
            root,
            model,
            system_prompt,
            session_id=session_id,
            messages=messages,
            summary=summary,
            summary_covers=summary_covers,
            created_at=created_at,
            episodic_watermark=episodic_watermark,
        )

    def append_user(self, text: str) -> None:
        """Append a user message and persist.

        Args:
            text: The user's message content.
        """
        self._messages.append({"role": "user", "content": text})
        self._persist()

    def append_steer(self, text: str) -> None:
        """Append a harness-guidance steer message and persist.

        A steer is harness-authored turn guidance (an empty-answer bounce, the
        H1 verification nudge) — NOT human input. It rides the ``user`` role for
        guaranteed OpenAI-compatibility (there is no mid-transcript system role,
        and a fabricated tool row would not follow a matching assistant
        tool_calls entry), but is marked two independent ways so nothing
        mistakes it for the user: the ``STEER_PREFIX`` content prefix tells the
        MODEL this is harness guidance, and the ``steer`` flag is the
        machine-readable signal read by the wire-stripping boundary (llm.py) and
        the compaction summarizer (compaction.py) so a steer is never sent as an
        unknown field nor quoted as the user's own words.

        Args:
            text: The steer guidance content (``STEER_PREFIX`` is added here —
                pass the message body only).
        """
        self._messages.append({"role": "user", "content": STEER_PREFIX + text, "steer": True})
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

    def set_summary(self, summary_text: str, covers_count: int) -> None:
        """Install a compaction summary covering the first *covers_count* messages.

        Affects the assembled view only — the on-disk transcript is never
        rewritten, so nothing is persisted here.

        Args:
            summary_text: The six-section summary of the covered messages.
            covers_count: Number of leading ``_messages`` entries the summary
                replaces in the assembled view.
        """
        self._summary = summary_text
        self._summary_covers = covers_count

    def set_episodic_watermark(self, count: int) -> None:
        """Persist the episodic-extraction watermark (row count already mined).

        Called by the episodic-extraction hook whenever it advances the
        watermark, so a resumed session picks up where mining left off instead
        of re-enqueueing already-mined history.

        Args:
            count: Total messages already handed to the episodic encoder.
        """
        self.episodic_watermark = count
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

        result: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

        if self._summary:
            content = (
                "Summary of the earlier conversation "
                "(older turns compacted to fit context):\n\n" + self._summary
            )
            # When compaction has folded the current turn's own user message into
            # the summary (its index is now behind the watermark), re-show it
            # verbatim so the model never loses the literal task it is working on
            # — the summary's paraphrase is a safety net, not a replacement.
            anchor = self._folded_task_anchor()
            if anchor is not None:
                content += (
                    "\n\n---\n\nYour current task (original request, shown "
                    "verbatim):\n\n" + anchor
                )
            result.append({"role": "user", "content": content})
            tail = self._messages[self._summary_covers:]
        else:
            tail = self._messages

        result.extend(_prune_messages(tail))
        return result

    # ------------------------------------------------------------------ private

    def _folded_task_anchor(self) -> str | None:
        """Return the current task's user text when it sits behind the watermark.

        The "current task" is the most recent ``user`` message. When compaction
        has advanced ``_summary_covers`` past it (its index < the watermark), it
        no longer appears in the assembled tail, so ``assemble_context`` re-shows
        it verbatim. Returns ``None`` when the latest user message is still
        visible in the tail (nothing to re-inject).
        """
        last_user = -1
        for i, m in enumerate(self._messages):
            if m.get("role") == "user":
                last_user = i
        if last_user < 0 or last_user >= self._summary_covers:
            return None
        content = self._messages[last_user].get("content", "")
        return content if isinstance(content, str) else str(content)

    def _persist(self) -> None:
        """Write YAML frontmatter followed by the JSON messages array to disk.

        Writes to a pid-suffixed sibling temp file first, then atomically
        renames it into place via ``os.replace`` so a resuming reader never
        observes a partially-written transcript (the file is now load-bearing
        for ``Session.resume``, not just an append-only log).
        """
        lines = [
            "---",
            f"session_id: {self.session_id}",
            f"cwd: {str(self.project_root)}",
            f"model: {self.model}",
            f"created_at: {self.created_at.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "---",
        ]

        body_obj: Dict[str, Any] = {"messages": self._messages}
        if self._summary:
            body_obj["summary"] = self._summary
            body_obj["summary_covers"] = self._summary_covers
        if self.episodic_watermark:
            body_obj["episodic_watermark"] = self.episodic_watermark

        body = _format_json(body_obj)
        file_contents = "\n".join(lines) + "\n" + body

        tmp_path = self.transcript_path.with_suffix(
            self.transcript_path.suffix + f".{os.getpid()}.tmp"
        )
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(file_contents)
        os.replace(tmp_path, self.transcript_path)

    def record_llm_call(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        tool_calls: int,
    ) -> None:
        """Fold one LLM call's usage into the session totals and upsert stats.json.

        Called after every provider response (main agent loop and the compaction
        summary call). Token counts are ``None`` on providers that don't report
        usage; those contribute zero rather than corrupting the running total.
        ``run_time`` is the resumed baseline plus this process's elapsed seconds.
        """
        self._stats_tool_calls += tool_calls
        self._stats_input_tokens += prompt_tokens or 0
        self._stats_output_tokens += completion_tokens or 0
        run_time = self._stats_run_base + (time.monotonic() - self._stats_run_started)
        stats.upsert(
            self.session_id,
            round(run_time, 3),
            self._stats_tool_calls,
            self._stats_input_tokens,
            self._stats_output_tokens,
        )

    def close(self) -> None:
        """Release this session's advisory lock file, if this process still owns it.

        Safe to call multiple times and safe to call even if the lock was
        never successfully acquired (e.g. constructor raised before writing
        it) — both cases are silently no-ops.
        """
        session_lock.release(self._lock_path)
