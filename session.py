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

import stats
from llm import ToolCall


# Module-level extension seam: add callables that accept a ``Session`` and
# return a non-empty string block.  Each provider renders context blocks that
# are appended to the system message during assemble_context().

CONTEXT_PROVIDERS: List[Callable[[Any], str]] = []


def _format_json(obj: Any) -> str:
    """Pretty-print *obj* as JSON with two-space indentation and trailing newline."""
    return json.dumps(obj, indent=2) + "\n"


def _parse_frontmatter(text: str) -> tuple[Dict[str, str], str]:
    """Split a persisted transcript file into its frontmatter dict and JSON body text.

    The file is expected to open with a ``---`` delimited block of ``key: value``
    lines followed by a second ``---`` line and then the JSON body. Files that do
    not start with the delimiter are treated as having no frontmatter at all.

    Args:
        text: Full contents of a transcript file.

    Returns:
        A tuple of (frontmatter dict, remaining body text).
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text

    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break

    if end is None:
        return {}, text

    frontmatter: Dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip()

    body_text = "\n".join(lines[end + 1:])
    return frontmatter, body_text


def _load_transcript(
    path: Path,
) -> tuple[List[Dict[str, Any]], str | None, int, datetime, int | None]:
    """Load a persisted transcript file for resuming a session.

    Supports both the original body schema (a bare JSON array of messages) and
    the additive schema (a JSON object with ``messages``, ``summary``,
    ``summary_covers`` and ``episodic_watermark`` keys), so old transcripts
    written before compaction state was persisted still load correctly.

    Args:
        path: Path to the ``<session_id>.json`` transcript file.

    Returns:
        A tuple of (messages, summary or None, summary_covers,
        created_at, episodic_watermark or None). ``episodic_watermark`` is
        ``None`` for legacy bare-array transcripts, signalling the caller
        should seed it to ``len(messages)`` rather than 0.

    Raises:
        ValueError: If the file's JSON body is malformed or is not shaped like
            a transcript (neither a bare list nor a dict).
    """
    text = path.read_text(encoding="utf-8")
    frontmatter, body_text = _parse_frontmatter(text)

    try:
        body = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"transcript corrupt: {path}: invalid JSON ({exc})") from exc

    episodic_watermark: int | None
    if isinstance(body, list):
        messages: List[Dict[str, Any]] = body
        summary: str | None = None
        summary_covers = 0
        episodic_watermark = None
    elif isinstance(body, dict) and isinstance(body.get("messages"), list):
        messages = body["messages"]
        summary = body.get("summary")
        summary_covers = body.get("summary_covers", 0)
        episodic_watermark = body.get("episodic_watermark", 0)
    else:
        raise ValueError(
            f"transcript corrupt: {path}: body is neither a message list nor a "
            f"dict with a 'messages' list"
        )

    created_at_raw = frontmatter.get("created_at")
    if created_at_raw:
        try:
            created_at = datetime.strptime(created_at_raw, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            created_at = datetime.now(timezone.utc)
    else:
        created_at = datetime.now(timezone.utc)

    return messages, summary, summary_covers, created_at, episodic_watermark


def list_sessions(project_root: str | Path) -> List[tuple[str, int | None]]:
    """List available session ids for *project_root*, newest first (by file mtime).

    Files whose body is neither a bare message list nor a dict with a
    ``messages`` list are skipped entirely — they are not valid transcripts.

    Args:
        project_root: Root directory of the project (str or Path).

    Returns:
        A list of (session_id, message_count) tuples. ``message_count`` is
        ``None`` when the transcript could not be parsed cheaply.
    """
    root = Path(project_root).resolve()
    session_dir = root / ".coding_agent" / "sessions"
    if not session_dir.is_dir():
        return []

    entries: List[tuple[str, float, int | None]] = []
    for p in session_dir.glob("*.json"):
        try:
            _, body_text = _parse_frontmatter(p.read_text(encoding="utf-8"))
            body = json.loads(body_text)
            if isinstance(body, list):
                count: int | None = len(body)
            elif isinstance(body, dict) and isinstance(body.get("messages"), list):
                count = len(body["messages"])
            else:
                continue
        except Exception:
            continue
        entries.append((p.stem, p.stat().st_mtime, count))

    entries.sort(key=lambda e: e[1], reverse=True)
    return [(sid, count) for sid, _mtime, count in entries]


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
            session's transcript (see ``_acquire_lock``).
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

        self._acquire_lock()

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

    def _acquire_lock(self) -> None:
        """Claim an advisory lock on this session's transcript.

        Writes ``<transcript>.lock`` containing this process's pid. If a lock
        file already exists and its pid is still alive, refuse to proceed —
        two processes must never append to the same transcript concurrently.
        A lock left behind by a dead process is treated as stale and replaced.

        Raises:
            RuntimeError: If another live process already holds the lock.
        """
        if self._lock_path.exists():
            try:
                existing_pid_text = self._lock_path.read_text(encoding="utf-8").strip()
                existing_pid = int(existing_pid_text)
            except (OSError, ValueError):
                existing_pid = None

            alive = False
            if existing_pid is not None:
                try:
                    os.kill(existing_pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    # Process exists but we can't signal it — treat as alive.
                    alive = True
                except OSError:
                    alive = False

            if alive:
                raise RuntimeError(
                    f"session '{self.session_id}' is already active in process "
                    f"{existing_pid} — resume it from that process or use a "
                    f"different --session id"
                )

        self._lock_path.write_text(str(os.getpid()), encoding="utf-8")

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
        try:
            if self._lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                self._lock_path.unlink()
        except OSError:
            pass
