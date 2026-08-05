"""The live state of one conversation: messages, summary, usage, lock.

``Session`` owns what the conversation currently is — the message list, the
compaction summary and the watermarks over it, the per-session usage totals —
and grows it a turn at a time, persisting after every append. The mechanics sit
in three siblings: ``session_store`` reads and writes the transcript file,
``session_context`` shapes the message list sent to the model, and
``session_lock`` keeps a single writer per transcript.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import session_context
import session_lock
import stats
from llm import ToolCall

from session_store import _load_transcript, _write_transcript, list_sessions


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

# How many of the most recent screenshots keep their actual image data in the
# assembled context. Vision payloads are the single heaviest thing a turn can
# put on the wire (a base64 PNG dwarfs any tool body), and a verify run takes
# many shots — so older ones degrade to a text line naming where the file was
# saved, which is what the model needs to cite it as evidence anyway. Applies to
# the ASSEMBLED view only; the message list itself is never rewritten.
#
# This is ONE, not a tunable budget: the llama.cpp server this project targets
# rejects any prompt carrying more than one image outright — "At most 1 image(s)
# may be provided in one prompt", HTTP 400. llm.py retries 5xx only, so that 400
# is fatal and kills the run on the SECOND screenshot of any verify session.
# One image is the value every OpenAI-compatible backend accepts, so raising
# this trades a working verify mode for a capability no recorded run has needed.
# Raise it only alongside a backend that is known to accept the higher count.
MAX_IMAGES = 1

# Placeholder that replaces a pruned screenshot's image data. Names the path so
# the model can still cite the artifact it saw earlier in the run.
_PRUNED_IMAGE = "[screenshot pruned — saved at {path}]"


class _ImageRef:
    """Where one screenshot lives: its index in ``_messages`` and its file path."""

    __slots__ = ("index", "path")

    def __init__(self, index: int, path: str) -> None:
        self.index = index
        self.path = path


def _collapse_persisted_images(messages: List[Dict[str, Any]]) -> None:
    """Turn resumed ``image_ref`` parts back into plain text, in place.

    The transcript stores a screenshot as ``{"type": "image_ref", "path": ...}``
    rather than the base64 data URI (see ``_transcript_messages``), so a resumed
    session has the path but not the pixels. That part is not valid provider
    content and the image cannot be reconstructed, so each such message collapses
    to the same placeholder a pruned screenshot gets — a resumed run is told
    exactly what it is: a screenshot was taken, here is where it was saved.
    """
    for entry in messages:
        content = entry.get("content")
        if not isinstance(content, list):
            continue
        if not any(
            isinstance(part, dict) and part.get("type") == "image_ref" for part in content
        ):
            continue
        texts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        paths = [
            part.get("path", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "image_ref"
        ]
        label = " ".join(t for t in texts if t)
        placeholder = " ".join(_PRUNED_IMAGE.format(path=p) for p in paths)
        entry["content"] = f"{label} {placeholder}".strip() if label else placeholder


# The process-wide "current" Session, set by the constructor below. Single-
# session app (mirrors the module-global pattern tools.registry uses for the
# active mode — see its ``_active_mode`` comment): exactly one Session drives
# a process, so "most recently constructed" is "the current one". Lets a tool
# that needs to bill against session usage (the on-demand ``vision`` tool)
# reach the session without threading it through every tool's ``run()``.
_current: "Session | None" = None


def current() -> "Session | None":
    """Return this process's Session, or None before one has been constructed."""
    return _current


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
        # Screenshots attached THIS process, newest last — the input to the
        # assembled view's image pruning. Deliberately empty on resume: a
        # resumed transcript carries paths, not pixels, so its screenshots are
        # collapsed to text here and there is nothing left to prune.
        self._image_refs: List[_ImageRef] = []
        if messages is not None:
            _collapse_persisted_images(self._messages)

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
        prior = stats.read_totals(self.session_id)
        self._stats_run_base: float = prior.run_time
        self._stats_tool_calls: int = prior.tool_calls
        self._stats_input_tokens: int = prior.input_tokens
        self._stats_output_tokens: int = prior.output_tokens
        self._stats_run_started: float = time.monotonic()

        session_lock.acquire(self._lock_path, self.session_id)

        global _current
        _current = self

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

    def append_screenshot(self, image_path: str, data_uri: str, label: str) -> None:
        """Append a user-role message carrying *label* plus the screenshot itself.

        Vision rides the user role with OpenAI content parts — a ``text`` part
        for the label and an ``image_url`` part for the image — because that is
        the only role the chat API accepts image content on. ``llm.to_wire_messages``
        passes ``content`` through untouched, so this reaches the provider as-is
        with no transport change.

        The message is recorded once and never rewritten: pruning happens in the
        assembled view (see ``session_context.assemble_context``) and the base64
        payload is swapped for a path when the transcript is written (see
        ``_transcript_messages``), so neither the context nor the file on disk
        carries every image forever.

        Args:
            image_path: Absolute path to the PNG saved by the screenshot tool.
            data_uri: ``data:image/png;base64,...`` URI for the image_url part.
            label: Text describing the screenshot (e.g. which call produced it).
        """
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
            # Rides the user role out of necessity, but is NOT a turn start —
            # same distinction the ``steer`` flag draws. Without it the pruner's
            # boundary scan would treat every screenshot as a new turn and fold
            # away the tool scaffolding of the very run that took it. Stripped at
            # the wire boundary (llm._NON_WIRE_MESSAGE_KEYS).
            "screenshot": True,
        })
        self._image_refs.append(_ImageRef(index=len(self._messages) - 1, path=image_path))
        self._persist()

    def append_screenshot_description(self, image_path: str, description: str) -> None:
        """Append a screenshot's TEXT description — no pixels — naming its path.

        What a configured vision provider produces instead of an attachment (see
        ``agent._attach_screenshot``): the image never reaches the main model, so
        this row carries prose only. The path is named because the description is
        the model's evidence and it must be able to cite where the artifact was
        saved.

        Carries the same ``screenshot`` flag as a real attachment for the same
        reason — it is a user-role row produced by a tool call MID-turn, not a
        turn start, and the pruner's boundary scan would otherwise fold away the
        tool scaffolding of the very run that took the screenshot. It is NOT
        registered in ``_image_refs``: there is no image data to prune, so it
        must never be counted against the image budget.

        Args:
            image_path: Absolute path to the PNG saved by the screenshot tool.
            description: The vision provider's description of that image.
        """
        self._messages.append({
            "role": "user",
            "content": (
                f"[screenshot saved at {image_path} — description from the vision "
                f"model; the image itself is not attached]\n{description}"
            ),
            "screenshot": True,
        })
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

        The shaping — provider blocks, the summary header, the folded task
        anchor and the pruned tail — lives in ``session_context``, beside the
        pruning it already owned. This method supplies the state and the
        provider registry; that module decides what the model actually sees.
        """
        return session_context.assemble_context(self, CONTEXT_PROVIDERS)

    # ------------------------------------------------------------------ private

    def _transcript_messages(self) -> List[Dict[str, Any]]:
        """The message list as it should be WRITTEN — image data swapped for paths.

        The only place the on-disk transcript deviates from ``_messages``, and it
        is a substitution rather than a loss: an ``image_url`` part becomes
        ``{"type": "image_ref", "path": ...}``, so the transcript records that a
        screenshot was taken and where the PNG lives, without embedding a base64
        payload that would dominate the file and be re-read on every append.
        Messages with no image parts are passed through by identity.
        """
        if not self._image_refs:
            return self._messages

        paths_by_index = {ref.index: ref.path for ref in self._image_refs}
        out: List[Dict[str, Any]] = []
        for index, entry in enumerate(self._messages):
            path = paths_by_index.get(index)
            content = entry.get("content")
            if path is None or not isinstance(content, list):
                out.append(entry)
                continue
            out.append({
                **entry,
                "content": [
                    {"type": "image_ref", "path": path}
                    if isinstance(part, dict) and part.get("type") == "image_url"
                    else part
                    for part in content
                ],
            })
        return out

    def _persist(self) -> None:
        """Hand this session's persisted state to the transcript store.

        The file format itself — the frontmatter keys, the body schema and the
        atomic rename — belongs to ``session_store``, alongside the reader that
        has to agree with it. This method only supplies the state.
        """
        _write_transcript(
            self.transcript_path,
            session_id=self.session_id,
            project_root=self.project_root,
            model=self.model,
            created_at=self.created_at,
            messages=self._transcript_messages(),
            summary=self._summary,
            summary_covers=self._summary_covers,
            episodic_watermark=self.episodic_watermark,
        )

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
            stats.Totals(
                round(run_time, 3),
                self._stats_tool_calls,
                self._stats_input_tokens,
                self._stats_output_tokens,
            ),
        )

    def close(self) -> None:
        """Release this session's advisory lock file, if this process still owns it.

        Safe to call multiple times and safe to call even if the lock was
        never successfully acquired (e.g. constructor raised before writing
        it) — both cases are silently no-ops.
        """
        session_lock.release(self._lock_path)
