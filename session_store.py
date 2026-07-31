"""Helpers for reading and listing transcript files on disk."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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


def _write_transcript(
    path: Path,
    *,
    session_id: str,
    project_root: Path,
    model: str,
    created_at: datetime,
    messages: List[Dict[str, Any]],
    summary: str | None,
    summary_covers: int,
    episodic_watermark: int,
) -> None:
    """Write a transcript file: YAML frontmatter, then the JSON body.

    The exact inverse of ``_load_transcript``. The four frontmatter keys and the
    body schema are spelled out here, next to the reader that has to agree with
    them, so the two halves of one format cannot drift apart in separate files.

    Writes to a pid-suffixed sibling temp file first, then atomically renames it
    into place via ``os.replace`` so a resuming reader never observes a
    partially-written transcript (the file is load-bearing for ``Session.resume``,
    not just an append-only log).

    Args:
        path: Destination ``<session_id>.json`` transcript file.
        session_id: Session id, written to the frontmatter.
        project_root: Resolved project root, written to the frontmatter as ``cwd``.
        model: Model identifier, written to the frontmatter.
        created_at: Session creation time, written to the frontmatter.
        messages: Full OpenAI-format message list forming the body.
        summary: Compaction summary, or None when the session has none.
        summary_covers: How many leading messages ``summary`` covers.
        episodic_watermark: Row count already handed to the episodic encoder.
    """
    lines = [
        "---",
        f"session_id: {session_id}",
        f"cwd: {str(project_root)}",
        f"model: {model}",
        f"created_at: {created_at.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "---",
    ]

    body_obj: Dict[str, Any] = {"messages": messages}
    if summary:
        body_obj["summary"] = summary
        body_obj["summary_covers"] = summary_covers
    if episodic_watermark:
        body_obj["episodic_watermark"] = episodic_watermark

    body = _format_json(body_obj)
    file_contents = "\n".join(lines) + "\n" + body

    tmp_path = path.with_suffix(
        path.suffix + f".{os.getpid()}.tmp"
    )
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(file_contents)
    os.replace(tmp_path, path)
