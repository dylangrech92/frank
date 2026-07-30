"""Helpers for reading and listing transcript files on disk."""

from __future__ import annotations

import json
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
