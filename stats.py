"""Per-session usage statistics, upserted to a gitignored ``stats.json``.

After every LLM call the agent records the running session's cumulative usage --
wall-clock run time, tool-call count, and prompt/completion token totals -- so
usage across sessions can be reviewed later. The file is a JSON array with one
row per ``session_id``; each row is upserted in place.

The file lives beside this module (the coding-agent install dir), NOT in the
target project's cwd, because a single agent run operates on arbitrary projects
and its usage should aggregate in one place. Parallel subagent runs share this
one file, so the read-modify-write is guarded by an exclusive ``flock`` to keep
concurrent processes from clobbering each other's rows.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any, Dict, List

STATS_PATH = Path(__file__).resolve().parent / "stats.json"
JS_PATH = Path(__file__).resolve().parent / "stats.js"


def _parse(content: str) -> List[Dict[str, Any]]:
    """Parse *content* as the stats array, tolerating an empty/corrupt file."""
    if not content.strip():
        return []
    try:
        rows = json.loads(content)
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def read_row(session_id: str, path: Path = STATS_PATH) -> Dict[str, Any] | None:
    """Return the stored row for *session_id*, or ``None`` if the file has none.

    Used to seed a (possibly resumed) session's in-memory counters so its totals
    continue rather than reset. Read without a lock: a point-in-time snapshot at
    session start is sufficient, and no writer for this same session_id can be
    running concurrently (the session lock in ``session.py`` forbids it).
    """
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    for row in _parse(content):
        if row.get("session_id") == session_id:
            return row
    return None


def upsert(
    session_id: str,
    run_time: float,
    tool_calls: int,
    input_tokens: int,
    output_tokens: int,
    path: Path = STATS_PATH,
    js_path: Path = JS_PATH,
) -> None:
    """Upsert this session's row into ``stats.json`` (absolute cumulative values).

    The whole read-modify-write is performed under an exclusive ``flock`` on the
    file so concurrent subagent processes serialize instead of losing rows.
    """
    row = {
        "session_id": session_id,
        "run_time": run_time,
        "tool_calls": tool_calls,
        "input": input_tokens,
        "output": output_tokens,
    }
    # "a+" creates the file if absent and never truncates on open, so an
    # existing array survives until we deliberately rewrite it under the lock.
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            rows = _parse(f.read())
            for i, existing in enumerate(rows):
                if existing.get("session_id") == session_id:
                    rows[i] = row
                    break
            else:
                rows.append(row)
            f.seek(0)
            f.truncate()
            # One row per line: a valid JSON array, but each session's object
            # stays on a single line so the file is readable and compact as it
            # grows to many sessions (pretty-printing would balloon it).
            body = ",\n".join(json.dumps(r) for r in rows)
            f.write(f"[\n{body}\n]" if rows else "[]")
            f.flush()

            # Mirror the same rows into stats.js as window.STATS_DATA so
            # dashboard.html can load it via <script src> — which works on
            # file:// (unlike fetch, which browsers block from file:// origins
            # with a CORS error). Written inside this lock so concurrent
            # subagent writes can't interleave the two files.
            try:
                js_path.write_text(
                    ("window.STATS_DATA = [\n" + body + "\n];\n")
                    if rows
                    else "window.STATS_DATA = [];\n",
                    encoding="utf-8",
                )
            except OSError:
                # Best-effort derived artifact: never let a stats.js write
                # failure (e.g. read-only mount) break stats.json recording.
                pass
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
