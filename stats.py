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

A file that exists but does not parse is never silently discarded: its bytes are
copied to ``stats.json.corrupt-<timestamp>-<pid>`` and the failure is reported on
stderr before recording restarts from an empty array. Treating unparseable
content as "no history" instead would let a single bad write erase every stored
row with no error anywhere -- the read succeeds, returns nothing, and the next
upsert rewrites the file with one row.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

STATS_PATH = Path(__file__).resolve().parent / "stats.json"
JS_PATH = Path(__file__).resolve().parent / "stats.js"


class CorruptStatsFile(Exception):
    """``stats.json`` exists but is not the JSON array this module writes."""


class Totals(NamedTuple):
    """One session's cumulative usage, and the only place the row keys live.

    ``stats.json``'s key names are a published format, not an internal detail:
    ``stats.js`` mirrors the rows verbatim and ``dashboard.html`` reads
    ``r.input`` / ``r.output`` / ``r.run_time`` / ``r.tool_calls`` straight off
    them. Callers pass and receive this type instead of a bare dict so a key is
    spelled in exactly one place -- a reader that spelled its own would keep
    parsing happily and silently report zeros if a name here ever moved.
    """

    run_time: float = 0.0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def as_row(self, session_id: str) -> Dict[str, Any]:
        """This session's ``stats.json`` row."""
        return {
            "session_id": session_id,
            "run_time": self.run_time,
            "tool_calls": self.tool_calls,
            "input": self.input_tokens,
            "output": self.output_tokens,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Totals:
        """Read a stored row back, defaulting any field the row omits."""
        return cls(
            float(row.get("run_time", 0.0)),
            int(row.get("tool_calls", 0)),
            int(row.get("input", 0)),
            int(row.get("output", 0)),
        )


def _parse(content: str) -> List[Dict[str, Any]]:
    """Parse *content* as the stats array. Empty is normal; malformed is not.

    Raises ``CorruptStatsFile`` rather than returning ``[]`` so that a caller
    about to rewrite the file can preserve the bytes first.
    """
    if not content.strip():
        return []
    try:
        rows = json.loads(content)
    except json.JSONDecodeError as exc:
        raise CorruptStatsFile(f"not valid JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise CorruptStatsFile(f"expected a JSON array, got {type(rows).__name__}")
    return rows


def _quarantine(path: Path, content: str, reason: Exception) -> None:
    """Copy unparseable *content* aside and report the loss on stderr.

    Called under the write lock, immediately before the file is rewritten from
    scratch. The copy is what makes the rows recoverable; the stderr line is what
    stops the loss from going unnoticed until someone opens the dashboard.
    """
    stamp = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    aside = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        aside.write_text(content, encoding="utf-8")
        kept = f"{len(content)} bytes kept at {aside.name}"
    except OSError as exc:
        kept = f"{len(content)} bytes COULD NOT be kept ({exc})"
    print(
        f"stats: {path.name} is corrupt ({reason}); {kept}; starting a new file",
        file=sys.stderr,
        flush=True,
    )


def read_totals(session_id: str, path: Path = STATS_PATH) -> Totals:
    """Return stored cumulative usage for *session_id*; zeros if it has no row.

    Used to seed a (possibly resumed) session's in-memory counters so its totals
    continue rather than reset. Read without a lock: a point-in-time snapshot at
    session start is sufficient, and no writer for this same session_id can be
    running concurrently (the session lock in ``session.py`` forbids it).

    A corrupt file seeds zeros rather than raising -- telemetry must not stop a
    session from starting -- but says so on stderr. Preserving the bytes is left
    to ``upsert``, which holds the lock this reader deliberately does not.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Totals()
    try:
        rows = _parse(content)
    except CorruptStatsFile as exc:
        print(
            f"stats: cannot seed totals from {path.name}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return Totals()
    for row in rows:
        if row.get("session_id") == session_id:
            return Totals.from_row(row)
    return Totals()


def upsert(
    session_id: str,
    totals: Totals,
    path: Path = STATS_PATH,
    js_path: Path = JS_PATH,
) -> None:
    """Upsert this session's row into ``stats.json`` (absolute cumulative values).

    The whole read-modify-write is performed under an exclusive ``flock`` on the
    file so concurrent subagent processes serialize instead of losing rows.
    """
    row = totals.as_row(session_id)
    # "a+" creates the file if absent and never truncates on open, so an
    # existing array survives until we deliberately rewrite it under the lock.
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            try:
                rows = _parse(content)
            except CorruptStatsFile as exc:
                # Rewriting from scratch is the only way forward, so keep the
                # old bytes first -- this rewrite is what would destroy them.
                _quarantine(path, content, exc)
                rows = []
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
