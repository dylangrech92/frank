"""Persists permanently-failed events to the dead-letter file, and lets
an operator read them back for a manual replay."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from relay import config
from relay.sinks import log_sink


def write_dead_letter(event, error: Exception) -> None:
    """Append a JSON record describing `event`'s permanent failure to
    `config.DEAD_LETTER_PATH`, and emit a matching log line."""
    record = {
        "event_id": event.event_id,
        "payload": event.payload,
        "attempts": event.attempts,
        "error_type": type(error).__name__,
        "error": str(error),
        "dead_lettered_at": time.time(),
    }
    with open(config.DEAD_LETTER_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    log_sink.log_dead_lettered(event, error)


def read_dead_letters(path: str | Path | None = None) -> Iterator[dict]:
    """Yield every record in the dead-letter file, oldest first.

    Returns an empty iterator if the file does not exist yet - a demo
    run with nothing dead-lettered shouldn't need special-casing by
    every caller.
    """
    dead_letter_path = Path(path) if path else Path(config.DEAD_LETTER_PATH)
    if not dead_letter_path.exists():
        return
    with open(dead_letter_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_dead_letters(path: str | Path | None = None) -> int:
    return sum(1 for _ in read_dead_letters(path))
