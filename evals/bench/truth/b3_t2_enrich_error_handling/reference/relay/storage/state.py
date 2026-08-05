"""Tracks which event ids have completed the pipeline, for idempotency
checks and for the `stats` CLI command."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

_HISTORY_MAX_SIZE = 50


class StateStore:
    """In-memory record of processed event ids and per-outcome counters.

    Optionally backed by a JSON snapshot file so a demo run can resume
    its counters across restarts - the pipeline's actual event data
    always lives in `delivered.jsonl` / `deadletters.jsonl`, this is
    only a small summary.
    """

    def __init__(self, snapshot_path: str | Path | None = None) -> None:
        self._processed: set[str] = set()
        self._counters: dict[str, int] = {"delivered": 0, "dead_lettered": 0, "retried": 0}
        self._history: deque[str] = deque(maxlen=_HISTORY_MAX_SIZE)
        self._snapshot_path = Path(snapshot_path) if snapshot_path else None
        if self._snapshot_path and self._snapshot_path.exists():
            self._load()

    def mark_processed(self, event_id: str) -> None:
        self._processed.add(event_id)
        self._counters["delivered"] += 1
        self._history.append(event_id)
        self._maybe_save()

    def mark_dead_lettered(self) -> None:
        self._counters["dead_lettered"] += 1
        self._maybe_save()

    def mark_retried(self) -> None:
        self._counters["retried"] += 1
        self._maybe_save()

    def is_processed(self, event_id: str) -> bool:
        return event_id in self._processed

    def processed_count(self) -> int:
        return len(self._processed)

    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    def recent_history(self) -> list[str]:
        """The last `_HISTORY_MAX_SIZE` successfully delivered event
        ids, oldest first."""
        return list(self._history)

    def reset(self) -> None:
        """Clear every processed id, counter, and history entry. Does
        not touch the snapshot file on disk until the next mutation
        saves over it."""
        self._processed.clear()
        self._history.clear()
        for key in self._counters:
            self._counters[key] = 0

    def _load(self) -> None:
        data = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        self._processed = set(data.get("processed", []))
        self._counters.update(data.get("counters", {}))

    def _maybe_save(self) -> None:
        if self._snapshot_path is None:
            return
        payload = {"processed": sorted(self._processed), "counters": self._counters}
        self._snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
