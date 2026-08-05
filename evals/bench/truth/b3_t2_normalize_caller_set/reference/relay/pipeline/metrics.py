"""In-process counters and stage timing.

Nothing here is persisted; the pipeline is a demo process, so metrics
only need to survive for the lifetime of one run and are printed by the
`stats` CLI command.
"""
from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from typing import Iterator


class MetricsRegistry:
    """A small counter/timer store keyed by arbitrary string names."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._durations_s: dict[str, list[float]] = {}

    def increment(self, name: str, by: int = 1) -> None:
        self._counts[name] += by

    def count(self, name: str) -> int:
        return self._counts[name]

    def record_duration(self, name: str, seconds: float) -> None:
        self._durations_s.setdefault(name, []).append(seconds)

    def mean_duration(self, name: str) -> float:
        samples = self._durations_s.get(name)
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def snapshot(self) -> dict[str, object]:
        """Return a plain-dict view suitable for printing or JSON
        serialization."""
        return {
            "counts": dict(self._counts),
            "mean_duration_s": {name: self.mean_duration(name) for name in self._durations_s},
        }

    def total_duration(self, name: str) -> float:
        """Sum of every recorded duration under `name`, in seconds."""
        return sum(self._durations_s.get(name, []))

    def slowest_stages(self, limit: int = 3) -> list[tuple[str, float]]:
        """The `limit` timer names with the highest mean duration,
        slowest first. Names with no recorded samples are excluded."""
        ranked = sorted(
            ((name, self.mean_duration(name)) for name in self._durations_s),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:limit]

    def reset(self) -> None:
        """Clear every counter and recorded duration. Mainly useful for
        a long-lived process that wants a fresh window of metrics
        without restarting."""
        self._counts.clear()
        self._durations_s.clear()

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Context manager: records how long the `with` block took under
        `name`, and also increments a matching `<name>.count` counter."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_duration(name, time.perf_counter() - started)
            self.increment(f"{name}.count")


metrics = MetricsRegistry()
