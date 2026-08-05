"""A minimal FIFO queue used to buffer raw events between ingestion and
processing."""
from __future__ import annotations

from collections import deque
from typing import Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


class QueueFullError(Exception):
    """Raised by `InMemoryQueue.push` when the queue is already at its
    configured maximum size."""


class InMemoryQueue(Generic[T]):
    """A bounded FIFO queue. `max_size=None` means unbounded."""

    def __init__(self, max_size: int | None = None) -> None:
        self._items: deque[T] = deque()
        self._max_size = max_size

    def push(self, item: T) -> None:
        if self.is_full():
            raise QueueFullError(f"queue is at its max size of {self._max_size}")
        self._items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.push(item)

    def pop(self) -> T | None:
        return self._items.popleft() if self._items else None

    def drain(self) -> list[T]:
        """Pop every item currently queued, oldest first, and return
        them as a list. Equivalent to calling `pop()` until it returns
        `None`, without the caller needing to check for the sentinel."""
        items = list(self._items)
        self._items.clear()
        return items

    def peek(self) -> T | None:
        return self._items[0] if self._items else None

    def clear(self) -> None:
        self._items.clear()

    def is_full(self) -> bool:
        return self._max_size is not None and len(self._items) >= self._max_size

    def remaining_capacity(self) -> int | None:
        """How many more items can be pushed before the queue is full,
        or `None` if it is unbounded."""
        if self._max_size is None:
            return None
        return max(self._max_size - len(self._items), 0)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)
