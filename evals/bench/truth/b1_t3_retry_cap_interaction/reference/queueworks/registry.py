"""A name -> callable registry for task functions.

Jobs are persisted to disk as plain JSON (see :mod:`queueworks.store`), so a
job cannot carry a pickled function reference. Instead each job stores the
*name* of the task it wants to run, and workers resolve that name back to a
callable through this registry. Applications register their task functions
with the :func:`task` decorator, typically at import time.
"""

from __future__ import annotations

from typing import Callable, Dict

_TASKS: Dict[str, Callable] = {}


def task(name: str) -> Callable[[Callable], Callable]:
    """Decorator that registers ``fn`` under ``name`` for later lookup."""

    def decorator(fn: Callable) -> Callable:
        if name in _TASKS and _TASKS[name] is not fn:
            raise ValueError(f"task {name!r} is already registered")
        _TASKS[name] = fn
        return fn

    return decorator


def get_task(name: str) -> Callable:
    """Resolve a registered task function by name."""

    try:
        return _TASKS[name]
    except KeyError as exc:
        raise LookupError(f"no task registered under name {name!r}") from exc


def registered_names() -> list:
    """Return the names of every currently registered task."""

    return sorted(_TASKS)


def clear() -> None:
    """Remove all registered tasks. Mainly useful for test isolation."""

    _TASKS.clear()
