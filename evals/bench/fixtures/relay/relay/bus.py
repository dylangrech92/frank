"""In-process publish/subscribe event bus.

Handlers register for a named event via the `on` decorator; `publish`
invokes every handler registered for that name, in registration order.
The bus does not know anything about pipeline stages - it only knows
event names and the callables subscribed to them.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

logger = logging.getLogger(__name__)

HandlerFunc = Callable[[object], None]


class EventBus:
    """Routes named events to zero or more registered handler callables."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[HandlerFunc]] = defaultdict(list)

    def on(self, event_name: str) -> Callable[[HandlerFunc], HandlerFunc]:
        """Decorator: register `func` to run whenever `event_name` is
        published."""
        def decorator(func: HandlerFunc) -> HandlerFunc:
            self._handlers[event_name].append(func)
            logger.debug("registered %s for %s", func.__qualname__, event_name)
            return func
        return decorator

    def publish(self, event_name: str, event: object) -> list[Exception]:
        """Invoke every handler registered for `event_name`, in
        registration order.

        A handler that raises does not prevent the remaining handlers
        from running; every exception raised is collected and returned
        instead of propagating.
        """
        errors: list[Exception] = []
        for handler in self._handlers.get(event_name, []):
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001 - handlers own their errors
                logger.warning(
                    "handler %s failed for %s: %s",
                    getattr(handler, "__qualname__", repr(handler)),
                    event_name,
                    exc,
                )
                errors.append(exc)
        return errors

    def handler_count(self, event_name: str) -> int:
        return len(self._handlers.get(event_name, []))

    def off(self, event_name: str, func: HandlerFunc) -> bool:
        """Unregister `func` from `event_name`. Returns True if it was
        registered, False otherwise."""
        handlers = self._handlers.get(event_name)
        if not handlers or func not in handlers:
            return False
        handlers.remove(func)
        return True

    def known_event_names(self) -> list[str]:
        """Every event name that currently has at least one handler
        registered."""
        return [name for name, handlers in self._handlers.items() if handlers]


bus = EventBus()
