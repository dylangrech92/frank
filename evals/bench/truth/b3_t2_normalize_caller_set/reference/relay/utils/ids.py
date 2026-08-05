"""Event id generation and lightweight id-shape helpers."""
from __future__ import annotations

import itertools
import re
import time

_counter = itertools.count(1)

_EVENT_ID_RE = re.compile(r"^evt-\d+-\d+$")


def generate_event_id() -> str:
    """Return a short, monotonically distinguishable event id.

    Not a UUID - this project favors ids that sort and read predictably
    in demo output over global uniqueness guarantees.
    """
    return f"evt-{int(time.time() * 1000)}-{next(_counter)}"


def is_well_formed_event_id(event_id: str) -> bool:
    """True if `event_id` has the shape `generate_event_id` produces."""
    return bool(_EVENT_ID_RE.match(event_id))


def short_id(event_id: str, length: int = 8) -> str:
    """A truncated id suitable for compact log lines, e.g. in the CLI's
    per-event summary output."""
    return event_id[-length:] if len(event_id) > length else event_id
