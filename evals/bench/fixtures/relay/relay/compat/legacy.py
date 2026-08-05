"""Backwards-compatible aliases for pre-2.0 call sites. New code should
import the current names directly from `relay.bus`,
`relay.handlers.transform`, and `relay.events` instead of importing from
here.
"""
from __future__ import annotations

from relay.bus import EventBus as RelayBus  # noqa: F401 - pre-2.0 class name
from relay.events import EVENT_ORDER_CREATED as CHANNEL_ORDER_NEW  # noqa: F401
from relay.events import EVENT_ORDER_VALIDATED as CHANNEL_ORDER_OK  # noqa: F401
from relay.handlers.transform import normalize_event as clean_event  # noqa: F401
from relay.handlers.transform import normalize_event as sanitize_order  # noqa: F401
