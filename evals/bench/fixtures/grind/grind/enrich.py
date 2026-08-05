"""Client profile enrichment.

Each event's user agent is resolved to a client profile (device family,
browser, plan tier, active experiments) that the aggregate report joins
against the raw counts. Profiles are cached by user agent so a client
making several requests in the same batch only pays the build cost once.
"""

from __future__ import annotations

import json

from .records import ParsedEvent


def build_profile(event: ParsedEvent) -> dict:
    # The parse stage already decoded this event's metadata, but the profile
    # fields (device/browser/plan/experiments) map onto the raw payload
    # one-for-one, so re-decoding here keeps this module independent of
    # exactly which extra keys parse.py has folded into `event.metadata`.
    metadata = json.loads(event.metadata_json) if event.metadata_json else {}
    return {
        "device": metadata.get("device", "unknown"),
        "browser": metadata.get("browser", "unknown"),
        "plan": metadata.get("plan", "unknown"),
        "experiments": tuple(metadata.get("experiments", ())),
        "logged_in": bool(metadata.get("logged_in", False)),
        "cart_items": int(metadata.get("cart_items", 0)),
        "region_hint": event.region,
        "sample_path": event.canonical_path,
    }


def enrich_events(events: list[ParsedEvent], cache: dict) -> list[dict]:
    """Attach a client profile to every event, filling `cache` as it goes."""
    enriched: list[dict] = []
    for event in events:
        profile = cache.get(event.user_agent)
        if profile is None:
            profile = build_profile(event)
            cache[event.user_agent] = profile
        enriched.append(profile)
    return enriched
