"""Pipeline orchestration: generate -> parse -> dedup -> enrich -> aggregate."""

from __future__ import annotations

from . import config, legacy
from .aggregate import build_report
from .dedup import mark_duplicates
from .enrich import enrich_events
from .generator import generate_events
from .parse import parse_events


def run(scale: int) -> dict:
    seed = config.SEED_BASE ^ (scale * 0x1000003)
    raw_events = generate_events(scale, seed)
    parsed_events, rejected = parse_events(raw_events)

    if config.ENABLE_LEGACY_RECONCILE:
        legacy.reconcile_batch_legacy(parsed_events)

    duplicate_count = mark_duplicates(parsed_events)

    profile_cache: dict = {}
    enrich_events(parsed_events, profile_cache)

    return build_report(parsed_events, rejected, duplicate_count, len(profile_cache))
