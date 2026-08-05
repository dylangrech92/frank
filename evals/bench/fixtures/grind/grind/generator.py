"""Synthetic input generation.

Nothing here reads the clock, the filesystem, or the environment for
randomness: every run is driven by a single seeded ``random.Random``
instance, so the same ``(scale, seed)`` pair always produces byte-identical
downstream output.
"""

from __future__ import annotations

import json
import random

from . import config
from .records import RawEvent

# A handful of path segments carry pre-encoded characters (spaces, dashes,
# accented letters) so downstream parsing sees realistic percent-encoding
# rather than clean ASCII on every record.
SEGMENT_SUFFIXES = (
    "",
    "%20edition",
    "-refurb%2Dgrade%2Db",
    "%2Fbundle",
    "%C3%A9dition",
    "-2024%20q4",
    "%20%28clearance%29",
)

SLUG_WORDS = (
    "wireless",
    "noise-cancelling",
    "refurbished",
    "limited",
    "heavy-duty",
    "compact",
    "premium",
    "waterproof",
    "adjustable",
    "stainless",
    "cordless",
    "portable",
    "rechargeable",
    "insulated",
    "ergonomic",
    "modular",
    "reversible",
    "extended",
    "graphite",
    "matte-black",
)

REFERRER_HOSTS = (
    "https://search.example.com/",
    "https://mail.example.net/",
    "https://social.example.org/",
    "-",
    "https://partner-ads.example.com/",
)

METHODS = ("GET", "GET", "GET", "POST", "PUT", "DELETE")


def _weighted_status(rng: random.Random) -> int:
    total = sum(weight for _, weight in config.STATUS_WEIGHTS)
    pick = rng.randint(1, total)
    upto = 0
    for status, weight in config.STATUS_WEIGHTS:
        upto += weight
        if pick <= upto:
            return status
    return config.STATUS_WEIGHTS[-1][0]


def _build_path(rng: random.Random) -> str:
    root = rng.choice(config.PATH_ROOTS)
    slug_id = rng.randint(1000, 99999)
    descriptors = "-".join(rng.sample(SLUG_WORDS, rng.randint(2, 4)))
    suffix = rng.choice(SEGMENT_SUFFIXES)
    segment = f"item-{slug_id}-{descriptors}{suffix}"
    if rng.random() < 0.35:
        query = f"?ref={rng.choice(('email', 'push', 'sms', 'organic'))}&pos={rng.randint(1, 40)}"
    else:
        query = ""
    return f"/{root}/{segment}{query}"


_HEADER_NAMES = (
    "accept",
    "accept-encoding",
    "accept-language",
    "cache-control",
    "connection",
    "dnt",
    "sec-fetch-mode",
    "sec-fetch-site",
    "upgrade-insecure-requests",
    "x-forwarded-proto",
)

_HEADER_VALUES = (
    "1",
    "same-origin",
    "no-cache",
    "keep-alive",
    "gzip, deflate, br",
    "en-US,en;q=0.9",
    "text/html,application/xhtml+xml",
    "https",
    "navigate",
    "none",
)

# Short page-view tokens used to synthesize a personalization history list
# (below). Not real paths — just enough variety that the history list
# doesn't compress to one repeated string.
_HISTORY_PAGES = (
    "home",
    "category",
    "product",
    "cart",
    "wishlist",
    "search-results",
    "account-settings",
    "order-status",
    "help-center",
    "promo-landing",
)


def _build_history(rng: random.Random) -> list[dict]:
    # The personalization service looks back over a client's recent page
    # views (with dwell time, so short bounces can be weighted down) to
    # bias ranking; a dozen or so per event is normal for an active session.
    # One wide getrandbits draw per entry, sliced into fields, keeps this
    # from turning into a per-field random-number-generator call per entry.
    depth = rng.randint(10, 20)
    entries = []
    for i in range(depth):
        bits = rng.getrandbits(48)
        page = _HISTORY_PAGES[bits % len(_HISTORY_PAGES)]
        page_num = (bits >> 8) % 900 + 1
        offset = (bits >> 20) % 3601
        duration = 200 + ((bits >> 32) % 44801)
        entries.append(
            {
                "index": i,
                "page": f"{page}-{page_num}",
                "ts_offset_s": offset,
                "duration_ms": duration,
                "bounced": duration < 1500,
            }
        )
    return entries


def _build_metadata(rng: random.Random) -> str:
    device = rng.choice(config.DEVICE_TYPES)
    browser = rng.choice(config.BROWSER_FAMILIES)
    plan = rng.choice(config.PLAN_TIERS)
    experiment_ids = rng.sample(range(1, 30), rng.randint(0, 3))
    experiments = sorted(f"exp_{n}" for n in experiment_ids)
    # Assignment detail the experimentation service attaches per event, kept
    # alongside the plain experiment id list so the report can audit bucket
    # assignment without a second lookup.
    experiment_assignments = {
        f"exp_{n}": {
            "bucket": rng.choice(("control", "treatment")),
            "weight": round(rng.uniform(0.05, 0.5), 3),
        }
        for n in experiment_ids
    }
    headers = {
        name: rng.choice(_HEADER_VALUES) for name in rng.sample(_HEADER_NAMES, 6)
    }
    payload = {
        "device": device,
        "browser": browser,
        "plan": plan,
        "recent_page_views": _build_history(rng),
        "experiments": experiments,
        "experiment_assignments": experiment_assignments,
        "cart_items": rng.randint(0, 12),
        "logged_in": rng.random() < 0.6,
        "request_headers": headers,
    }
    return json.dumps(payload, sort_keys=True)


def _session_pool(rng: random.Random, count: int) -> list[str]:
    """A bounded pool of session ids so some sessions legitimately repeat
    (retries, pagination) while most events belong to distinct sessions."""
    pool_size = max(1, count // 6)
    return [f"sess-{rng.getrandbits(32):08x}" for _ in range(pool_size)]


def generate_events(scale: int, seed: int) -> list[RawEvent]:
    if scale < 1:
        raise ValueError("scale must be >= 1")

    count = config.BASE_RECORD_COUNT * scale
    rng = random.Random(seed)
    sessions = _session_pool(rng, count)

    events: list[RawEvent] = []
    for i in range(count):
        record_id = f"evt-{i:07d}"
        session_id = rng.choice(sessions)
        # Retries within a session land close together; drawing repeatedly
        # from the same small session pool is what makes those retries
        # show up as adjacent or near-adjacent duplicate keys downstream.
        ts_offset = i * 2 + rng.randint(0, 1)
        ts = f"2025-01-01T{(ts_offset // 3600) % 24:02d}:{(ts_offset // 60) % 60:02d}:{ts_offset % 60:02d}Z"
        method = rng.choice(METHODS)
        path = _build_path(rng)
        referrer = rng.choice(REFERRER_HOSTS)
        status = _weighted_status(rng)
        bytes_sent = rng.randint(180, 48000)
        region = rng.choice(config.REGIONS)
        ua_variant = rng.randint(0, 999999)
        user_agent = (
            f"GrindBrowser/{rng.randint(1, 9)}.{rng.randint(0, 40)} "
            f"(build-{ua_variant:06d})"
        )
        metadata_json = _build_metadata(rng)

        events.append(
            RawEvent(
                record_id=record_id,
                session_id=session_id,
                ts=ts,
                method=method,
                path=path,
                referrer=referrer,
                status=status,
                bytes_sent=bytes_sent,
                region=region,
                user_agent=user_agent,
                metadata_json=metadata_json,
            )
        )

    # A small, fixed fraction of exact re-sends: the same record replayed
    # verbatim, as a client retry would. Deterministic positions (every
    # 37th record mirrors an earlier one) rather than a further RNG draw,
    # so the duplicate rate does not shift if upstream generation changes.
    for i in range(37, count, 37):
        source = events[i - 19]
        events[i] = RawEvent(
            record_id=f"evt-{i:07d}",
            session_id=source.session_id,
            ts=events[i].ts,
            method=source.method,
            path=source.path,
            referrer=source.referrer,
            status=source.status,
            bytes_sent=source.bytes_sent,
            region=source.region,
            user_agent=source.user_agent,
            metadata_json=source.metadata_json,
        )

    return events
