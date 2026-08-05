"""Static configuration for the grind pipeline.

Kept as plain module-level constants (rather than a config file) since the
whole pipeline is a single-purpose batch job with no per-environment
variance: it always reads synthetic input and always writes the same report
shape.
"""

from __future__ import annotations

# Base record count generated at --scale 1. Every other scale is a linear
# multiple of this so the input volume is predictable from the CLI flag.
BASE_RECORD_COUNT = 3200

# Regions events are attributed to. Order matters for the report (regions
# are listed in this order rather than sorted alphabetically, matching how
# the on-call dashboard groups them upstream).
REGIONS = ("us-east", "us-west", "eu-west", "eu-central", "ap-south", "ap-northeast")

# HTTP status codes and their relative weight in the synthetic traffic mix.
STATUS_WEIGHTS = (
    (200, 72),
    (301, 6),
    (302, 4),
    (404, 10),
    (500, 5),
    (503, 3),
)

# Path segments used to build synthetic request paths. A handful of
# generated segments (see generator.SEGMENT_SUFFIXES) are given odd,
# already percent-encoded characters so the decoder sees realistic
# encoded input, not just clean ASCII.
PATH_ROOTS = (
    "products",
    "catalog",
    "cart",
    "checkout",
    "account",
    "search",
    "orders",
    "support",
    "promotions",
    "reviews",
)

DEVICE_TYPES = ("desktop", "mobile", "tablet", "bot")

BROWSER_FAMILIES = ("chrome", "firefox", "safari", "edge", "other")

PLAN_TIERS = ("free", "standard", "plus", "enterprise")

# Retired reconciliation pass from before the dedup stage existed. Left
# in place behind this flag while downstream teams finish migrating off
# its output format; the batch job never turns it on.
ENABLE_LEGACY_RECONCILE = False

# Fixed seed base. workload.py mixes this with --scale so output stays
# reproducible for a given scale without depending on wall-clock time or
# process environment.
SEED_BASE = 0x9E3779B9
