"""Runtime configuration for the relay pipeline.

In production this would be sourced from environment variables or a
config service; for this project it is a plain module so the demo has
zero external dependencies. Every stage that needs a tunable value reads
it from here rather than hard-coding it.
"""

# --- output locations -----------------------------------------------------
OUTPUT_PATH = "delivered.jsonl"
DEAD_LETTER_PATH = "deadletters.jsonl"

# --- delivery ---------------------------------------------------------
# One of the keys in `relay.handlers.delivery.SINK_DISPATCH`. The demo
# ships with "file"; "webhook" is exercised in isolation by the plugin
# authors before it is turned on for a whole deployment.
DELIVERY_SINK = "file"

# --- retry policy -----------------------------------------------------
RETRY_ENABLED = True
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_S = 0.05
RETRY_BACKOFF_MAX_S = 5.0
# How long a scheduled retry may sit in `Scheduler` waiting for its
# delay to elapse before the demo's settle loop gives up on it via
# `Scheduler.expire_stale` rather than draining it forever.
RETRY_STALE_TIMEOUT_S = 30.0

# Whether an operator can pull dead-lettered events back through the
# pipeline with `main.py replay`. Independent of `RETRY_ENABLED`: this
# gates the manual CLI command, not the pipeline's own automatic
# in-flight retry attempts.
REPLAY_ENABLED = True

# --- enrichment ---------------------------------------------------------
# One of the plugin module names under `relay/plugins/` that exposes a
# `PLUGIN` entry point; loaded dynamically by `relay.plugins.loader`.
ENRICHMENT_PLUGIN = "builtin_enricher"

# --- validation -----------------------------------------------------------
REQUIRED_ORDER_FIELDS = ("sku", "qty", "customer_id")
MAX_QTY_PER_ORDER = 500
SKU_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{1,31}$"

# --- queueing -------------------------------------------------------------
INBOX_MAX_SIZE = 10_000

# --- health thresholds ------------------------------------------------
# Consulted by `relay.pipeline.health`, not by the pipeline itself - a
# breach doesn't stop processing, it only changes what the CLI `health`
# command reports.
HEALTH_MAX_DEAD_LETTER_BACKLOG = 25
HEALTH_MAX_PENDING_RETRIES = 10


def validate_config() -> None:
    """Sanity-check the tunables above for internal consistency. Raises
    `ValueError` on the first problem found - meant to be called once at
    process startup, not on a hot path.
    """
    import re

    if MAX_RETRIES < 0:
        raise ValueError("MAX_RETRIES must be >= 0")
    if RETRY_BACKOFF_BASE_S <= 0:
        raise ValueError("RETRY_BACKOFF_BASE_S must be > 0")
    if RETRY_BACKOFF_MAX_S < RETRY_BACKOFF_BASE_S:
        raise ValueError("RETRY_BACKOFF_MAX_S must be >= RETRY_BACKOFF_BASE_S")
    if not REQUIRED_ORDER_FIELDS:
        raise ValueError("REQUIRED_ORDER_FIELDS must not be empty")
    if MAX_QTY_PER_ORDER <= 0:
        raise ValueError("MAX_QTY_PER_ORDER must be > 0")
    try:
        re.compile(SKU_PATTERN)
    except re.error as exc:
        raise ValueError(f"SKU_PATTERN does not compile: {exc}") from exc
    if INBOX_MAX_SIZE <= 0:
        raise ValueError("INBOX_MAX_SIZE must be > 0")
    if not ENRICHMENT_PLUGIN:
        raise ValueError("ENRICHMENT_PLUGIN must name a plugin module")
