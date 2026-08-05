"""The fixed processing pipeline: every event moves through the same
five stages in the same order, defined by `STAGE_ORDER` and
`STAGE_DISPATCH`.
"""
from __future__ import annotations

import logging

from relay import config
from relay.bus import bus
from relay.errors import RelayError
from relay.events import EVENT_ORDER_CREATED, EVENT_ORDER_ENRICHED, EVENT_ORDER_VALIDATED, Event
from relay.handlers import audit, delivery, ingest, validate  # noqa: F401 - audit registers on import
from relay.pipeline import retry
from relay.pipeline.metrics import metrics
from relay.pipeline.scheduler import scheduler
from relay.plugins import loader
from relay.sinks import deadletter
from relay.storage.state import StateStore

logger = logging.getLogger(__name__)

state = StateStore()


def run_ingest(raw: dict) -> Event:
    with metrics.timer("stage.ingest"):
        return ingest.build_order_event(raw)


def run_transform(event: Event) -> Event:
    with metrics.timer("stage.transform"):
        bus.publish(EVENT_ORDER_CREATED, event)
    return event


def run_validate(event: Event) -> Event:
    with metrics.timer("stage.validate"):
        validate.validate_order(event)
        bus.publish(EVENT_ORDER_VALIDATED, event)
    return event


def run_enrich(event: Event) -> Event:
    with metrics.timer("stage.enrich"):
        plugin = loader.load_plugin(config.ENRICHMENT_PLUGIN)
        plugin.enrich(event)
    bus.publish(EVENT_ORDER_ENRICHED, event)
    return event


def run_deliver(event: Event) -> Event:
    with metrics.timer("stage.deliver"):
        delivery.finalize_and_send(event)
    return event


STAGE_ORDER = ["ingest", "transform", "validate", "enrich", "deliver"]

STAGE_DISPATCH = {
    "ingest": run_ingest,
    "transform": run_transform,
    "validate": run_validate,
    "enrich": run_enrich,
    "deliver": run_deliver,
}


def run_pipeline(raw):
    """Push one raw payload through every stage in `STAGE_ORDER`.

    `raw` may be a plain dict (a brand new event, run from `ingest`
    onward) or an `Event` already in flight (a retry, resumed from
    `transform` onward since only `ingest` needs the original raw dict).

    Returns the final `Event` on success, or `None` if the event was
    dead-lettered - either immediately, or after its retries were
    exhausted.
    """
    if isinstance(raw, Event):
        event = raw
        stages = STAGE_ORDER[1:]
    else:
        event = STAGE_DISPATCH["ingest"](raw)
        stages = STAGE_ORDER[1:]

    for stage_name in stages:
        stage_fn = STAGE_DISPATCH[stage_name]
        try:
            event = stage_fn(event)
        except RelayError as exc:
            if retry.should_retry(exc, event.attempts):
                event.bump_attempt()
                delay = retry.backoff_for(event.attempts)
                scheduler.schedule_retry(event, run_pipeline, delay)
                state.mark_retried()
                logger.info(
                    "scheduled retry %s/%s for %s after %s",
                    event.attempts,
                    config.MAX_RETRIES,
                    event.event_id,
                    type(exc).__name__,
                )
            else:
                deadletter.write_dead_letter(event, exc)
                state.mark_dead_lettered()
            return None

    state.mark_processed(event.event_id)
    return event


def stage_metrics_snapshot() -> dict[str, object]:
    """Return the current per-stage timing/count snapshot, for the CLI
    `metrics` command."""
    return metrics.snapshot()
