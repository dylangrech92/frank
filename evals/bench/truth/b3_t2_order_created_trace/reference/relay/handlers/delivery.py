"""Final-mile helpers for the deliver stage."""
from __future__ import annotations

from relay import config
from relay.compat.legacy import clean_event
from relay.errors import DeliveryError
from relay.sinks import file_sink, webhook_sink

SINK_DISPATCH = {
    "file": file_sink.send_to_file_sink,
    "webhook": webhook_sink.send_to_webhook_sink,
}


def run_deliver(event) -> None:
    """Pre-refactor deliver-stage entry point, from before the pipeline
    moved to `STAGE_DISPATCH`-driven stage lookup in the orchestrator.
    Superseded by `relay.pipeline.orchestrator.run_deliver`; nothing in
    the current pipeline imports or calls this copy."""
    finalize_and_send(event)


def finalize_and_send(event) -> None:
    """Run the last normalization pass, then hand `event` to whichever
    sink `config.DELIVERY_SINK` selects.

    Raises `DeliveryError` if `config.DELIVERY_SINK` doesn't name a
    known sink.
    """
    clean_event(event)
    send = SINK_DISPATCH.get(config.DELIVERY_SINK)
    if send is None:
        raise DeliveryError(f"no such delivery sink: {config.DELIVERY_SINK!r}")
    send(event)
