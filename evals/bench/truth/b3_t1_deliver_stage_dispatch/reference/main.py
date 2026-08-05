"""Command-line entry point and demo runner for the relay pipeline.

Usage:
    python3 main.py demo                 run the bundled sample events
    python3 main.py quarantine <path>    manually dead-letter one raw
                                          event read from a JSON file
    python3 main.py replay               re-run every dead-lettered
                                          event through the pipeline
    python3 main.py stats                print processed/dead-letter
                                          counters
    python3 main.py metrics              print per-stage timing/counts
    python3 main.py report               print a combined operator
                                          report (counters, slowest
                                          stages, pending retries)
    python3 main.py plugins              list enrichment plugins that
                                          could be loaded by name
    python3 main.py health                compare current backlogs
                                          against configured thresholds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relay import config, reporting
from relay.errors import ManualQuarantineError
from relay.events import EVENT_ORDER_CREATED, Event
from relay.pipeline import health
from relay.pipeline.orchestrator import run_pipeline, stage_metrics_snapshot, state
from relay.pipeline.scheduler import scheduler
from relay.plugins import loader
from relay.sinks import deadletter
from relay.storage.queue import InMemoryQueue, QueueFullError
from relay.utils.ids import generate_event_id, short_id
from relay.utils.logging_utils import configure_logging

SAMPLE_EVENTS_PATH = Path(__file__).parent / "sample_events.jsonl"


def run_demo(sample_path: Path = SAMPLE_EVENTS_PATH, settle_rounds: int = 5) -> None:
    """Feed every line of `sample_path` through the pipeline, then drain
    any retries that were scheduled along the way."""
    inbox: InMemoryQueue = InMemoryQueue(max_size=config.INBOX_MAX_SIZE)
    with open(sample_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            incoming = json.loads(line)
            try:
                inbox.push(incoming)
            except QueueFullError as exc:
                # The inbox is a bounded buffer; a raw payload that
                # arrives when it's already full has nowhere to wait, so
                # it is dead-lettered directly instead of being dropped
                # silently.
                overflow_event = Event(event_id=generate_event_id(), name=EVENT_ORDER_CREATED, payload=incoming)
                deadletter.write_dead_letter(overflow_event, exc)
                state.mark_dead_lettered()

    raw = inbox.pop()
    while raw is not None:
        result = run_pipeline(raw)
        if result is not None:
            print(f"delivered {short_id(result.event_id)}")
        else:
            print("dead-lettered or scheduled for retry")
        raw = inbox.pop()

    for _ in range(settle_rounds):
        if scheduler.drain_ready() == 0:
            break
    expired = scheduler.expire_stale(config.RETRY_STALE_TIMEOUT_S)
    if expired:
        print(f"expired {expired} stale retr{'y' if expired == 1 else 'ies'}")

    counters = state.counters()
    print(f"processed: {counters['delivered']} delivered, {counters['dead_lettered']} dead-lettered")


def run_quarantine(raw_path: str) -> None:
    """Manually dead-letter one raw event, bypassing the pipeline
    entirely."""
    with open(raw_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    event = Event(event_id=generate_event_id(), name="order.created", payload=raw)
    deadletter.write_dead_letter(event, ManualQuarantineError("quarantined by operator"))
    print(f"quarantined {event.event_id}")


def run_replay() -> None:
    """Re-run every currently dead-lettered event through the pipeline
    from scratch (as a brand-new raw payload, not a resumed `Event`, so
    it goes through `ingest` again with a fresh id).

    The records are snapshotted into a list before any of them are
    replayed - a repeat failure appends a new record to the same file
    `read_dead_letters` reads from, and processing that append as part
    of the same pass would replay forever.
    """
    if not config.REPLAY_ENABLED:
        print("replay is disabled")
        return
    records = list(deadletter.read_dead_letters())
    replayed = 0
    for record in records:
        result = run_pipeline(dict(record["payload"]))
        replayed += 1
        outcome = "delivered" if result is not None else "dead-lettered again"
        print(f"replay {short_id(record['event_id'])}: {outcome}")
    print(f"replayed {replayed} event(s)")


def run_stats() -> None:
    counters = state.counters()
    print(f"delivered: {counters['delivered']}")
    print(f"dead_lettered: {counters['dead_lettered']}")
    print(f"retried: {counters['retried']}")
    print(f"dead_letter_file_count: {deadletter.count_dead_letters()}")


def run_metrics() -> None:
    snapshot = stage_metrics_snapshot()
    print(json.dumps(snapshot, indent=2, sort_keys=True))


def run_report() -> None:
    print(reporting.build_report(state))


def run_plugins() -> None:
    for name in loader.list_available_plugin_names():
        marker = " (active)" if name == config.ENRICHMENT_PLUGIN else ""
        print(f"{name}{marker}")


def run_health() -> None:
    for check in health.run_health_checks():
        print(check)


def main(argv: list[str] | None = None) -> int:
    config.validate_config()
    configure_logging()
    parser = argparse.ArgumentParser(prog="relay")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    quarantine_parser = sub.add_parser("quarantine")
    quarantine_parser.add_argument("path")
    sub.add_parser("replay")
    sub.add_parser("stats")
    sub.add_parser("metrics")
    sub.add_parser("report")
    sub.add_parser("plugins")
    sub.add_parser("health")

    args = parser.parse_args(argv)
    if args.command == "demo":
        run_demo()
    elif args.command == "quarantine":
        run_quarantine(args.path)
    elif args.command == "replay":
        run_replay()
    elif args.command == "stats":
        run_stats()
    elif args.command == "metrics":
        run_metrics()
    elif args.command == "report":
        run_report()
    elif args.command == "plugins":
        run_plugins()
    elif args.command == "health":
        run_health()
    return 0


if __name__ == "__main__":
    sys.exit(main())
