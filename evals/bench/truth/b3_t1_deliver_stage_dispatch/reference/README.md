# relay

A small in-process event pipeline for processing incoming order events:
ingest -> transform -> validate -> enrich -> deliver.

## Requirements

Python 3.11+, standard library only. Nothing to install.

## Running the demo

    python3 main.py demo

Reads `sample_events.jsonl`, pushes each line through the pipeline, and
prints one line per event describing the outcome. Successfully delivered
events are appended to `delivered.jsonl`; permanently failed events are
appended to `deadletters.jsonl`.

## Other commands

    python3 main.py quarantine <path-to-raw-event.json>
    python3 main.py stats

`quarantine` records a raw event as permanently failed without running it
through the pipeline, for operators who have already decided by hand that
an event cannot be recovered. `stats` prints how many events have been
processed in the current run.

## Layout

- `relay/events.py` - the `Event` type and the canonical event-name
  constants.
- `relay/errors.py` - the exception hierarchy shared by every stage.
- `relay/bus.py` - the in-process publish/subscribe event bus.
- `relay/handlers/` - per-event and per-stage handler functions.
- `relay/pipeline/` - the stage pipeline driver, retry policy, and the
  retry scheduler.
- `relay/plugins/` - enrichment plugins, selected by name at runtime.
- `relay/sinks/` - where processed events end up: delivered output,
  operational logs, dead letters.
- `relay/compat/` - aliases for integrations written against the pre-2.0
  API. New code should not import from here.
- `relay/storage/` - in-memory queueing and processed-event tracking.
- `relay/utils/` - id generation and logging setup.

## Configuration

Runtime tunables live in `relay/config.py` as plain module attributes
(no environment/config-service dependency, to keep the demo
self-contained).
