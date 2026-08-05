# queueworks

A small, dependency-free job queue for Python 3.11+. queueworks lets an
application enqueue units of work, run them through a pool of workers,
retry failures with exponential backoff, and persist queue state to disk so
a process can resume where it left off after a restart.

No third-party dependencies, no network access required -- everything is
built on the standard library.

## Features

- **Priorities.** Jobs are scheduled `CRITICAL` > `HIGH` > `NORMAL` > `LOW`,
  FIFO within a priority band.
- **Retries with backoff.** A configurable `RetryPolicy` controls how many
  times a failing job is retried and how long each retry waits.
- **Durability.** Queue state is mirrored to a JSON file on disk after every
  mutation, so a process can restart and pick its pending jobs back up.
- **Workers.** Run jobs synchronously in-process (`QueueManager.run_pending`)
  or on a background thread pool (`WorkerPool`).
- **Stats & reporting.** Lightweight helpers summarize recently-completed
  jobs for dashboards and operational reports.
- **CLI.** `python3 -m queueworks.cli` covers everyday operations --
  enqueue, run, status, query, purge, stats -- against a state file of your
  choosing.

## Quick start

```python
from queueworks.manager import QueueManager
from queueworks.models import Priority
from queueworks import sample_tasks  # registers example tasks; register your own the same way

manager = QueueManager("queue_state.json")
manager.enqueue("noop", priority=Priority.HIGH)
manager.run_pending()
```

Tasks are registered by name rather than passed around as callables, so a
persisted job can be reloaded and re-run by a fresh process without
depending on pickling:

```python
from queueworks import registry

@registry.task("send_welcome_email")
def send_welcome_email(user_id: int):
    ...
```

## Retry semantics

`RetryPolicy.max_retries` is the number of retries allowed **after** the
initial attempt. With the default of 3, a job that keeps failing is
attempted 4 times in total (1 initial attempt + 3 retries) before it is
marked permanently `FAILED`. Each retry waits an exponentially increasing
delay, starting at `base_delay` and multiplying by `backoff_factor` per
attempt, capped at `max_delay`.

## Demo scenarios

Three runnable scenarios live in `queueworks/demo.py` and double as a quick
smoke test of the library:

```
python3 -m queueworks.demo restart     # enqueue, process, restart, enqueue more
python3 -m queueworks.demo dashboard   # render the stats + reporting panels
python3 -m queueworks.demo retries     # run a task that always fails
```

## Command-line interface

For everyday operations against a real state file, use `queueworks.cli`
instead of writing a script:

```
python3 -m queueworks.cli enqueue send_notification \
    --kwargs '{"recipient": "ops@example.com", "subject": "hi"}'
python3 -m queueworks.cli run
python3 -m queueworks.cli status
python3 -m queueworks.cli query --status failed
python3 -m queueworks.cli stats --chronological
python3 -m queueworks.cli purge
```

The state file defaults to `queueworks_state.json` in the current
directory; override it with `--state-path` or the `QUEUEWORKS_STATE_PATH`
environment variable. Retry tuning can likewise be set via
`QUEUEWORKS_MAX_RETRIES`, `QUEUEWORKS_BASE_DELAY`,
`QUEUEWORKS_BACKOFF_FACTOR`, and `QUEUEWORKS_MAX_DELAY` (see
`queueworks/config.py`).

## Project layout

```
queueworks/
  models.py        Job, Priority, JobStatus, CompletionRecord
  registry.py       name -> task callable registry
  store.py          JSON-backed persistent job store + recent-completions feed
  queue.py          in-memory priority queue (heapq)
  retry.py          retry policy: attempt counting and backoff
  worker.py         Worker / WorkerPool: execute jobs, drive retries
  manager.py        QueueManager facade tying store + queue + workers together
  config.py         QueueConfig, loaded from QUEUEWORKS_* environment variables
  queries.py        read-only helpers for slicing a job store's contents
  stats.py          summary statistics over recently completed jobs
  reporting.py      human-readable reports over recently completed jobs
  sample_tasks.py   example task functions used by the demo and CLI
  demo.py           runnable demo scenarios
  cli.py            everyday operations (python3 -m queueworks.cli)
```

## Running the test suite

This project doesn't ship its own test suite; it's a small internal
library maintained alongside the services that depend on it.
