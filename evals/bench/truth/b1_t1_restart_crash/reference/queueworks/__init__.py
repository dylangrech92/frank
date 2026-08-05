"""queueworks - a small, dependency-free job queue library.

queueworks lets an application enqueue units of work ("jobs"), run them
through a pool of workers, retry failures with exponential backoff, and
persist queue state to disk so a process can resume after a restart.

The package is intentionally dependency-free (standard library only) so it
can be dropped into any Python 3.11+ project without a package manager.

Modules:
    models      -- Job, Priority, JobStatus, CompletionRecord
    registry    -- name -> callable task registry
    store       -- JSON-backed persistent job store
    queue       -- priority queue built on heapq
    retry       -- retry policy (backoff schedule, retry decision)
    worker      -- worker pool that executes jobs and drives retries
    manager     -- high-level facade wiring store + queue + workers together
    config      -- QueueConfig, loaded from QUEUEWORKS_* environment variables
    queries     -- read-only helpers for slicing a job store's contents
    stats       -- summary statistics over completed jobs
    reporting   -- human-readable reports over completed jobs
    sample_tasks -- example task functions used by the demo and CLI
    demo        -- runnable scenarios (``python3 -m queueworks.demo``)
    cli         -- everyday operations (``python3 -m queueworks.cli``)
"""

__version__ = "0.3.0"

from .models import Job, JobStatus, Priority

__all__ = ["Job", "JobStatus", "Priority", "__version__"]
