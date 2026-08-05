"""Example task functions registered for use by the demo scenarios.

Real applications register their own task functions the same way -- import
this module (or your own equivalent) before starting any workers so the
names referenced by persisted jobs can be resolved.
"""

from __future__ import annotations

import time

from . import registry


@registry.task("noop")
def noop(*args, **kwargs):
    """Succeed immediately without doing any work."""

    return "ok"


@registry.task("timed_work")
def timed_work(duration: float, label: str = ""):
    """Busy-work that takes approximately ``duration`` seconds."""

    time.sleep(duration)
    return {"label": label, "duration": duration}


@registry.task("always_fails")
def always_fails(message: str = "simulated failure"):
    """A task that always raises, used to exercise the retry policy."""

    raise RuntimeError(message)


@registry.task("send_notification")
def send_notification(recipient: str, subject: str, body: str = ""):
    """Simulate delivering a notification to ``recipient``.

    Real deployments would swap this for an actual mail/SMS/webhook client;
    it exists here so demos and tests have a task with more than one
    required argument to enqueue.
    """

    if not recipient:
        raise ValueError("recipient must be non-empty")
    return {"recipient": recipient, "subject": subject, "body_length": len(body)}


@registry.task("cleanup_tmp_files")
def cleanup_tmp_files(directory: str, max_age_seconds: float = 3600.0):
    """Simulate sweeping stale files out of ``directory``.

    Reports what it *would* remove rather than touching the filesystem, so
    it is safe to enqueue in demos without side effects outside the queue's
    own state.
    """

    now = time.time()
    cutoff = now - max_age_seconds
    return {"directory": directory, "swept_before": cutoff, "removed": 0}
