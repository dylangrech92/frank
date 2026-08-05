#!/usr/bin/env python3
"""Adversarial acceptance probe for b1_t3_retry_cap_interaction.

The other adversarial probe (probe_adversarial_delay_never_exceeds_cap.py)
replays the exact repro's RetryPolicy constants (max_retries=3,
base_delay=0.05, backoff_factor=4.0, max_delay=1.0). A fix built by staring
at that one printed transcript can get away with matching those specific
numbers -- e.g. hard-coding a stop condition around "3 attempts" or a cap
literal around "1.0" -- without actually reading the count from
self.max_retries or the ceiling from self.retry_policy.max_delay.

This probe drives the same two-bug interaction through a *different*
RetryPolicy the repro never shows, so a fix has to be the general one
(attempts <= self.max_retries; delay = min(raw_delay, self.retry_policy.max_delay))
rather than a fit to the specific numbers on screen.

Usage: python3 probe_adversarial_generalized_policy.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t3-adv-gen-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_adversarial_generalized_policy.py <tree_path>", file=sys.stderr)
        return 2
    tree_path = os.path.abspath(sys.argv[1])
    state_dir = _import_queueworks_from(tree_path)

    try:
        from queueworks.manager import QueueManager
        from queueworks.models import Priority
        from queueworks.retry import RetryPolicy
        from queueworks.worker import Worker
        from queueworks import sample_tasks  # noqa: F401 -- registers "always_fails"
    except Exception as exc:
        print(f"FAIL: could not import queueworks: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    state_path = os.path.join(state_dir, "queue_state.json")
    # Deliberately different from the repro's policy (max_retries=3,
    # base_delay=0.05, backoff_factor=4.0, max_delay=1.0) so a fix hard-coded
    # to those specific numbers fails here even though it would pass a probe
    # that only replays the repro's own constants.
    policy = RetryPolicy(max_retries=5, base_delay=0.01, backoff_factor=6.0, max_delay=0.3)

    try:
        manager = QueueManager(state_path, retry_policy=policy)
        job = manager.enqueue(
            "always_fails",
            kwargs={"message": "simulated failure"},
            priority=Priority.NORMAL,
        )
        worker = Worker(manager.queue, manager.store, retry_policy=policy, sleep_fn=lambda s: None)
        worker.run_one()
        finished = manager.store.get(job.id)
    except Exception as exc:
        print(f"FAIL: retry scenario raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    expected_attempts = policy.max_retries + 1
    if finished.attempts != expected_attempts:
        print(
            f"FAIL: expected {expected_attempts} total attempts (1 initial + "
            f"max_retries={policy.max_retries} retries) under a non-default "
            f"RetryPolicy, got {finished.attempts}",
            file=sys.stderr,
        )
        return 1

    epsilon = 1e-9
    breaches = [d for d in finished.retry_delays if d > policy.max_delay + epsilon]
    if breaches:
        print(
            f"FAIL: retry delay(s) exceeded max_delay={policy.max_delay:.3f}s under a "
            f"non-default RetryPolicy: {breaches} (full sequence: {finished.retry_delays})",
            file=sys.stderr,
        )
        return 1

    if finished.status.value != "failed":
        print(f"FAIL: expected final status 'failed', got {finished.status.value!r}", file=sys.stderr)
        return 1

    print(
        f"PASS: {finished.attempts} total attempts and all "
        f"{len(finished.retry_delays)} delays stayed within max_delay="
        f"{policy.max_delay:.3f}s under a non-default RetryPolicy"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
