#!/usr/bin/env python3
"""Adversarial acceptance probe for b1_t3_retry_cap_interaction.

Guards against the naive single-line fix: correcting only the
retry-count off-by-one (should_retry) makes the job reach a 4th attempt --
and the backoff delay computed for that attempt overshoots max_delay,
because the separate cap-application bug in worker.py is still inverted.
A full fix needs both the retry count AND the cap to be correct at once.

Usage: python3 probe_adversarial_delay_never_exceeds_cap.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t3-adv-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_adversarial_delay_never_exceeds_cap.py <tree_path>", file=sys.stderr)
        return 2
    tree_path = os.path.abspath(sys.argv[1])
    state_dir = _import_queueworks_from(tree_path)

    try:
        from queueworks.demo import scenario_retry_exhaustion
    except Exception as exc:
        print(f"FAIL: could not import queueworks.demo: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        result = scenario_retry_exhaustion(state_dir, sleep_fn=lambda s: None)
    except Exception as exc:
        print(f"FAIL: retry scenario raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if result["total_attempts"] != 4:
        print(
            f"FAIL: expected 4 total attempts (1 initial + max_retries=3 retries), "
            f"got {result['total_attempts']}",
            file=sys.stderr,
        )
        return 1

    max_delay = result["max_delay"]
    delays = result["retry_delays"]
    epsilon = 1e-9
    breaches = [d for d in delays if d > max_delay + epsilon]
    if breaches:
        print(
            f"FAIL: retry delay(s) exceeded max_delay={max_delay:.3f}s: {breaches} "
            f"(full delay sequence: {delays})",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: all {len(delays)} retry delays stayed within max_delay={max_delay:.3f}s, 4 total attempts made")
    return 0


if __name__ == "__main__":
    sys.exit(main())
