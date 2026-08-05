#!/usr/bin/env python3
"""Edge acceptance probe for b1_t3_retry_cap_interaction.

The first retry's backoff delay should be close to its true uncapped value
(base_delay * backoff_factor = 0.05 * 4 = 0.2s), not needlessly maxed out
to max_delay (1.0s). This is a distinct defect from the retry-count bug --
fixing only the retry count leaves every early delay pinned at max_delay.

Usage: python3 probe_edge_first_delay_not_wasted.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t3-edge-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_edge_first_delay_not_wasted.py <tree_path>", file=sys.stderr)
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

    delays = result["retry_delays"]
    if not delays:
        print("FAIL: no retry delays were recorded at all", file=sys.stderr)
        return 1

    first_delay = delays[0]
    # True uncapped value is base_delay(0.05) * backoff_factor(4.0) ** 1 = 0.2s.
    # A generous threshold well below max_delay(1.0s) but comfortably above
    # 0.2s, so ordinary float noise never causes a false result either way.
    threshold = 0.5
    if first_delay >= threshold:
        print(
            f"FAIL: first retry delay was {first_delay:.3f}s, expected well under {threshold}s "
            "(true uncapped value is ~0.2s) -- the delay is being needlessly maxed out",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: first retry delay ({first_delay:.3f}s) reflects its true backoff value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
