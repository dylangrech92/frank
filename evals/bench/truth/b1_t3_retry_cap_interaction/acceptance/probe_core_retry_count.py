#!/usr/bin/env python3
"""Core acceptance probe for b1_t3_retry_cap_interaction.

The reported bug: with max_retries=3, a job whose task always raises is
marked permanently FAILED after only 3 total attempts instead of 4 (1
initial attempt + 3 retries, per the documented retry semantics).

Usage: python3 probe_core_retry_count.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t3-core-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_core_retry_count.py <tree_path>", file=sys.stderr)
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

    if result["final_status"] != "failed":
        print(f"FAIL: expected final status 'failed', got {result['final_status']!r}", file=sys.stderr)
        return 1

    if result["total_attempts"] != 4:
        print(
            f"FAIL: expected 4 total attempts (1 initial + max_retries=3 retries), "
            f"got {result['total_attempts']}",
            file=sys.stderr,
        )
        return 1

    print("PASS: job was attempted 4 times (1 initial + 3 retries) before failing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
