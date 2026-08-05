#!/usr/bin/env python3
"""CLI entry point for the grind batch pipeline.

Usage:
    python3 workload.py --scale N

Generates a synthetic batch of N * 3200 access-log-style events, runs it
through parsing/validation, canonicalization, duplicate detection, and
client-profile enrichment, then prints a per-region summary report as
JSON. The same scale always produces the same report: input generation is
seeded from the scale alone, with no dependency on wall-clock time, the
filesystem, or the environment.
"""

from __future__ import annotations

import argparse
import json
import sys

from grind.pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the grind batch pipeline.")
    parser.add_argument(
        "--scale",
        type=int,
        required=True,
        help="workload scale factor; record volume grows linearly with this",
    )
    args = parser.parse_args(argv)

    if args.scale < 1:
        print("scale must be >= 1", file=sys.stderr)
        return 2

    report = run(args.scale)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
