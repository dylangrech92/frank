"""Build the per-region summary report."""

from __future__ import annotations

from . import config
from .records import ParsedEvent


def build_report(
    events: list[ParsedEvent],
    rejected_count: int,
    duplicate_count: int,
    cache_size: int,
) -> dict:
    region_totals = {
        region: {"total": 0, "duplicates": 0, "bytes": 0, "statuses": {}}
        for region in config.REGIONS
    }

    for event in events:
        bucket = region_totals[event.region]
        bucket["total"] += 1
        bucket["bytes"] += event.bytes_sent
        if event.is_duplicate:
            bucket["duplicates"] += 1
        bucket["statuses"][event.status] = bucket["statuses"].get(event.status, 0) + 1

    regions_report = []
    for region in config.REGIONS:
        bucket = region_totals[region]
        regions_report.append(
            {
                "region": region,
                "total": bucket["total"],
                "duplicates": bucket["duplicates"],
                "bytes": bucket["bytes"],
                "statuses": sorted(bucket["statuses"].items()),
            }
        )

    return {
        "records_parsed": len(events),
        "records_rejected": rejected_count,
        "duplicates_flagged": duplicate_count,
        "profile_cache_entries": cache_size,
        "regions": regions_report,
    }
