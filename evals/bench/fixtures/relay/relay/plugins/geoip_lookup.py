"""Prototype geo-enrichment helpers. Not wired into the plugin loader
configuration; kept around from an early spike."""
from __future__ import annotations


def normalize_event(payload: dict) -> dict:
    """Lowercase and trim string fields in a raw geo payload."""
    return {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in payload.items()}


def lookup_region(ip_address: str) -> str:
    """Resolve an IP address to a shipping region. No GeoIP database is
    bundled with this project, so this always returns "unknown"."""
    return "unknown"
