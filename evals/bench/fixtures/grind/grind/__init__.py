"""grind: a small batch pipeline that turns raw access-log style events into
a per-region/per-status summary report.

Stages: generate -> parse/validate -> normalize -> dedup -> enrich -> aggregate.
See workload.py for the command-line entry point.
"""

__version__ = "0.4.0"
