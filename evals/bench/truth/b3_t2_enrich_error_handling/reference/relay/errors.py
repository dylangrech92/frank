"""Exception hierarchy for the relay pipeline.

Every stage-level failure is a `RelayError` subclass so the orchestrator's
error handling can catch one base type regardless of which stage raised.
"""


class RelayError(Exception):
    """Base class for all relay pipeline failures."""


class ValidationError(RelayError):
    """Raised when an event's payload fails schema validation before it
    is allowed further into the pipeline."""


class EnrichmentError(RelayError):
    """Raised when the configured enrichment plugin cannot complete for
    a given event."""


class PluginLoadError(RelayError):
    """Raised when the configured enrichment plugin module cannot be
    imported, or does not expose a usable entry point."""


class DeliveryError(RelayError):
    """Raised when a sink fails to persist an event."""


class ManualQuarantineError(RelayError):
    """Attached to events an operator dead-letters directly via the CLI
    `quarantine` command, outside of normal pipeline processing."""


class RetryExpiredError(RelayError):
    """Attached to an event whose scheduled retry sat pending longer than
    the scheduler's stale-retry timeout, and was given up on before its
    delay ever elapsed."""
