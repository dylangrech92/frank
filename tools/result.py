"""Frozen result contract returned by every tool."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A frozen, slot-backed result contract with strict validation.

    Use the classmethods :meth:`ok` and :meth:`err` to construct instances.
    """

    status: str  # 'success' or 'error'
    body: str | dict | list  # payload
    meta: dict[str, object] = field(default_factory=dict)  # scalar key-value pairs
    code: str | None = None  # kebab-case error code (required on error only)
    hint: str | None = None  # optional recovery hint

    def __post_init__(self) -> None:
        if self.status not in ("success", "error"):
            raise ValueError(
                f"status must be 'success' or 'error', got {self.status!r}"
            )
        if not isinstance(self.body, (str, dict, list)):
            raise TypeError(
                f"body must be str | dict | list, got {type(self.body).__name__}"
            )
        for k, v in self.meta.items():
            if not isinstance(k, str):
                raise TypeError(f"meta key must be str, got {k!r}")
            if not isinstance(v, (str, int, float, bool)):
                raise TypeError(
                    f"meta value '{v!r}' for key '{k}' "
                    f"must be a scalar (str/int/float/bool)"
                )
        if self.status == "error" and not self.code:
            raise ValueError("code is required when status is 'error'")
        if self.status != "error" and self.code is not None:
            raise ValueError("code must not be set when status is 'success'")
        if self.code is not None:
            if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", self.code):
                raise ValueError(f"code must be kebab-case, got {self.code!r}")

    @classmethod
    def ok(cls, body: str | dict | list, **meta: object) -> ToolResult:
        """Return a success ToolResult.

        Args:
            body: The payload (str / dict / list).
            **meta: Metadata key-value pairs (values must be str/int/float/bool).
        """
        return cls(status="success", body=body, meta={**meta})

    @classmethod
    def err(
        cls, message: str, *, code: str, hint: str | None = None, **meta: object
    ) -> ToolResult:
        """Return an error ToolResult.

        Args:
            message: The error body text.
            code: A kebab-case machine-readable error code (required).
            hint: Optional recovery hint for the caller.
            **meta: Metadata key-value pairs (values must be str/int/float/bool).
        """
        return cls(status="error", body=message, code=code, hint=hint, meta={**meta})


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """Return ``(text, False)`` when ``len(text) <= limit``, else ``(text[:limit], True)``.

    Args:
        text: The text to possibly truncate.
        limit: Maximum allowed length (must be >= 0).

    Raises:
        ValueError: If *limit* is negative.
    """
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    if len(text) <= limit:
        return text, False
    return text[:limit], True
