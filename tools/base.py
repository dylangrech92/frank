"""Abstract base class (and class-level attribute protocol) for every tool."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tools.result import ToolResult


class Tool(ABC):
    """A single discoverable CLI tool.

    Subclasses must declare the class attributes ``name``, ``description``,
    and ``parameters`` (a JSON Schema dict).  The only entry point is
    :meth:`run`.
    """

    name: str  # unique tool name, set by subclass
    description: str  # LLM-facing description
    parameters: dict  # JSON Schema describing arguments

    @abstractmethod
    def run(self, **kwargs: object) -> ToolResult:
        """Execute the tool with the given keyword arguments.

        Args:
            **kwargs: Parsed and validated from the LLM function-call payload.

        Returns:
            A ``ToolResult`` representing success or failure.
        """
        ...
