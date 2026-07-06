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
    summary: str  # one-line description shown in the catalog (before a tool is loaded)
    description: str  # full LLM-facing description (shown once the tool is loaded)
    parameters: dict  # JSON Schema describing arguments

    # Optional attributes used to render an oversize-result error (see
    # registry.dispatch's result-too-large guard) and the repeated-identical-
    # failure loop-guard.  Subclasses may override; sensible defaults apply
    # otherwise.
    action: str = "complete the operation"  # short verb phrase, e.g. "read the file"
    oversize_hint: str = "narrow the request or use a more specific tool"
    alternative: str = "a different tool or approach"  # suggested when this tool keeps failing

    # True only for tools that neither mutate any state nor touch a resource
    # that is unsafe off the main thread (LSP document sync, the main-thread
    # SQLite connection, process handles).  When every call in an assistant
    # batch is parallel_safe, the agent loop dispatches the batch concurrently.
    parallel_safe: bool = False

    @abstractmethod
    def run(self, **kwargs: object) -> ToolResult:
        """Execute the tool with the given keyword arguments.

        Args:
            **kwargs: Parsed and validated from the LLM function-call payload.

        Returns:
            A ``ToolResult`` representing success or failure.
        """
        ...
