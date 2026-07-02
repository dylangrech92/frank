"""Tool discovery, schema generation, and dispatch for the tools/ package."""

from __future__ import annotations

import inspect
import pkgutil
import sys
from importlib import import_module
from typing import Any

from tools.base import Tool
from tools.result import ToolResult

# populated by discover() — one instance of each concrete tool class
_registry: dict[str, "Tool"] = {}


def _is_concrete_tool(cls: type) -> bool:
    """Return True if *cls* is a direct subclass with ``name`` and ``parameters`` defined on it."""
    return (
        inspect.isclass(cls)
        and issubclass(cls, Tool)
        and cls is not Tool
        and not inspect.isabstract(cls)
        and "name" in cls.__dict__
        and "parameters" in cls.__dict__
    )


def discover() -> None:
    r"""Scan tools/ for concrete Tool subclasses and populate ``_registry``.

    Ignores *_-*-prefixed modules and base, registry, result.  Calling a second
    time is safely a no-op (idempotent).
    """
    if _registry:
        return

    for importer, modname, ispkg in pkgutil.iter_modules(sys.modules[__package__].__path__):
        if modname.startswith("_") or modname in ("base", "registry", "result"):
            continue

        # Import tools.modname relative to the package
        mod = import_module(f"{sys.modules[__package__].__name__}.{modname}")

        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if not _is_concrete_tool(cls):
                continue
            instance = cls()  # type: ignore[arg-type]
            dup = _registry.get(instance.name)
            if dup is not None:
                raise ValueError(
                    f"duplicate tool name {instance.name!r} "
                    f"({_registry[instance.name].__module__} and {mod.__name__})"
                )
            _registry[instance.name] = instance


def schemas() -> list[dict[str, Any]]:
    """Return the OpenAI Chat Completions tools array.

    One entry per registered tool with ``type=``function` and a ``function`` key
    carrying ``name``, ``description``, and ``parameters``.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in _registry.values()
    ]


def dispatch(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Look up tool *name*, call ``run(**arguments)``, and return the result.

    All exceptions from an unknown or a crashed tool are captured as errors;
    nothing escapes this function.

    Args:
        name: Registered tool name.
        arguments: Keyword arguments forwarded to ``tool.run()``.

    Returns:
        A ``ToolResult`` (ok or err).
    """
    if not _registry:  # auto-discover on first dispatch call
        discover()

    tool = _registry.get(name)
    if tool is None:
        valid = sorted(_registry)
        hint = f"Valid tools: {', '.join(valid)}" if valid else "None registered"
        return ToolResult.err(
            f"unknown tool: {name!r}",
            code="unknown-tool",
            hint=hint,
        )

    # run every tool -- call and catch exceptions
    try:
        result = tool.run(**arguments)  # type: ignore[operator]
    except TypeError as exc:
        return ToolResult.err(
            f"{tool.name}: bad arguments — {exc}",
            code="bad-arguments",
        )
    except Exception as exc:
        return ToolResult.err(
            f"{tool.name}: crashed ({type(exc).__name__} '{str(exc)}')",
            code="tool-crashed",
        )

    return result  # type: ignore[return-value]
