"""Tool discovery, schema generation, and dispatch for the tools/ package."""

from __future__ import annotations

import inspect
import pkgutil
import sys
from importlib import import_module
from typing import Any

from tools.base import Tool
from tools.result import ToolResult

_PY_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_arguments(tool: Tool, arguments: dict[str, Any]) -> str | None:
    """Validate *arguments* against the tool's JSON Schema ``parameters``.

    Checks three things:

    1. **Unknown keys rejected** — every key in *arguments* must appear in the
       schema's ``properties`` mapping.  Unknown keys produce a message naming
       the bad key followed by `` (allowed: ...)`` with the allowed parameter
       names listed.

    2. **Required keys checked** — every name in the schema's ``required`` list
       must be present in *arguments*.  Missing ones produce a message listing
       all the missing keys and their declared types.

    3. **Type checking** — each provided value is checked against the declared
       JSON Schema ``type`` for that property, using the mapping below::

          string       → str
          integer      → int (rejects bool)
          number       → int or float (rejects bool)
          boolean      → bool
          array        → list
          object       → dict

       Mismatches are rejected with a message naming the parameter, the expected
       JSON Schema type, and the actual Python type name.

    Returns:
        ``None`` when arguments are valid, or an error-message string when
        one or more checks fail (the first failure encountered).
    """
    schema: dict = tool.parameters
    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] | None = schema.get("required")

    # 1. Unknown keys
    allowed_params = list(properties.keys())
    for key in arguments:
        if key not in properties:
            return (
                f"unknown parameter '{key}'; "
                f"allowed parameters are {', '.join(allowed_params)}"
            )

    # 2. Missing required keys
    if required:
        missing = [name for name in required if name not in arguments]
        if missing:
            parts = [f"{name!r}" for name in missing]
            type_parts = []
            for name in missing:
                param_schema = properties.get(name, {})
                ptype = param_schema.get("type")
                if ptype:
                    type_parts.append(f"({ptype})")
                else:
                    type_parts.append("(unknown)")
            return (
                f"missing required parameter(s) "
                f"{', '.join(parts)} {''.join(type_parts)}"
            )

    # 3. Type validation
    for key, value in arguments.items():
        prop_schema = properties.get(key, {})
        expected_type_str: str | None = prop_schema.get("type")
        if expected_type_str is None:
            continue

        python_types = _PY_TYPE_MAP.get(expected_type_str)
        if python_types is None:
            continue

        # reject bool for integer/number (bool is subclass of int in Python)
        if expected_type_str == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return (
                    f"parameter '{key}' must be {expected_type_str}, "
                    f"got {type(value).__name__}"
                )

        elif expected_type_str == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return (
                    f"parameter '{key}' must be {expected_type_str}, "
                    f"got {type(value).__name__}"
                )

        else:
            expected_python = python_types
            if not isinstance(value, expected_python):
                singular_type = str(expected_python).split(".")[-1].strip("()'")
                # Handle tuple display like "(int, float)"
                if "," in singular_type:
                    type_display = f"{{{expected_type_str} types}}"
                else:
                    type_display = f"{expected_type_str}"
                return (
                    f"parameter '{key}' must be {type_display}, "
                    f"got {type(value).__name__}"
                )

    return None

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
    """Look up tool *name*, validate and call ``run(**arguments)``, and return the result.

    Validation proceeds in three stages before execution: (1) unknown parameter keys
    are rejected, (2) missing required keys are rejected, and (3) each value is checked
    against the declared JSON Schema type.  Any validation failure returns a
    ``bad-arguments`` error immediately without calling ``run``.

    When arguments pass validation the tool's ``run()`` is invoked; all exceptions from
    an unknown or a crashed tool are captured as errors — nothing escapes this function.

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

    # Validate arguments against the tool's JSON Schema before execution
    validation_error = validate_arguments(tool, arguments)
    if validation_error is not None:
        return ToolResult.err(
            validation_error,
            code="bad-arguments",
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
