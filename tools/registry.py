"""Tool discovery, schema generation, and dispatch for the tools/ package."""

from __future__ import annotations

import inspect
import pkgutil
import sys
from importlib import import_module
from typing import Any

from modes import get_mode
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

# The currently active mode's name, and its fixed tool set. Single-session
# app, so module-level state is the correct home (mirrors the existing
# module-global pattern used for MANAGER / DEBUG_MANAGER). Set once per
# process by ``activate_mode()``; ``None`` / empty until then.
_active_mode: str | None = None
_active_tools: frozenset[str] = frozenset()


def is_loaded(name: str) -> bool:
    """Return True when *name* is in the active mode's tool set.

    Single source of truth for the mode gate: ``schemas()`` exposes exactly
    these tools and ``dispatch()`` refuses to run anything else.
    """
    return name in _active_tools


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


def activate_mode(name: str) -> tuple[str, ...]:
    """Activate mode *name*, replacing the active tool set wholesale.

    Auto-discovers if the registry has not been populated yet, then verifies
    every tool the mode names actually resolves in ``_registry`` — an unknown
    name raises ``ValueError`` naming both the mode and the bad name, since a
    mode referencing a tool that does not exist is a harness bug, not a
    runtime condition to degrade gracefully from. Idempotent: reactivating the
    same (or another) mode simply recomputes the active set.

    Args:
        name: A key of ``modes.MODES``.

    Returns:
        The mode's tool names, in the order declared in ``modes.py``.

    Raises:
        ValueError: If *name* is not a known mode, or if the mode names a
            tool that is not registered.
    """
    if not _registry:
        discover()

    mode = get_mode(name)
    unknown = [t for t in mode.tools if t not in _registry]
    if unknown:
        raise ValueError(
            f"mode {name!r} names unknown tool(s): {', '.join(unknown)}"
        )

    global _active_mode, _active_tools
    _active_mode = mode.name
    _active_tools = frozenset(mode.tools)
    return mode.tools


def current_mode() -> str | None:
    """Return the active mode's name, or ``None`` when no mode has been activated."""
    return _active_mode


def schemas() -> list[dict[str, Any]]:
    """Return the OpenAI Chat Completions tools array for the active mode.

    One entry per tool in the active mode's tool set, with ``type="function"``
    and a ``function`` key carrying ``name``, ``description``, and
    ``parameters`` — exactly that mode's tools, full schemas, nothing else.

    Raises:
        RuntimeError: If no mode is active. A modeless process is a bug, not
            a degraded mode to serve schemas for.
    """
    if _active_mode is None:
        raise RuntimeError(
            "no mode is active — call registry.activate_mode(name) before schemas()"
        )
    if not _registry:
        discover()

    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for name, t in _registry.items() if name in _active_tools
    ]


def get_tool(name: str) -> Tool | None:
    """Return the registered ``Tool`` instance for *name*, or ``None`` if unknown.

    Auto-discovers if the registry has not been populated yet.  Used by callers
    that need a tool's optional presentation attributes (``action``,
    ``oversize_hint``, ``alternative``) outside of ``dispatch()``.
    """
    if not _registry:
        discover()
    return _registry.get(name)


def dispatch(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Look up tool *name*, validate and call ``run(**arguments)``, and return the result.

    Validation proceeds in stages before execution: (0) the tool must be in the
    active mode's tool set — a tool outside the active mode is rejected with a
    ``not-in-mode`` error rather than executed; (1) unknown parameter keys are
    rejected, (2) missing required keys are rejected, and (3) each value is checked
    against the declared JSON Schema type.  Any validation failure returns an error
    immediately without calling ``run``.

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

    # A tool-call whose arguments string failed to parse (even after repair
    # attempts in llm.py) arrives here flagged rather than silently emptied —
    # report it as malformed JSON, not a misleading missing-parameter error.
    if isinstance(arguments, dict) and "__json_error__" in arguments:
        return ToolResult.err(
            f"{name}: your tool-call arguments were not valid JSON "
            f"({arguments['__json_error__']})",
            code="malformed-json",
            hint=(
                "Re-emit the call with valid JSON. Most common causes: unescaped "
                "quotes or raw newlines inside string values, trailing commas. For "
                "large file content, double-check every quote inside the content "
                "is escaped."
            ),
        )

    # Gate: the tool must be in the active mode's tool set before it can be
    # dispatched — mirrors the schemas() contract so a tool never executes
    # without the model having seen its full definition this run. There is no
    # rescue path: the active mode's tool set is fixed for the life of the
    # process.
    if not is_loaded(name):
        mode_label = _active_mode if _active_mode is not None else "(no mode active)"
        mode_tools = ", ".join(sorted(_active_tools)) if _active_tools else "(none)"
        return ToolResult.err(
            f"tool {name!r} is not in the active mode {mode_label!r}",
            code="not-in-mode",
            hint=f"Mode {mode_label!r} has: {mode_tools}",
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
            hint=(
                "Unexpected internal error, not a usage mistake — do not retry "
                "with the same arguments; try a different approach or report the "
                "problem to the user."
            ),
        )

    return result  # type: ignore[return-value]
