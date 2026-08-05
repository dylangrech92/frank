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

_PY_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_arguments(tool: Tool, arguments: dict[str, Any]) -> str | None:
    """Validate *arguments* against the tool's JSON Schema ``parameters``.

    Checks three things, and reports **every** violation it finds rather than
    stopping at the first:

    1. **Unknown keys rejected** — every key in *arguments* must appear in the
       schema's ``properties`` mapping.  Unknown keys produce a message naming
       them, followed by the allowed parameter names.

    2. **Required keys checked** — every name in the schema's ``required`` list
       must be present in *arguments*.  Each missing name is listed with its
       own declared type beside it.

    3. **Type checking** — each provided value is checked against the declared
       JSON Schema ``type`` for that property, using the mapping below::

          string       → str
          integer      → int (rejects bool)
          number       → int or float (rejects bool)
          boolean      → bool
          array        → list
          object       → dict

       Mismatches name the parameter, the expected JSON Schema type, and the
       actual Python type name.

    Reporting all three classes at once is deliberate.  Returning after the
    first class made a caller fix one violation per round-trip, so a tool with
    four required parameters could take four rejected calls to get right — long
    enough for the repeat-call guard to step in and block the correction that
    was on its way.  One message means one corrected retry.

    Returns:
        ``None`` when arguments are valid, or a message describing every
        violation found, one per line.
    """
    schema: dict = tool.parameters
    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] | None = schema.get("required")

    problems: list[str] = []

    # 1. Unknown keys
    unknown = [key for key in arguments if key not in properties]
    if unknown:
        allowed = ", ".join(properties) or "(none)"
        problems.append(
            f"unknown parameter(s) {', '.join(repr(key) for key in unknown)}; "
            f"allowed parameters are {allowed}"
        )

    # 2. Missing required keys.  Each name carries its own type, so a caller
    #    reading the message never has to align two parallel lists.
    if required:
        missing = [
            f"{name!r} ({properties.get(name, {}).get('type') or 'unknown'})"
            for name in required
            if name not in arguments
        ]
        if missing:
            problems.append(
                f"missing required parameter(s) {', '.join(missing)}"
            )

    # 3. Type validation.  A key with no declared type is skipped, which covers
    #    both an unknown key (check 1 already named it, and it has no schema
    #    entry to check against) and a property whose schema declares no type.
    #    So is a type string with no mapping: an unrecognised schema is not the
    #    caller's fault, so it is not the caller's error.
    for key, value in arguments.items():
        expected: str | None = properties.get(key, {}).get("type")
        python_types = _PY_TYPE_MAP.get(expected) if expected is not None else None
        if python_types is None:
            continue

        # bool is a subclass of int in Python, so the numeric types have to
        # exclude it explicitly or ``True`` passes as an integer.
        if expected in ("integer", "number") and isinstance(value, bool):
            matches = False
        else:
            matches = isinstance(value, python_types)

        if not matches:
            problems.append(
                f"parameter '{key}' must be {expected}, got {type(value).__name__}"
            )

    return "\n".join(problems) if problems else None

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


def activate_mode(name: str, extra_tools: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Activate mode *name*, replacing the active tool set wholesale.

    Auto-discovers if the registry has not been populated yet, then verifies
    every tool the mode names — plus every name in *extra_tools* — actually
    resolves in ``_registry``; an unknown name raises ``ValueError`` naming
    both the mode and the bad name, since a mode referencing a tool that does
    not exist is a harness bug, not a runtime condition to degrade gracefully
    from. Idempotent: reactivating the same (or another) mode simply
    recomputes the active set.

    Args:
        name: A key of ``modes.MODES``.
        extra_tools: Names to add to *name*'s static tool set for this
            activation only — how a tool whose availability depends on
            something outside modes.py (see ``modes.CONDITIONAL_TOOLS``, e.g.
            ``vision`` gated on config.json's optional ``vision`` block) gets
            into the active set without being baked into every mode's fixed
            tuple. Empty (the default) reproduces the mode's tools exactly as
            declared — the byte-identical-when-unconfigured guarantee lives in
            the caller passing nothing here, not in this function guessing.

    Returns:
        The active tool names: *name*'s declared tools followed by any of
        *extra_tools* not already among them, in that order.

    Raises:
        ValueError: If *name* is not a known mode, or if the mode or
            *extra_tools* names a tool that is not registered.
    """
    if not _registry:
        discover()

    mode = get_mode(name)
    tool_names = mode.tools + tuple(t for t in extra_tools if t not in mode.tools)
    unknown = [t for t in tool_names if t not in _registry]
    if unknown:
        raise ValueError(
            f"mode {name!r} names unknown tool(s): {', '.join(unknown)}"
        )

    global _active_mode, _active_tools
    _active_mode = mode.name
    _active_tools = frozenset(tool_names)
    return tool_names


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
    against the declared JSON Schema type.  All three are reported together, so one
    corrected retry can clear them; any violation returns an error without calling
    ``run``.

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
