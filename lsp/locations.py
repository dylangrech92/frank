"""Shared helpers for LSP code-navigation — locations, symbols, and rendering utilities."""

from __future__ import annotations

import os

from lsp.manager import uri_to_path


# Full LSP SymbolKind mapping (kind integer 1-26) to its lowercase name.
SYMBOL_KIND_NAMES: dict[int, str] = {
    1: "file",
    2: "module",
    3: "namespace",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    15: "string",
    16: "number",
    17: "boolean",
    18: "array",
    19: "object",
    20: "key",
    21: "null",
    22: "enum-member",
    23: "struct",
    24: "event",
    25: "operator",
    26: "type-parameter",
}


def symbol_kind_name(kind: int) -> str:
    """Return the human-readable lowercase name for an LSP SymbolKind integer.

    Args:
        kind: An LSP SymbolKind constant (1-26).

    Returns:
        The mapped lowercase name string, or ``"symbol"`` for unknown values.
    """
    return SYMBOL_KIND_NAMES.get(kind, "symbol")


def normalize_locations(result) -> list[dict]:
    """Normalise the raw result of an LSP code-navigation request into a uniform list.

    Handles all of these shapes: ``None``, a single Location dict (has ``"uri"`` and
    ``"range"``), a list of Location dicts, and a list of LocationLink dicts (have
    ``"targetUri"``; position comes from ``"targetSelectionRange"`` when present,
    else ``"targetRange"``).

    Args:
        result: The raw response value from a definition/implementation/typeDefinition/references request.

    Returns:
        A list of dicts with keys ``"uri"``, ``"line"``, and ``"character"`` (all 0-based positions).
    """
    if result is None:
        return []

    # Single Location dict (has "uri" at top level)
    if isinstance(result, dict) and "uri" in result:
        try:
            range_data = result["range"]
            start = range_data["start"]
            return [{
                "uri": result["uri"],
                "line": int(start["line"]),
                "character": int(start["character"]),
            }]
        except (KeyError, TypeError, ValueError):
            return []

    # Could be a list of Locations or LocationLinks
    if not isinstance(result, list):
        return []

    out: list[dict] = []
    for item in result:
        try:
            if not isinstance(item, dict):
                continue

            # LocationLink: has "targetUri" at top level (in addition to possibly "uri")
            if "targetUri" in item:
                target_uri = item["targetUri"]
                tsr = item.get("targetSelectionRange")
                tr = item.get("targetRange")
                range_data = tsr if tsr is not None else tr
                if range_data is None:
                    continue
                start = range_data["start"]
                out.append({
                    "uri": target_uri,
                    "line": int(start["line"]),
                    "character": int(start["character"]),
                })
            # Regular Location: has "uri" and "range" at top level
            elif "uri" in item and "range" in item:
                range_data = item["range"]
                start = range_data["start"]
                out.append({
                    "uri": item["uri"],
                    "line": int(start["line"]),
                    "character": int(start["character"]),
                })
        except (KeyError, TypeError, ValueError):
            continue

    return out


def format_location(uri: str, line: int, character: int, root_path: str) -> str:
    """Format a single location as ``"rel:line1:col1"``.

    The relative path is derived from *uri* via :func:`uri_to_path` and
    :func:`os.path.relpath` against *root_path*.  Positions are rendered as 1-based.

    Args:
        uri: A ``file://`` URI string (or any string that falls back safely).
        line: 0-based line position.
        character: 0-based character (column) position.
        root_path: Absolute path used to compute the relative distance.

    Returns:
        A string of the form ``"rel:line1:col1"`` with 1-based positions.
    """
    try:
        abs_path = uri_to_path(uri)
        rel = os.path.relpath(abs_path, root_path)
    except Exception:  # noqa: E722
        rel = uri

    return f"{rel}:{line + 1}:{character + 1}"


def render_locations(result, root_path: str) -> list[str]:
    """Normalize an LSP navigation result and render each location as a ``rel:line:col`` string.

    Preserves order and removes exact duplicate strings while keeping the first occurrence.

    Args:
        result: The raw response value from a definition/implementation/typeDefinition/references request.
        root_path: Absolute path used to compute relative paths.

    Returns:
        A de-duplicated list of ``"rel:line1:col1"`` strings in original order.
    """
    locations = normalize_locations(result)
    seen: set[str] = set()
    result_list: list[str] = []
    for loc in locations:
        rendered = format_location(loc["uri"], loc["line"], loc["character"], root_path)
        if rendered not in seen:
            seen.add(rendered)
            result_list.append(rendered)
    return result_list


def flatten_symbols(result) -> list[dict]:
    """Flatten the raw result of textDocument/documentSymbol or workspace/symbol into a uniform list.

    Handles two shapes: hierarchical :class:`DocumentSymbol` entries (have ``"selectionRange"``;
    recurse into ``"children"``, child depth = parent depth + 1, child container = parent name,
    uri stays ``None``) and flat SymbolInformation/WorkspaceSymbol entries (have ``"location"``
    with ``"uri"`` and ``"range"``: depth 0; container from ``"containerName"`` when present).

    Args:
        result: The raw response value from a documentSymbol or workspace/symbol request.

    Returns:
        A flat list of dicts, each with keys ``name``, ``kind``, ``line``, ``character``,
        ``uri``, ``container``, and ``depth``.
    """
    if result is None:
        return []

    out: list[dict] = []

    def _recurse(symbols: list, parent_depth: int, parent_container: str | None) -> None:
        for sym in symbols:
            try:
                if not isinstance(sym, dict):
                    continue

                name = sym.get("name", "")
                kind_int = sym.get("kind", 0)
                kind_str = symbol_kind_name(kind_int)

                # Hierarchical DocumentSymbol: has "selectionRange" and children
                if "selectionRange" in sym:
                    range_data = sym.get("selectionRange", {})
                    start = range_data.get("start")
                    line = int(start["line"]) if start else 0
                    character = int(start["character"]) if start else 0

                    out.append({
                        "name": name,
                        "kind": kind_str,
                        "line": line,
                        "character": character,
                        "uri": None,
                        "container": parent_container,
                        "depth": parent_depth,
                    })

                    children = sym.get("children")
                    if isinstance(children, list):
                        _recurse(
                            children,
                            parent_depth=parent_depth + 1,
                            parent_container=name,
                        )

                # Flat SymbolInformation / WorkspaceSymbol: has "location" with uri and range
                elif "location" in sym and isinstance(sym["location"], dict):
                    location = sym["location"]
                    loc_range = location.get("range")
                    loc_start = loc_range.get("start") if loc_range else None

                    uri_val: str | None = None
                    try:
                        uri_val = location["uri"]
                    except (KeyError, TypeError):
                        pass

                    line = int(loc_start["line"]) if loc_start else 0
                    character = int(loc_start["character"]) if loc_start else 0

                    out.append({
                        "name": name,
                        "kind": kind_str,
                        "line": line,
                        "character": character,
                        "uri": uri_val,
                        "container": sym.get("containerName") if isinstance(sym.get("containerName"), str) else None,
                        "depth": parent_depth,
                    })
            except (KeyError, TypeError, ValueError):
                continue

    _recurse(result, parent_depth=0, parent_container=None)
    return out


def render_symbol_line(sym: dict, root_path: str) -> str:
    """Render a single flattened symbol as an indented display string with a location suffix.

    For each level of *depth*, prepends two spaces of indentation. Then prints
    ``"{name} [{kind}]"``. The location suffix follows:
    - When ``"uri"`` is set, appends separator and
      :func:`format_location` result.
    - Otherwise, appends ``" — line {line+1}"``.
    - When ``"container"`` is set and depth == 0, appends the container name in parentheses
      before the location suffix.

    Args:
        sym: A dict from :func:`flatten_symbols` containing at least ``name``, ``kind``,
            ``line``, ``character``, ``uri``, ``container``, and ``depth`` keys.
        root_path: Absolute path used to compute relative paths for the location suffix.

    Returns:
        A formatted display string ready for console output or log rendering.
    """
    prefix = "  " * sym.get("depth", 0)
    name = sym["name"]
    kind = sym["kind"]
    line = sym["line"]
    uri = sym.get("uri")
    container = sym.get("container")
    character = sym.get("character", 0)

    base = f"{prefix}{name} [{kind}]"

    suffix_parts: list[str] = []

    if container is not None and sym.get("depth", 0) == 0:
        suffix_parts.append(f" (in {container})")

    if uri is not None and isinstance(uri, str):
        formatted_loc = format_location(uri, line, character, root_path)
        suffix_parts.append(f" — {formatted_loc}")
    else:
        suffix_parts.append(f" — line {line + 1}")

    return f"{base}{''.join(suffix_parts)}"
