"""Xdebug/Valgrind cachegrind text-format parser.

Parses the subset produced by Xdebug 3 ``profile_hotspots`` and Valgrind's
cachegrind output.  Detects gzip compression automatically by magic bytes
and supports both forms transparently.
"""

from __future__ import annotations

import gzip
from typing import IO


def parse_cachegrind(path: str) -> dict:
    """Parse an Xdebug 3 cachegrind profile file.

    The file may be gzip-compressed (detected by the ``\\x1f\\x8b`` magic
    bytes); otherwise it is opened as plain text with ``errors="replace"``.

    Supports the grammar subset used by ``profile_hotspots``: an
    ``events:`` header, ``fl=(id)`` / ``fn=(id)`` declarations with the
    compressed-name ``(id)`` alias mechanism, cost lines after ``fn=``
    blocks, ``cfn=`` / ``calls=`` edges, and a ``summary:`` line.

    Raises:
        ValueError: If the file has no ``events:`` header.
    """
    try:
        with open(path, "rb") as fh:
            magic = fh.read(2)
    except OSError as exc:
        raise ValueError(f"Failed to read cachegrind file {path}: {exc}") from exc

    is_gzip = magic == b"\x1f\x8b"

    events: list[str] = []
    file_table: dict[int, str] = {}
    fn_table: dict[int, str] = {}
    functions: dict[str, dict] = {}
    summary: dict[str, int] = {}

    current_file: str = ""
    current_fn_name: str | None = None
    current_fn_file: str = ""

    # State that bridges a ``cfl=``/``cfn=``/``calls=`` call-edge sequence to
    # the cost line that follows it.
    cfl_pending: str | None = None
    pending_cfn_name: str | None = None
    pending_cfn_file: str = ""
    awaiting_edge_cost = False

    def _open_input() -> IO[str]:
        if is_gzip:
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return open(path, "r", encoding="utf-8", errors="replace")

    def _ensure_fn(name: str, file_: str) -> dict:
        """Return the per-name aggregate dict, creating it if needed."""
        entry = functions.get(name)
        if entry is None:
            entry = {
                "function": name,
                "file": file_,
                "calls": 0,
                "self": {e: 0 for e in events},
                "inclusive": {e: 0 for e in events},
            }
            functions[name] = entry
        return entry

    def _parse_alias(rest: str) -> tuple[int | None, str | None]:
        """Parse a compressed-name payload: ``(id) value``, bare ``(id)``,
        or a plain literal.

        Returns ``(id, value)``. ``id`` is the compressed-name integer when
        the payload starts with ``(id)``, else ``None``. ``value`` is the
        trailing definition text when present, else ``None`` — a bare
        ``(id)`` with no trailing text is a REFERENCE to a previously
        defined id, never a redefinition to an empty string.
        """
        rest = rest.strip()
        if rest.startswith("(") and ")" in rest:
            idx = rest.index(")")
            try:
                id_ = int(rest[1:idx])
            except ValueError:
                id_ = None
            value = rest[idx + 1 :].strip() or None
            return id_, value
        return None, (rest or None)

    with _open_input() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("events:"):
                events = line[len("events:"):].split()
                continue

            if line.startswith("fl="):
                id_, value = _parse_alias(line[len("fl="):])
                if id_ is not None:
                    if value is not None:
                        file_table[id_] = value
                        current_file = value
                    else:
                        current_file = file_table.get(id_, current_file)
                elif value is not None:
                    current_file = value
                continue

            if line.startswith("fn="):
                id_, value = _parse_alias(line[len("fn="):])
                if id_ is not None:
                    if value is not None:
                        fn_table[id_] = value
                        current_fn_name = value
                    else:
                        current_fn_name = fn_table.get(id_, f"({id_})")
                elif value is not None:
                    current_fn_name = value
                else:
                    current_fn_name = None
                current_fn_file = current_file
                pending_cfn_name = None
                awaiting_edge_cost = False
                continue

            if line.startswith("cfl="):
                id_, value = _parse_alias(line[len("cfl="):])
                if id_ is not None:
                    if value is not None:
                        file_table[id_] = value
                        cfl_pending = value
                    else:
                        cfl_pending = file_table.get(id_, current_file)
                elif value is not None:
                    cfl_pending = value
                continue

            if line.startswith("cob=") or line.startswith("ob="):
                # Object-file annotations — not part of the cost model.
                continue

            if line.startswith("cfn="):
                id_, value = _parse_alias(line[len("cfn="):])
                if id_ is not None:
                    if value is not None:
                        fn_table[id_] = value
                        resolved = value
                    else:
                        resolved = fn_table.get(id_, f"({id_})")
                elif value is not None:
                    resolved = value
                else:
                    resolved = "(unknown)"
                pending_cfn_name = resolved
                pending_cfn_file = cfl_pending if cfl_pending is not None else current_file
                cfl_pending = None
                awaiting_edge_cost = False
                continue

            if line.startswith("calls="):
                parts = line.split()
                try:
                    count = int(parts[0].split("=", 1)[1])
                except (ValueError, IndexError):
                    count = 0
                # The remaining tokens (e.g. the ``20`` in ``calls=4 20``)
                # are target POSITIONS, not costs — deliberately ignored.
                name = pending_cfn_name if pending_cfn_name is not None else "(unknown)"
                fn_data = _ensure_fn(name, pending_cfn_file)
                fn_data["calls"] += count
                pending_cfn_name = None
                awaiting_edge_cost = True
                continue

            if line.startswith("summary:"):
                parts = line[len("summary:"):].split()
                for i, cost_str in enumerate(parts):
                    if i < len(events):
                        try:
                            summary[events[i]] = summary.get(events[i], 0) + int(
                                cost_str
                            )
                        except ValueError:
                            pass
                continue

            # Cost line: ``<line> <cost1> [<cost2>]``.
            parts = line.split()
            if len(parts) < 2:
                continue
            if awaiting_edge_cost:
                # The cost line right after a cfn=/calls= pair is the
                # INCLUSIVE cost of that call edge, attributed to the
                # caller's inclusive total — never to the caller's self
                # cost, and never double-counted into the callee (whose
                # own self/inclusive comes from its own fn= block).
                awaiting_edge_cost = False
                if current_fn_name is not None:
                    fn_data = _ensure_fn(current_fn_name, current_fn_file)
                    for i, cost_str in enumerate(parts[1:]):
                        if i < len(events):
                            try:
                                fn_data["inclusive"][events[i]] += int(cost_str)
                            except ValueError:
                                pass
                continue
            if current_fn_name is not None:
                fn_data = _ensure_fn(current_fn_name, current_fn_file)
                for i, cost_str in enumerate(parts[1:]):
                    if i < len(events):
                        try:
                            cost = int(cost_str)
                        except ValueError:
                            continue
                        fn_data["self"][events[i]] += cost
                        fn_data["inclusive"][events[i]] += cost

    if not events:
        raise ValueError("cachegrind file has no 'events:' header")

    func_list = list(functions.values())
    first_event = events[0]
    func_list.sort(key=lambda f: f["self"].get(first_event, 0), reverse=True)

    return {
        "events": events,
        "functions": func_list,
        "summary": summary,
    }

