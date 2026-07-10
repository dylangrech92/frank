"""record tool: write one typed node (rule/decision/pivot/spec) into the data graph.

Collapses the former record_rule / record_decision / record_pivot / record_spec /
link_nodes tools into a single ``record`` call. ``supersedes`` and ``implements``
accept node TITLES as well as raw ids, resolved against the graph before anything
is written: an ambiguous or unknown reference refuses the whole write (nothing is
recorded) and lists the closest candidates so the model can retry disambiguated.
On success the result echoes the new node's id (so it becomes knowable) and the
resolved target of every edge it created.
"""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult

_KINDS = ("rule", "decision", "pivot", "spec")


def _as_ref_list(raw: Any) -> list[Any]:
    """Normalise a supersedes/implements argument to a list of refs (ids or titles)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [r for r in raw if not isinstance(r, bool) and (isinstance(r, (int, str)))]
    if isinstance(raw, bool):
        return []
    if isinstance(raw, (int, str)):
        return [raw]
    return []


def _describe(row: dict) -> str:
    """Render a resolved node row as ``#id 'title' (type)`` for the echo/candidate lists."""
    return f"#{row['id']} '{row['title']}' ({row['type']})"


class Record(Tool):
    """Record one typed node (rule/decision/pivot/spec) into the project data graph."""

    name = "record"
    summary = "Record a rule/decision/pivot/spec into the project graph memory."
    description = (
        "Record durable project knowledge as one typed node. kind is one of:\n"
        "- rule: an always-enforced constraint (title = name, body = the constraint); "
        "active rules are injected into every context.\n"
        "- decision: a design choice (title = name, body = the rationale); optional "
        "alternatives (other options considered) and implements (nodes it realises).\n"
        "- spec: a specification (title = name, body = the spec); optional acceptance "
        "and status (e.g. draft/accepted/done).\n"
        "- pivot: a reversal (title = name, body = why it changed); supersedes is "
        "REQUIRED and lists the node(s) it replaces.\n"
        "supersedes and implements each take a list of node TITLES or ids -- titles "
        "are matched against existing nodes. If a title is ambiguous or unknown the "
        "whole write is refused and the candidates are listed; retry with an exact "
        "title or an id. The result echoes the new node's id so you can reference it "
        "later."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(_KINDS),
                "description": "rule | decision | pivot | spec.",
            },
            "title": {"type": "string", "description": "A short name for the node."},
            "body": {
                "type": "string",
                "description": "The content: rule=constraint, decision=rationale, pivot=why, spec=body.",
            },
            "alternatives": {
                "type": "array",
                "items": {"type": "string"},
                "description": "decision: other options that were considered.",
            },
            "acceptance": {"type": "string", "description": "spec: acceptance criteria."},
            "status": {"type": "string", "description": "spec: status, e.g. draft/accepted/done."},
            "supersedes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Titles or ids of node(s) this replaces (REQUIRED for pivot).",
            },
            "implements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "decision: titles or ids of node(s) this decision implements.",
            },
        },
        "required": ["kind", "title", "body"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        kind_raw = kwargs.get("kind")
        kind = kind_raw.strip().lower() if isinstance(kind_raw, str) else ""
        if kind not in _KINDS:
            return ToolResult.err(
                f'The "kind" argument must be one of {_KINDS}.', code="bad-arguments"
            )
        title_raw = kwargs.get("title")
        title = title_raw if isinstance(title_raw, str) else ""
        body_raw = kwargs.get("body")
        body = body_raw if isinstance(body_raw, str) else ""
        if not title.strip():
            return ToolResult.err('The "title" argument is required.', code="missing-argument")
        if not body.strip():
            return ToolResult.err('The "body" argument is required.', code="missing-argument")

        supersedes_refs = _as_ref_list(kwargs.get("supersedes"))
        implements_refs = _as_ref_list(kwargs.get("implements"))
        if kind == "pivot" and not supersedes_refs:
            return ToolResult.err(
                'A pivot must "supersede" at least one existing node (by title or id).',
                code="missing-argument",
            )

        from memory.recall import get_memory
        from memory import graph

        try:
            ctx = get_memory()
        except Exception as exc:
            return ToolResult.err(f"Memory is unavailable: {exc}", code="graph-error")

        # --- Resolve every edge target BEFORE writing anything (transactional). ---
        supersede_targets, err = self._resolve_all(ctx, graph, supersedes_refs, "supersedes")
        if err is not None:
            return err
        implement_targets, err = self._resolve_all(ctx, graph, implements_refs, "implements")
        if err is not None:
            return err

        # --- Write the node (+ edges), reusing the graph's own semantics. ---
        try:
            if kind == "pivot":
                node_id = graph.record_pivot(
                    ctx, title, body, [t["id"] for t in supersede_targets]
                )
            else:
                extra = self._build_extra(kind, kwargs)
                node_id = graph.create_node(ctx, kind, title, body, extra=extra or None)
                if supersede_targets:
                    graph.supersede(ctx, node_id, [t["id"] for t in supersede_targets])
                for tgt in implement_targets:
                    graph.link_nodes(ctx, node_id, tgt["id"], "implements")
        except Exception as exc:
            if kind == "pivot":
                # record_pivot is transactional: node + edges commit or roll back together.
                return ToolResult.err(
                    f"Failed to record pivot (no changes were written): {exc}",
                    code="graph-error",
                )
            # Other kinds commit the node before its edges, so a late edge failure
            # can leave the node written without them.
            return ToolResult.err(
                f"Failed while recording {kind}: {exc}. The node may have been "
                "written without its edges; check before retrying.",
                code="graph-error",
            )

        return ToolResult.ok(
            self._echo(kind, node_id, title, supersede_targets, implement_targets),
            node_id=node_id,
            node_type=kind,
        )

    @staticmethod
    def _build_extra(kind: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Collect the kind-specific extra payload (alternatives / acceptance / status)."""
        extra: dict[str, Any] = {}
        if kind == "decision":
            alternatives = kwargs.get("alternatives")
            if isinstance(alternatives, str):
                alternatives = [a.strip() for a in alternatives.split(",") if a.strip()]
            elif isinstance(alternatives, list):
                alternatives = [str(a) for a in alternatives if str(a).strip()]
            else:
                alternatives = []
            if alternatives:
                extra["alternatives"] = alternatives
        if kind == "spec":
            acceptance = kwargs.get("acceptance")
            status = kwargs.get("status")
            if isinstance(acceptance, str) and acceptance.strip():
                extra["acceptance"] = acceptance.strip()
            if isinstance(status, str) and status.strip():
                extra["status"] = status.strip()
        return extra

    @staticmethod
    def _resolve_all(
        ctx, graph, refs: list[Any], label: str
    ) -> tuple[list[dict], ToolResult | None]:
        """Resolve every ref to a unique active node; return an error on the first failure.

        De-duplicates resolved targets (a title and its id resolving to the same node
        are the same target) so a double reference cannot trip the unique-edge guard.
        """
        resolved: list[dict] = []
        seen: set[int] = set()
        for ref in refs:
            row, candidates = graph.resolve_node_ref(ctx, ref)
            if row is None:
                if candidates:
                    listing = "; ".join(_describe(c) for c in candidates)
                    return [], ToolResult.err(
                        f'The {label} reference "{ref}" is ambiguous -- it matches '
                        f"several active nodes: {listing}. Retry with an exact title or an id.",
                        code="ambiguous-reference",
                    )
                return [], ToolResult.err(
                    f'The {label} reference "{ref}" matched no active node. Nothing was '
                    "written. Use an exact node title or a node id.",
                    code="unknown-reference",
                )
            if row["id"] not in seen:
                seen.add(row["id"])
                resolved.append(row)
        return resolved, None

    @staticmethod
    def _echo(
        kind: str,
        node_id: int,
        title: str,
        supersede_targets: list[dict],
        implement_targets: list[dict],
    ) -> str:
        """Build the success body, echoing the new id and every resolved edge target."""
        parts = [f"Recorded {kind} #{node_id} '{title}'."]
        if supersede_targets:
            parts.append("Supersedes " + ", ".join(_describe(t) for t in supersede_targets) + ".")
        if implement_targets:
            parts.append("Implements " + ", ".join(_describe(t) for t in implement_targets) + ".")
        return " ".join(parts)
