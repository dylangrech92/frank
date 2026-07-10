"""Dispatch-level check for the consolidation op-apply path (no LLM).

Drives ``memory.consolidation.apply_ops`` -- the real seam ``consolidate`` uses
after parsing the model's JSON -- against a throwaway memory db in a temp project
dir, so the DECISION/PIVOT graph writes are exercised deterministically without a
live model. Asserts:

a. a DECISION op writes a decision node (title + body land in graph_nodes);
b. a PIVOT op whose ``supersedes`` is the EXACT TITLE of a seeded decision creates
   the ``supersedes`` edge and stamps the target ``superseded_at`` + ``active=0``;
c. a PIVOT with an unknown title writes nothing, raises nothing, and is counted as
   a noop (the skip is observable in the returned stats);
d. a PIVOT whose ``supersedes`` is a raw int id still supersedes that node;
e. an ADD atom op still writes a fact row (the refactor didn't break the atom path).

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it) and restores the
cwd it changed.
"""

from __future__ import annotations

import os
import sys
import tempfile


class _Fail(Exception):
    """Raised to abort the eval with a specific assertion message."""


def _node_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM graph_nodes").fetchone()["c"]


def _run(ctx, conn, project_root: str) -> None:
    from memory.consolidation import apply_ops

    # ---- a. DECISION op writes a decision node ----
    decision_title = "Use event bus for module comms"
    apply_ops(
        ctx,
        [
            {
                "op": "DECISION",
                "title": decision_title,
                "body": "decouples producers from consumers; direct calls rejected as too coupled",
            }
        ],
        project_root=project_root,
    )
    drow = conn.execute(
        "SELECT id, type, body, active FROM graph_nodes WHERE title=?", (decision_title,)
    ).fetchone()
    if drow is None:
        raise _Fail("DECISION op wrote no graph node")
    if drow["type"] != "decision":
        raise _Fail(f"DECISION op wrote a {drow['type']!r} node, expected 'decision'")
    if not (drow["body"] or "").strip():
        raise _Fail("DECISION op wrote an empty body")
    if drow["active"] != 1:
        raise _Fail("newly written DECISION node is not active")
    decision_id = drow["id"]

    # ---- b. PIVOT supersedes a decision BY EXACT TITLE ----
    pivot_title = "Replace event bus with direct calls"
    stats = apply_ops(
        ctx,
        [
            {
                "op": "PIVOT",
                "title": pivot_title,
                "body": "bus indirection outweighed its decoupling benefit for this small tree",
                "supersedes": [decision_title],
            }
        ],
        project_root=project_root,
    )
    if stats["pivots"] != 1:
        raise _Fail(f"PIVOT-by-title not counted (stats={stats})")
    prow = conn.execute(
        "SELECT id FROM graph_nodes WHERE title=? AND type='pivot'", (pivot_title,)
    ).fetchone()
    if prow is None:
        raise _Fail("PIVOT-by-title wrote no pivot node")
    pivot_id = prow["id"]
    edge = conn.execute(
        "SELECT 1 FROM graph_edges WHERE from_id=? AND to_id=? AND edge_type='supersedes'",
        (pivot_id, decision_id),
    ).fetchone()
    if edge is None:
        raise _Fail("PIVOT-by-title created no supersedes edge to the resolved decision")
    trow = conn.execute(
        "SELECT superseded_at, active FROM graph_nodes WHERE id=?", (decision_id,)
    ).fetchone()
    if trow["superseded_at"] is None or trow["active"] != 0:
        raise _Fail("PIVOT-by-title did not stamp the target superseded_at/active")

    # ---- c. PIVOT with an unknown title: nothing written, no exception, noop ----
    count_before = _node_count(conn)
    stats = apply_ops(
        ctx,
        [
            {
                "op": "PIVOT",
                "title": "Rework nothing at all",
                "body": "there is no such prior decision",
                "supersedes": ["this decision title does not exist anywhere"],
            }
        ],
        project_root=project_root,
    )
    if _node_count(conn) != count_before:
        raise _Fail("unknown-title PIVOT wrote a node (should be a noop)")
    if stats["pivots"] != 0:
        raise _Fail(f"unknown-title PIVOT was counted as a pivot (stats={stats})")
    if stats["noop"] != 1:
        raise _Fail(f"unknown-title PIVOT was not counted as a noop (stats={stats})")

    # ---- d. PIVOT with a raw int id still works ----
    id_target_title = "Poll the work queue every minute"
    apply_ops(
        ctx,
        [
            {
                "op": "DECISION",
                "title": id_target_title,
                "body": "a simple cron; event-driven wakeups rejected as premature",
            }
        ],
        project_root=project_root,
    )
    id_target = conn.execute(
        "SELECT id FROM graph_nodes WHERE title=? AND active=1", (id_target_title,)
    ).fetchone()
    if id_target is None:
        raise _Fail("could not seed the raw-id pivot target decision")
    id_target_id = id_target["id"]
    id_pivot_title = "Switch to event-driven queue wakeups"
    stats = apply_ops(
        ctx,
        [
            {
                "op": "PIVOT",
                "title": id_pivot_title,
                "body": "polling wasted cycles under low load",
                "supersedes": [id_target_id],
            }
        ],
        project_root=project_root,
    )
    if stats["pivots"] != 1:
        raise _Fail(f"raw-id PIVOT not counted (stats={stats})")
    id_pivot = conn.execute(
        "SELECT id FROM graph_nodes WHERE title=? AND type='pivot'", (id_pivot_title,)
    ).fetchone()
    if id_pivot is None:
        raise _Fail("raw-id PIVOT wrote no pivot node")
    edge = conn.execute(
        "SELECT 1 FROM graph_edges WHERE from_id=? AND to_id=? AND edge_type='supersedes'",
        (id_pivot["id"], id_target_id),
    ).fetchone()
    if edge is None:
        raise _Fail("raw-id PIVOT created no supersedes edge")
    trow = conn.execute(
        "SELECT superseded_at, active FROM graph_nodes WHERE id=?", (id_target_id,)
    ).fetchone()
    if trow["superseded_at"] is None or trow["active"] != 0:
        raise _Fail("raw-id PIVOT did not stamp the target superseded_at/active")

    # ---- e. an ADD atom op still writes a fact row ----
    stats = apply_ops(
        ctx,
        [
            {
                "op": "ADD",
                "kind": "project",
                "key": "widget-registry-count",
                "value": "This project registers 12 widgets via widgets/_registry.py::load",
                "confidence": 0.9,
                "anchor_path": None,
            }
        ],
        project_root=project_root,
    )
    if stats["added"] != 1:
        raise _Fail(f"ADD atom op not counted as added (stats={stats})")
    frow = conn.execute(
        "SELECT value FROM facts WHERE key=? AND valid_to IS NULL AND active=1 AND deleted_at IS NULL",
        ("widget-registry-count",),
    ).fetchone()
    if frow is None:
        raise _Fail("ADD atom op wrote no live fact row")
    if "12 widgets" not in frow["value"]:
        raise _Fail(f"ADD atom op stored the wrong value: {frow['value']!r}")


def main() -> int:
    from memory.recall import get_memory

    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="consolidation_ops_")
    try:
        os.chdir(tmp)
        ctx = get_memory(tmp)  # builds the throwaway store under tmp/.coding_agent
        conn = getattr(ctx.store, "conn")
        _run(ctx, conn, tmp)
    except _Fail as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"FAIL: unexpected error: {exc}", file=sys.stderr)
        return 1
    finally:
        os.chdir(prev_cwd)

    print("PASS: consolidation apply_ops writes decisions, resolves pivots by title/id, noops unknown refs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
