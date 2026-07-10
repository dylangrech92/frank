"""Dispatch-level check for the unified ``record`` graph-memory tool (no LLM).

Drives the real ``tools.registry.dispatch`` path against a throwaway memory db in
a temp project dir and asserts the guarantees that make the collapsed tool usable:

1. one node of each kind (rule / decision / spec / pivot) is written through the
   real tool, and every success body echoes the new node's id;
2. a pivot that supersedes a decision BY TITLE creates the ``supersedes`` edge and
   stamps the target ``superseded_at`` + ``active=0``;
3. an ambiguous title (two active near-identical nodes) refuses the WHOLE write —
   the error lists the candidates and nothing is recorded;
4. an unknown title refuses the write — nothing is recorded (transactionality);
5. ``supersedes`` by raw id still works;
6. a multi-token title that merely shares one token with a single active node
   refuses instead of fuzzy-resolving to it (title matching requires every token).

Exits 0 on success, 1 on any assertion failure. Runs with the repo root on
``sys.path`` (evals/run.py inserts it) and restores the cwd it changed.
"""

from __future__ import annotations

import os
import sys
import tempfile


class _Fail(Exception):
    """Raised to abort the eval with a specific assertion message."""


def _record(dispatch, **args):
    """Dispatch a ``record`` call and return the ToolResult."""
    return dispatch("record", args)


def _node_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM graph_nodes").fetchone()["c"]


def _run(conn, dispatch) -> None:
    # ---- 1. one node of each simple kind; body echoes the new id ----
    for kind, title, body in (
        ("rule", "camelCase identifiers", "all identifiers use camelCase"),
        ("decision", "use event bus", "decouples producers from consumers"),
        ("spec", "auth module", "email + password login with lockout"),
    ):
        res = _record(dispatch, kind=kind, title=title, body=body)
        if res.status != "success":
            raise _Fail(f"{kind} write failed: {res.body}")
        nid = res.meta.get("node_id")
        if not isinstance(nid, int):
            raise _Fail(f"{kind} result carried no integer node_id meta: {res.meta}")
        if f"#{nid}" not in res.body:
            raise _Fail(f"{kind} success body did not echo the new id #{nid}: {res.body!r}")

    # ---- 2. pivot supersedes a decision BY TITLE ----
    dres = _record(
        dispatch, kind="decision", title="store blobs on disk", body="filesystem is simplest"
    )
    if dres.status != "success":
        raise _Fail(f"target decision write failed: {dres.body}")
    target_id = dres.meta["node_id"]

    pres = _record(
        dispatch,
        kind="pivot",
        title="move blobs to object store",
        body="disk does not scale across nodes",
        supersedes=["store blobs on disk"],
    )
    if pres.status != "success":
        raise _Fail(f"pivot-by-title write failed: {pres.body}")
    pivot_id = pres.meta["node_id"]
    if f"#{target_id}" not in pres.body:
        raise _Fail(f"pivot body did not echo the resolved target #{target_id}: {pres.body!r}")

    edge = conn.execute(
        "SELECT 1 FROM graph_edges WHERE from_id=? AND to_id=? AND edge_type='supersedes'",
        (pivot_id, target_id),
    ).fetchone()
    if edge is None:
        raise _Fail("pivot-by-title created no supersedes edge to the resolved target")
    trow = conn.execute(
        "SELECT superseded_at, active FROM graph_nodes WHERE id=?", (target_id,)
    ).fetchone()
    if trow["superseded_at"] is None or trow["active"] != 0:
        raise _Fail("superseded target was not stamped (superseded_at/active)")

    # ---- 3. ambiguous title refuses the whole write ----
    a1 = _record(dispatch, kind="decision", title="sqlite storage engine", body="engine choice")
    a2 = _record(dispatch, kind="decision", title="sqlite storage cache", body="cache choice")
    if a1.status != "success" or a2.status != "success":
        raise _Fail("could not seed the two ambiguous decisions")

    count_before = _node_count(conn)
    amb = _record(
        dispatch,
        kind="pivot",
        title="rework storage",
        body="single backend",
        supersedes=["sqlite storage"],
    )
    if amb.status != "error" or amb.code != "ambiguous-reference":
        raise _Fail(f"ambiguous title was NOT refused with ambiguous-reference: {amb.status}/{amb.code}")
    if f"#{a1.meta['node_id']}" not in amb.body or f"#{a2.meta['node_id']}" not in amb.body:
        raise _Fail(f"ambiguous error did not list both candidate ids: {amb.body!r}")
    if _node_count(conn) != count_before:
        raise _Fail("ambiguous write left a partial node behind (not transactional)")

    # ---- 4. unknown title refuses the write ----
    count_before = _node_count(conn)
    unk = _record(
        dispatch,
        kind="pivot",
        title="rework nothing",
        body="no such target",
        supersedes=["this title does not exist anywhere"],
    )
    if unk.status != "error" or unk.code != "unknown-reference":
        raise _Fail(f"unknown title was NOT refused with unknown-reference: {unk.status}/{unk.code}")
    if _node_count(conn) != count_before:
        raise _Fail("unknown-title write left a partial node behind (not transactional)")

    # ---- 5. supersedes by raw id still works ----
    rtar = _record(dispatch, kind="decision", title="cron scheduler", body="poll every minute")
    rid = rtar.meta["node_id"]
    rpiv = _record(
        dispatch,
        kind="pivot",
        title="event-driven scheduler",
        body="polling wastes cycles",
        supersedes=[rid],
    )
    if rpiv.status != "success":
        raise _Fail(f"supersedes-by-raw-id write failed: {rpiv.body}")
    edge = conn.execute(
        "SELECT 1 FROM graph_edges WHERE from_id=? AND to_id=? AND edge_type='supersedes'",
        (rpiv.meta["node_id"], rid),
    ).fetchone()
    if edge is None:
        raise _Fail("supersedes-by-raw-id created no edge")

    # ---- 6. partial-overlap wrong title must refuse, not fuzzy-resolve ----
    # "use event bus" (case 1) is the only active node whose title contains "bus";
    # a multi-token reference sharing just that one token must NOT resolve to it —
    # a wrong supersedes target would stamp the wrong node inactive.
    count_before = _node_count(conn)
    fuzz = _record(
        dispatch,
        kind="pivot",
        title="drop the message queue",
        body="direct calls are enough",
        supersedes=["bus timetable rendering"],
    )
    if fuzz.status != "error":
        raise _Fail(
            f"partial-overlap title fuzzy-resolved instead of refusing: {fuzz.body!r}"
        )
    if _node_count(conn) != count_before:
        raise _Fail("partial-overlap refusal left a partial node behind")


def main() -> int:
    from tools import registry
    from memory.recall import get_memory

    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="record_graph_")
    try:
        os.chdir(tmp)
        registry.discover()
        registry.activate("record")  # tool must be loaded before dispatch will run it

        ctx = get_memory(tmp)  # builds the throwaway store under tmp/.coding_agent
        conn = getattr(ctx.store, "conn")

        _run(conn, registry.dispatch)
    except _Fail as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"FAIL: unexpected error: {exc}", file=sys.stderr)
        return 1
    finally:
        os.chdir(prev_cwd)

    print("PASS: record tool writes each kind, resolves titles, and refuses ambiguous/unknown refs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
