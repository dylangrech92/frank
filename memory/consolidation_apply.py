from __future__ import annotations

from memory import anchor as _anchor
from memory.consolidation_ops import (
    _best_existing_key_match,
    _coerce_kind,
    _log,
    _resolve_anchor_path,
)

CONFIDENCE_FLOOR = 0.45        # ADD/UPDATE below this confidence is dropped (poisoning defence)


def _apply_atom_op(ctx, op, action, *, project_root, existing_atoms, stats) -> None:
    """Apply one ADD/UPDATE atom op via ``memory.atomic.remember`` (code-anchored)."""
    from memory.atomic import remember

    conn = ctx.store.conn
    key = op.get("key")
    key = key.strip() if isinstance(key, str) else ""
    value = op.get("value")
    value = value.strip() if isinstance(value, str) else ""
    if not key or not value:
        stats["noop"] += 1
        return
    try:
        confidence = float(op.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    if confidence < CONFIDENCE_FLOOR:
        stats["skipped_low_confidence"] += 1
        return
    # Resolve the EFFECTIVE key before anything else: two independent weak-LLM
    # key choices for the same underlying fact need not match (e.g. a correction
    # turn inventing a different key than the original wrong atom -- see
    # consolidation module docstring). Try the model's own
    # key first; if no live atom holds it, fall back to the key of the
    # most-similar existing atom shown to the model, so supersession below closes
    # the stale/wrong atom instead of leaving it live alongside a new sibling.
    pre = conn.execute(
        "SELECT kind FROM facts WHERE key=? AND valid_to IS NULL "
        "AND active=1 AND deleted_at IS NULL",
        (key,),
    ).fetchone()
    effective_key = key
    if pre is None:
        reused_key = _best_existing_key_match(key, value, existing_atoms)
        if reused_key is not None:
            effective_key = reused_key
            pre = conn.execute(
                "SELECT kind FROM facts WHERE key=? AND valid_to IS NULL "
                "AND active=1 AND deleted_at IS NULL",
                (effective_key,),
            ).fetchone()
    # Same-key identity wins over the model's kind choice: if a live atom already
    # holds this key (under ANY kind -- it may predate consolidation and carry a
    # legacy/TTL kind), reuse ITS kind so remember()'s (kind, key)-scoped
    # supersession actually closes it instead of leaving it live alongside a
    # same-key sibling under a different kind.
    kind = pre["kind"] if pre is not None else _coerce_kind(op.get("kind"))
    anchor_path = _resolve_anchor_path(project_root, op.get("anchor_path"))
    anchor_hash, learned_commit = _anchor.anchor_for(anchor_path)

    try:
        remember(
            ctx,
            kind,
            effective_key,
            value,
            anchor_path=anchor_path,
            anchor_hash=anchor_hash,
            learned_commit=learned_commit,
            confidence=confidence,
            source="consolidation",
        )
    except Exception as exc:
        _log(f"write-error op={action} key={effective_key!r}: {exc}")
        return
    stats["updated" if pre is not None else "added"] += 1


def _apply_delete_op(ctx, op, stats) -> None:
    """Apply one DELETE atom op: soft-delete every live atom under the given key."""
    from memory.atomic import forget

    key = op.get("key")
    key = key.strip() if isinstance(key, str) else ""
    if not key:
        stats["noop"] += 1
        return
    # Same identity-over-metadata reasoning as ADD/UPDATE: match this key under
    # ANY kind rather than trusting the model's kind guess.
    try:
        n = forget(ctx, key, None)
    except Exception as exc:
        _log(f"delete-error key={key!r}: {exc}")
        return
    stats["deleted"] += n


def _apply_decision_op(ctx, op, stats) -> None:
    """Apply one DECISION op: create a decision node in the graph layer."""
    from memory import graph

    title = op.get("title")
    body = op.get("body")
    title = title.strip() if isinstance(title, str) else ""
    body = body.strip() if isinstance(body, str) else ""
    if not title or not body:
        stats["noop"] += 1
        return
    try:
        graph.create_node(ctx, "decision", title, body)
        stats["decisions"] += 1
    except Exception as exc:
        _log(f"decision-error title={title!r}: {exc}")


def _apply_pivot_op(ctx, op, stats) -> None:
    """Apply one PIVOT op, resolving each ``supersedes`` ref (title OR id) first.

    Each ``supersedes`` entry may be an integer node id or a node TITLE. Titles
    are resolved via ``graph.resolve_node_ref`` (exact-id -> exact lower(title)
    -> all-tokens FTS): an unambiguous hit yields its node id; a zero- or
    many-candidate ref is dropped LOUDLY (it must never fuzzy-resolve to the
    wrong node and stamp it inactive). Resolved ids are de-duped so the same
    target given twice (once by id, once by title) is superseded once. The pivot
    proceeds only with >=1 resolved target; otherwise it is a loud noop.
    """
    from memory import graph

    title = op.get("title")
    why = op.get("body")
    title = title.strip() if isinstance(title, str) else ""
    why = why.strip() if isinstance(why, str) else ""
    raw_supersedes = op.get("supersedes")
    refs = raw_supersedes if isinstance(raw_supersedes, list) else []
    if not title or not why or not refs:
        stats["noop"] += 1
        return

    resolved: list[int] = []
    for ref in refs:
        node, candidates = graph.resolve_node_ref(ctx, ref)
        if node is None:
            _log(f"pivot-skip: unresolved supersedes ref {ref!r} ({len(candidates)} candidates)")
            continue
        node_id = node["id"]
        if node_id not in resolved:
            resolved.append(node_id)
    if not resolved:
        _log(f"pivot-noop: title={title!r} has no resolvable supersedes target")
        stats["noop"] += 1
        return
    try:
        graph.record_pivot(ctx, title, why, resolved)
        stats["pivots"] += 1
    except Exception as exc:
        _log(f"pivot-error title={title!r}: {exc}")
