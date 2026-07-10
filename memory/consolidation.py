"""End-of-task consolidation: durable knowledge atoms from what this task did.

Replaces ``memory.episodic``'s episode-gist mining as the source consolidation
reconciles from. Where the old pipeline mined a compressed *conversation*
gist, this module mines the accepted turn's **code diff + transcript tail**
directly -- concepts, vocabulary, cross-cutting flows, invariants, gotchas,
and the "why" behind a change, never pure structural facts (that is the
zero-LLM derived skeleton's job, ``memory.skeleton``). Reconciliation reuses
the same ADD/UPDATE/DELETE/NOOP shape as ``memory.atomic.extract_facts``
against the top-k most similar existing atoms, but every write is
code-anchored (``source='consolidation'``) and genuine design reasoning /
reversals route to the graph layer instead of the facts table. See
MEMORY_REDESIGN.md section 8.

Runs off the hot path: the turn-end hook in agent.py
(``consolidation_maybe_extract``) only *enqueues* a snapshot of this turn onto
the background writer below and returns immediately; ``main.session_end_jobs``
drains that same queue at teardown so the last turn's pass is guaranteed to
finish before the process exits, in both one-shot and REPL modes, without
ever firing twice for the same turn.
"""

from __future__ import annotations

import difflib
import os
import queue
import re
import subprocess
import sys
import threading
from typing import TYPE_CHECKING

from memory import anchor as _anchor

if TYPE_CHECKING:
    from memory.recall import MemoryContext

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

TRANSCRIPT_TAIL_MESSAGES = 12  # how many recent transcript messages to mine
TOP_K_SIMILAR = 8              # existing atoms shown to the model for reconciliation
GRAPH_CANDIDATES = 5           # existing decisions/specs shown for dedupe + pivot targets
MAX_DIFF_CHARS = 4000          # per-file diff cap fed to the extractor
MAX_MINED_CHARS = 8000         # total diff+transcript cap embedded in the prompt
CONFIDENCE_FLOOR = 0.45        # ADD/UPDATE below this confidence is dropped (poisoning defence)
KEY_REUSE_SIMILARITY = 0.62    # min (key+value) similarity to reuse a live atom's key (see _best_existing_key_match)

_CONSOLIDATION_SYSTEM_PROMPT = """You maintain TWO durable records for a software project, from one completed task's code diff and/or transcript tail:

1. KNOWLEDGE ATOMS -- reusable concepts, vocabulary, cross-cutting flows, invariants, conventions, gotchas, and domain facts learned by investigation (counts, inventories, configuration values, capabilities, behavior). NOT pure code-location trivia like "the function at line N does Z" or "X is defined at path:line" -- a separate, zero-LLM project skeleton already answers exactly where symbols live. A file path may still appear as an atom's anchor; the ban is on facts whose ONLY content is a location.
2. THE DECISION RECORD -- the project's design decisions and the reversals of them. When this task chose one approach over alternatives, adopted a constraint, or settled a design question, that is a DECISION. When this task overturns an earlier decision, that is a PIVOT.

You are shown the diff/transcript for this task plus the existing atoms and decisions most similar to it. Decide what durable knowledge AND what decisions this task establishes, then return operations against the store.

Return ONLY a JSON array of operation objects (no prose, no markdown fences). Each object is ONE of:

Atom operation:
{"op": "ADD" | "UPDATE" | "DELETE" | "NOOP", "kind": "project" | "convention", "key": "short-stable-identifier", "value": "the durable insight, stated atomically", "confidence": 0.0-1.0, "anchor_path": "relative/path/to/file.ext" or null}

Decision operation:
{"op": "DECISION", "title": "short imperative title", "body": "what was chosen AND which alternative was rejected and why"}

Pivot operation (this task reversed or replaced a decision listed under "Existing decisions" below):
{"op": "PIVOT", "title": "short imperative title", "body": "what changed and why it overturns the prior decision", "supersedes": ["exact title copied from the existing-decisions block", 12]}

Rules:
- ADD: a new durable, reusable fact not already present below -- including a concrete count/inventory/configuration value the task discovered (e.g. "this project registers N of X"), not only abstract concepts. Example NOOP-worthy: "I checked the file and found the function" (transient narration, no reusable content).
- UPDATE: this task refines, corrects, or SUPERSEDES an atom shown below -- reuse that atom's EXACT key.
- DELETE: an atom shown below is now wrong/abandoned -- give its exact key.
- NOOP: nothing durable (transient chatter, restates something already known). Prefer NOOP over a low-value ADD, but a concrete fact the user or agent explicitly established this task is NOT low-value merely for being short.
- If a later reply in the transcript CORRECTS an earlier claim (yours or an existing atom shown below), you MUST emit an UPDATE (same key when an existing atom matches) with the corrected value -- never leave a superseded wrong fact standing.
- confidence: how durable/important/reusable this atom is for a FUTURE task (0=trivial, 1=core invariant). Be conservative.
- anchor_path: the single file this insight is most about, if the transcript/diff names one; else null (repo-wide).
- kind is ALWAYS "project" (a fact about this codebase) or "convention" (an observed coding convention/style).
- DECISION: emit one when the task chose an approach among alternatives, adopted a constraint, or settled a design question -- name the rejected alternative and why in the body. Do NOT emit a decision for a mechanical edit (a rename, a value tweak, a formatting change). A few real decisions beat many noisy ones.
- PIVOT: emit one only when the task reverses or replaces a decision shown in the existing-decisions block. Each "supersedes" entry is the TITLE of that decision copied verbatim from the block (preferred) or its [id].
- Prefer FEW high-value atoms and decisions over many trivial ones."""

_NAMED_PATH_RE = re.compile(r"[\w./-]+\.[A-Za-z]{1,10}(?::\d+)?")


def _log(msg: str) -> None:
    print(f"consolidation: {msg}", file=sys.stderr, flush=True)


def _file_diff(path: str) -> str | None:
    """Best-effort ``git diff HEAD`` for *path*; None on any failure/empty diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--", path],
            cwd=os.path.dirname(path) or ".",
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    diff = result.stdout
    if not diff.strip():
        return None
    return diff[:MAX_DIFF_CHARS]


def _transcript_tail(messages: list, limit: int) -> str:
    """Render the last *limit* transcript messages with string content as labelled lines."""
    tail = messages[-limit:] if messages else []
    lines = []
    for message in tail:
        role = message.get("role", "?")
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)


def _named_paths(text: str) -> set[str]:
    """Best-effort extraction of file-path-shaped tokens mentioned in *text*."""
    if not text:
        return set()
    return {m.group(0) for m in _NAMED_PATH_RE.finditer(text)}


def _coerce_kind(raw) -> str:
    """Clamp a mined atom's kind to a durable, non-TTL kind (never discovery/misc)."""
    if isinstance(raw, str) and raw.strip().lower() in ("project", "convention"):
        return raw.strip().lower()
    return "project"


def _resolve_anchor_path(project_root: str, raw_path) -> str | None:
    """Resolve a model-supplied path against *project_root* to an absolute path.

    Anchors must be absolute: ``memory.anchor.is_stale`` re-opens the path
    directly at recall time, regardless of the process's current working
    directory. Returns None (repo-wide) for anything not a non-empty string.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = raw_path.strip()
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(project_root, path))


def _normalize_for_match(text: str) -> str:
    """Lowercase + collapse whitespace, for similarity comparison only."""
    return " ".join(text.lower().split())


def _decision_line(hit: dict) -> str:
    """Render one existing decision/spec/pivot as a line whose title is quotable.

    ``graph.recall_graph`` returns the node title under ``value`` and the full
    ``[kind] title: body`` render under ``text``; the model must be able to copy
    the title VERBATIM into a PIVOT ``supersedes`` entry, so the title is quoted
    explicitly and the body appended for context.
    """
    node_id = hit.get("id")
    kind = hit.get("kind")
    title = (hit.get("value") or "").strip()
    text = hit.get("text") or ""
    body = text.split(": ", 1)[1].strip() if ": " in text else ""
    line = f'- [{node_id}] {kind} titled "{title}"'
    return f"{line} -- {body}" if body else line


def _best_existing_key_match(key: str, value: str, existing_atoms: list[dict]) -> str | None:
    """Best-effort reuse of a live atom's key for a same-subject fact whose
    key the model chose differently this time.

    Two independent weak-LLM key choices for the same underlying fact need
    not match (e.g. a correction turn inventing ``abilities_total`` where the
    original wrong atom lives under ``chalie.abilities.count``). Compares the
    proposed ``key + value`` text against each existing atom's ``key + value``
    text via ``difflib.SequenceMatcher`` on lowercased, whitespace-normalized
    strings, and returns the single best match's key if its ratio clears
    ``KEY_REUSE_SIMILARITY``. Conservative by design -- distinct facts about a
    similar broad topic (e.g. a count vs. a registry path vs. an unrelated
    module's purpose) must stay separate, not merge. Returns None if nothing
    clears the bar, ``existing_atoms`` is empty, or on any failure.
    """
    if not existing_atoms:
        return None
    try:
        proposed = _normalize_for_match(f"{key} {value}")
        best_key: str | None = None
        best_ratio = 0.0
        for atom in existing_atoms:
            existing_key = atom.get("key")
            if not isinstance(existing_key, str) or not existing_key:
                continue
            candidate = _normalize_for_match(f"{existing_key} {atom.get('value') or ''}")
            ratio = difflib.SequenceMatcher(None, proposed, candidate).ratio()
            if ratio > best_ratio:
                best_ratio, best_key = ratio, existing_key
        if best_key is not None and best_ratio >= KEY_REUSE_SIMILARITY:
            _log(f"key-reuse: model_key={key!r} -> existing_key={best_key!r} (ratio={best_ratio:.2f})")
            return best_key
        return None
    except Exception:
        return None


def _new_stats() -> dict:
    """Return a fresh observability/stats dict for one consolidation pass."""
    return {
        "ran": False,
        "reason": "not-run",
        "processed": 0,
        "added": 0,
        "updated": 0,
        "deleted": 0,
        "noop": 0,
        "decisions": 0,
        "pivots": 0,
        "skipped_low_confidence": 0,
    }


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
    # MEMORY_REDESIGN.md / consolidation module docstring). Try the model's own
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


def apply_ops(ctx, ops, *, project_root, existing_atoms=None, stats=None) -> dict:
    """Apply a parsed list of consolidation ops against the memory stores.

    The seam the deterministic eval drives directly: no LLM call and no diff
    mining -- just dispatch of the ADD/UPDATE/DELETE/NOOP atom ops and the
    DECISION/PIVOT graph ops to their per-op handlers. *existing_atoms* feeds the
    key-reuse fallback for ADD/UPDATE (the top-k atoms the model was shown);
    *stats* is created fresh when not supplied. Returns the stats dict.
    """
    if stats is None:
        stats = _new_stats()
    existing_atoms = existing_atoms or []
    for op in ops:
        if not isinstance(op, dict):
            continue
        action = str(op.get("op", "")).strip().upper()
        if action in ("ADD", "UPDATE"):
            _apply_atom_op(
                ctx, op, action, project_root=project_root, existing_atoms=existing_atoms, stats=stats
            )
        elif action == "DELETE":
            _apply_delete_op(ctx, op, stats)
        elif action == "DECISION":
            _apply_decision_op(ctx, op, stats)
        elif action == "PIVOT":
            _apply_pivot_op(ctx, op, stats)
        else:
            stats["noop"] += 1
    return stats


def consolidate(ctx: "MemoryContext", client, session) -> dict:
    """Reconcile one completed turn's diff + transcript tail into durable atoms.

    Gathers (a) diffs for each path in ``session.turn_report['files_changed']``
    (if any) and (b) a copy of the recent transcript tail from
    ``session._messages`` -- mining the transcript tail even when no files
    changed, so a read-only turn's own findings (and any user corrections to
    them) still become durable knowledge. One LLM call reconciles the mined
    text against the top-k most similar existing atoms and decisions (the
    same ADD/UPDATE/DELETE/NOOP shape as ``memory.atomic.extract_facts``) and
    applies the result via ``memory.atomic.remember``/``forget`` -- WITH code
    anchors (``source='consolidation'``; a repo-wide insight gets
    ``anchor_path=None``). Genuine design "why" and reversals route to
    ``memory.graph.create_node``/``record_pivot`` instead of the facts table.
    A confidence floor on ADD/UPDATE guards against writing low-value chatter
    (poisoning defence). *session* only needs ``turn_report``, ``_messages``,
    and ``project_root`` -- a real ``Session`` or a ``_SessionSnapshot`` both
    work. Never raises.

    Returns an observability dict: ``{"ran", "reason", "processed", "added",
    "updated", "deleted", "noop", "decisions", "pivots",
    "skipped_low_confidence"}``.
    """
    stats = _new_stats()
    try:
        project_root = str(getattr(session, "project_root", "") or os.getcwd())
        turn_report = getattr(session, "turn_report", None) or {}
        files_changed = turn_report.get("files_changed") or []
        messages = getattr(session, "_messages", None) or []

        changed_paths = sorted(
            {e.get("path") for e in files_changed if isinstance(e, dict) and e.get("path")}
        )
        diff_blocks = []
        for path in changed_paths:
            diff = _file_diff(path)
            diff_blocks.append(
                f"--- diff: {path} ---\n{diff}" if diff else f"--- touched (no diff available): {path} ---"
            )

        transcript_tail = _transcript_tail(messages, TRANSCRIPT_TAIL_MESSAGES)
        mined_text = "\n\n".join(b for b in (*diff_blocks, transcript_tail) if b)
        if not mined_text.strip():
            stats["reason"] = "nothing-to-mine"
            return stats

        from memory.atomic import recall_facts
        from memory import graph

        try:
            existing_atoms = recall_facts(ctx, mined_text[:MAX_MINED_CHARS], TOP_K_SIMILAR)
        except Exception:
            existing_atoms = []
        existing_block = (
            "\n".join(
                f'- kind={a.get("kind")} key={a.get("key")!r}: {a.get("value")}'
                for a in existing_atoms
            )
            or "(none)"
        )

        try:
            existing_decisions = [
                hit
                for hit in graph.recall_graph(ctx, mined_text[:MAX_MINED_CHARS], GRAPH_CANDIDATES)
                if hit.get("kind") != "rule"
            ]
        except Exception:
            existing_decisions = []
        decisions_block = "\n".join(_decision_line(d) for d in existing_decisions) or "(none)"

        named = sorted(_named_paths(mined_text))
        user_msg = (
            f"Files changed this task: {', '.join(changed_paths) or '(none -- read-only turn)'}\n\n"
            f"Diff + transcript tail for this task:\n{mined_text[:MAX_MINED_CHARS]}\n\n"
            f"Existing knowledge atoms most similar to this task:\n{existing_block}\n\n"
            f"Existing decisions/specs most similar to this task:\n{decisions_block}\n\n"
            f"File paths visible above (candidates for anchor_path): {', '.join(named) or '(none)'}"
        )
        llm_messages = [
            {"role": "system", "content": _CONSOLIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        from memory.episodic import _safe_json_array

        try:
            resp = client.chat(llm_messages, tools=None)
            raw = resp.text
        except Exception as exc:
            stats["reason"] = f"llm-error: {exc}"
            return stats

        stats["ran"] = True
        stats["processed"] = 1
        ops = _safe_json_array(raw)
        apply_ops(ctx, ops, project_root=project_root, existing_atoms=existing_atoms, stats=stats)

        stats["reason"] = "ok"
        return stats
    except Exception as exc:
        stats["reason"] = f"error: {exc}"
        _log(f"error: {exc}")
        return stats


# ---------------------------------------------------------------------------
# Off-thread writer: mirrors memory.episodic's single-writer queue pattern so
# a REPL turn never blocks on consolidation's LLM call.
# ---------------------------------------------------------------------------


class _SessionSnapshot:
    """Minimal duck-typed stand-in for a ``Session``, safe to hand to a
    background thread: a frozen copy of just the fields ``consolidate`` reads,
    taken at enqueue time so a background pass never races the live session's
    next turn mutating ``_messages``/``turn_report``.
    """

    __slots__ = ("turn_report", "_messages", "project_root")

    def __init__(self, turn_report: dict, messages: list, project_root: str) -> None:
        self.turn_report = turn_report
        self._messages = messages
        self.project_root = project_root


_store_cache: dict[str, tuple] = {}  # project_root -> (store, embedder), worker-thread-only

_queue: "queue.Queue" = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_last_run: dict | None = None
_last_run_lock = threading.Lock()


def _worker_context(project_root: str) -> "MemoryContext":
    """Open (and cache, per project root) a store/embedder pair for the worker
    thread -- NEVER the main thread's ``memory.recall.get_memory`` cache, whose
    sqlite connection is bound to whichever thread first created it.
    """
    cached = _store_cache.get(project_root)
    if cached is not None:
        store, embedder = cached
    else:
        from memory.embedding import EmbeddingService
        from memory.store import open_store

        store = open_store(project_root)
        model_path = os.environ.get("CODING_AGENT_EMBED_MODEL") or None
        embedder = EmbeddingService(model_path=model_path)
        _store_cache[project_root] = (store, embedder)

    from memory.recall import MemoryContext

    return MemoryContext(store=store, embedder=embedder, project_root=project_root)


def _ensure_worker() -> None:
    """Start the consolidation writer worker if not already running."""
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_drain, name="consolidation-writer", daemon=True)
            _worker.start()


def _drain() -> None:
    """Drain items from the queue and run ``consolidate`` on each, on this thread."""
    global _last_run
    while True:
        item = _queue.get()
        try:
            project_root, snapshot, client = item
            ctx = _worker_context(project_root)
            result = consolidate(ctx, client, snapshot)
            with _last_run_lock:
                _last_run = result
        except Exception as exc:
            with _last_run_lock:
                _last_run = {"ran": False, "reason": f"worker-error: {exc}"}
        finally:
            _queue.task_done()


def enqueue(project_root: str, turn_report: dict, messages_tail: list, client) -> None:
    """Enqueue one completed turn's snapshot for off-thread consolidation."""
    _ensure_worker()
    snapshot = _SessionSnapshot(turn_report, messages_tail, project_root)
    _queue.put((project_root, snapshot, client))


def drain_and_join(timeout: float | None = None) -> None:
    """Block until the queue is empty and every enqueued pass has completed.

    Called once at session-end (both one-shot and REPL) so the final turn's
    consolidation is guaranteed to finish before the process exits -- this is
    the ONLY place a one-shot task's single enqueued pass is waited on, so it
    is never separately re-invoked (and so never double-fires).
    """
    try:
        _queue.join()
    except Exception:
        pass


def pop_last_run() -> dict | None:
    """Return (and clear) the most recent observability dict from the worker."""
    global _last_run
    with _last_run_lock:
        result = _last_run
        _last_run = None
        return result
