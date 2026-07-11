"""Task-start orientation: a code-anchored brief instead of a conversation flashback.

Replaces ``memory.flashback`` at the task-start hook (agent.py). Where flashback
recalled *conversation* (episodes, chat-shaped facts) gated by a chat-continuation
heuristic, orientation recalls the *codebase*: a zero-LLM derived skeleton
(``memory.skeleton``) plus anchored, code-linked knowledge atoms and graph
decisions/specs relevant to the current task. It runs every turn (retrieval is
already task-relative, so there is no topic-shift gate to maintain) and never
raises.
"""

from __future__ import annotations

import os
import re
import sys
import time

# Rules are injected separately by the always-on graph rules provider
# (``memory.graph.render_active_rules``); the orientation brief never re-emits them,
# it only reads them to dedupe against, mirroring ``memory.flashback.render_bundle``.

SKELETON_TOKEN_BUDGET = 1000
RECALL_CANDIDATES = 12
RECALL_KEEP = 8
GRAPH_LIMIT = 5
COLD_ATOM_FLOOR = 2  # fewer than this many FRESH anchored atoms => "cold area"

_PATH_OR_SYMBOL_RE = re.compile(r"[\w./-]+\.[A-Za-z]{1,10}(?::\d+)?|\b\w+(?:::|\.)\w+\b")

# In-process guard (D5 / M5): (project_root, normalized task text) pairs the
# lazy gap-fill explorer has already attempted this process, so a repeatedly
# cold-looking area (e.g. a dead LLM endpoint) is retried at most once per
# process instead of re-spawning an explorer on every single turn.
_EXPLORED_AREAS: set[tuple[str, str]] = set()


def _log(msg: str) -> None:
    print(f"orientation: {msg}", file=sys.stderr, flush=True)


def _current_task_text(session) -> str:
    """Return the current turn's user text (last user message in the transcript)."""
    messages = getattr(session, "_messages", None) or []
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _named_paths_and_symbols(text: str) -> set[str]:
    """Best-effort extraction of file paths / dotted symbol names mentioned in *text*."""
    if not text:
        return set()
    return {m.group(0) for m in _PATH_OR_SYMBOL_RE.finditer(text)}


def _augmented_query(task_text: str, named: set[str]) -> str:
    if not named:
        return task_text
    return task_text + " " + " ".join(sorted(named))


def _rerank_and_trim(atoms: list[dict], named: set[str], keep: int) -> list[dict]:
    """Boost atoms anchored to a path/symbol named in the task, then trim to *keep*."""

    def _matches_named(anchor_path: str | None) -> bool:
        if not anchor_path or not named:
            return False
        return any(n in anchor_path or anchor_path.endswith(n) for n in named)

    def _key(atom: dict) -> tuple[int, float]:
        boosted = _matches_named(atom.get("anchor_path"))
        return (0 if boosted else 1, -float(atom.get("score", 0.0)))

    return sorted(atoms, key=_key)[:keep]


def _render_atoms(atoms: list[dict]) -> str:
    if not atoms:
        return ""
    lines = ["## Recalled knowledge (anchored)"]
    for atom in atoms:
        text = (atom.get("text") or atom.get("value") or "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _render_graph_hits(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = ["## Decisions & specs"]
    for hit in hits:
        text = (hit.get("text") or "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _maybe_explore(session, cold_signal: bool, skeleton: str) -> None:
    """Lazy gap-fill explorer (D5 / M5).

    When *cold_signal* is True the current task area has few/no fresh
    anchored atoms: spawn a bounded, memory-less explorer pass
    (``memory.explorer.explore``) over the task area, using *skeleton* (the
    already-derived project map) as its starting point. The PARENT (this
    function) persists the returned brief as anchored, durable knowledge
    atoms (``source='explorer'``) via ``memory.explorer.persist_brief`` --
    so the *next* task in this area is a cheap recall instead of a
    re-exploration -- and appends the brief to ``session._orientation_block``
    so the CURRENT task benefits immediately too.

    Guarded against runaway/recursive exploration two ways: (1) the explorer
    itself never calls orientation/consolidation and never writes memory, so
    it structurally cannot re-trigger this function; (2) an in-process
    ``_EXPLORED_AREAS`` set skips a (project, task-text) pair this process has
    already attempted, so a persistently-cold area (e.g. an unreachable LLM
    endpoint) is retried at most once per process rather than every turn.

    Never raises -- a failing/slow/erroring explorer is logged loudly and
    orientation proceeds with just the skeleton + recall brief already built
    by the caller (this function must not let an internal error propagate,
    since that would otherwise unwind into ``orientation_maybe_seed``'s own
    try/except and wipe the ALREADY-GOOD brief back to empty).
    """
    if not cold_signal:
        return
    try:
        project_root = getattr(session, "project_root", None)
        root_str = str(project_root) if project_root else os.getcwd()
        task_text = _current_task_text(session)

        area_key = (root_str, task_text.strip().lower()[:200])
        if area_key in _EXPLORED_AREAS:
            _log("explore skipped: area already attempted this process")
            return
        _EXPLORED_AREAS.add(area_key)

        _log("cold area -- spawning bounded gap-fill explorer")
        from memory import explorer

        t0 = time.perf_counter()
        brief_text, fallback_anchor = explorer.explore(root_str, task_text, skeleton)
        elapsed = time.perf_counter() - t0

        if not brief_text:
            _log(f"explore produced nothing usable after {elapsed:.1f}s -- proceeding without gap-fill")
            return

        from memory.recall import get_memory

        ctx = get_memory(root_str)
        written = explorer.persist_brief(ctx, root_str, brief_text, fallback_anchor)

        existing = getattr(session, "_orientation_block", "") or ""
        addition = "## Explorer brief (new -- cold area)\n" + brief_text
        session._orientation_block = (existing + "\n\n" + addition) if existing else addition

        _log(f"explore done in {elapsed:.1f}s: wrote {len(written)} atom(s), injected into current turn")
    except Exception as exc:
        _log(f"explore-error: {exc}")


def orientation_maybe_seed(session) -> None:
    """Build this turn's orientation brief and stash it on ``session._orientation_block``.

    Brief = (1) the zero-LLM derived project skeleton, personalised to the
    task text, (2) anchored knowledge atoms recalled by a task-relative query
    (task text + any file paths/symbols named in it), reranked to prefer
    atoms anchored to those named paths, and (3) graph decisions/specs
    relevant to the same query (never Rules -- those are always-on via a
    separate provider). Cheap and non-raising: the skeleton is deterministic
    and zero-LLM, recall is FTS/vector, never an LLM call.
    """
    try:
        task_text = _current_task_text(session)
        project_root = getattr(session, "project_root", None)
        root_str = str(project_root) if project_root else os.getcwd()

        from memory.skeleton import build_skeleton

        skeleton = build_skeleton(root_str, task_text, token_budget=SKELETON_TOKEN_BUDGET)

        named = _named_paths_and_symbols(task_text)
        query = _augmented_query(task_text, named)

        atoms: list[dict] = []
        graph_hits: list[dict] = []
        if query.strip():
            try:
                from memory.recall import get_memory
                from memory.atomic import recall_facts
                from memory import graph

                ctx = get_memory(root_str)

                try:
                    candidates = recall_facts(ctx, query, RECALL_CANDIDATES)
                except Exception:
                    candidates = []
                atoms = _rerank_and_trim(candidates, named, RECALL_KEEP)

                try:
                    rules_text = graph.render_active_rules(ctx) or ""
                except Exception:
                    rules_text = ""

                try:
                    raw_hits = graph.recall_graph(ctx, query, GRAPH_LIMIT)
                except Exception:
                    raw_hits = []

                def _already_a_rule(text: str) -> bool:
                    head = (text or "").strip()
                    return bool(head) and head in rules_text

                graph_hits = [
                    hit
                    for hit in raw_hits
                    if hit.get("kind") != "rule" and not _already_a_rule(hit.get("text", ""))
                ][:GRAPH_LIMIT]
            except Exception as exc:
                _log(f"recall skipped: {exc}")

        blocks = [b for b in (skeleton, _render_atoms(atoms), _render_graph_hits(graph_hits)) if b]
        session._orientation_block = "\n\n".join(blocks)

        fresh_count = sum(1 for atom in atoms if not atom.get("stale"))
        cold_signal = fresh_count < COLD_ATOM_FLOOR
        _maybe_explore(session, cold_signal, skeleton)

        _log(
            f"seeded: skeleton={len(skeleton)} chars, atoms={len(atoms)}, "
            f"decisions={len(graph_hits)}, cold={cold_signal}"
        )
    except Exception as exc:
        try:
            session._orientation_block = ""
        except Exception:
            pass
        _log(f"error: {exc}")


def _orientation_provider(session) -> str:
    return getattr(session, "_orientation_block", "") or ""


_provider_registered = False


def register_orientation_provider() -> None:
    """Idempotently append ``_orientation_provider`` to ``session.CONTEXT_PROVIDERS``."""
    global _provider_registered
    if _provider_registered:
        return
    try:
        import session as _session_mod

        if _orientation_provider not in _session_mod.CONTEXT_PROVIDERS:
            _session_mod.CONTEXT_PROVIDERS.append(_orientation_provider)
    except Exception:
        return
    _provider_registered = True
