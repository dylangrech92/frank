"""Turn-0 flashback: a gated memory seed injected on topic shifts.

Two zero-LLM gates decide whether a new user message opens a fresh topic that
warrants re-seeding recalled memory:

  1. terse gate   -- messages under ``TERSE_MIN_TOKENS`` whitespace tokens are
                     too thin to seed against (checked FIRST).
  2. continuation -- a message whose embedding sits close (cosine >=
                     ``CONTINUATION_THRESHOLD``) to the centroid of the recent
                     conversation is a continuation of the current thread, not a
                     new topic, so nothing is re-seeded. Degrades to terse-only
                     when embeddings are unavailable (FTS-only mode).

When both gates pass, a compact bundle is rendered -- top similarity-recalled
Decisions/Specs (already 1-hop graph-expanded by ``recall_graph``) + <= 3 dated
episode gists + <= 5 atoms -- and stashed on ``session._flashback_block`` for the
duration of the turn. It is injected via a CONTEXT_PROVIDERS entry, NOT persisted
to the transcript. Active **Rules** are deliberately excluded from this bundle:
the P14 rules provider injects them into *every* context unconditionally; the
flashback renderer only reads the rules to de-duplicate, never to re-emit them.
A gate-skip means "no recalled bundle this turn" -- never "no rules".
"""

from __future__ import annotations

import math
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

TERSE_MIN_TOKENS = 8          # messages with FEWER than this many tokens skip (terse)
CONTINUATION_THRESHOLD = 0.55  # cosine >= this vs recent centroid => continuation (skip)
CENTROID_WINDOW = 6           # number of recent prior messages forming the centroid
MAX_DECISIONS = 5             # cap on similarity-recalled Decisions/Specs
MAX_GISTS = 3                 # cap on dated episode gists
MAX_ATOMS = 5                 # cap on atoms


# ---------------------------------------------------------------------------
# Vector helpers (embeddings are L2-normalized, but never assume it)
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 on any degeneracy."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _centroid(vectors: list[list[float]]) -> list[float] | None:
    """Component-wise mean of a non-empty list of equal-length vectors."""
    vectors = [v for v in vectors if v]
    if not vectors:
        return None
    dim = len(vectors[0])
    acc = [0.0] * dim
    n = 0
    for v in vectors:
        if len(v) != dim:
            continue
        for i in range(dim):
            acc[i] += v[i]
        n += 1
    if n == 0:
        return None
    return [x / n for x in acc]


def _recent_texts(messages: list[dict], exclude_last: bool, limit: int) -> list[str]:
    """Return up to *limit* recent message contents (newest-last order preserved).

    When *exclude_last* is True the final message (the current user turn) is
    dropped first so the centroid reflects only the PRIOR conversation.
    """
    msgs = messages[:-1] if (exclude_last and messages) else list(messages)
    texts: list[str] = []
    for m in reversed(msgs):
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(content)
        if len(texts) >= limit:
            break
    texts.reverse()
    return texts


# ---------------------------------------------------------------------------
# Bundle renderer
# ---------------------------------------------------------------------------


def render_bundle(ctx, query: str) -> str:
    """Render the gated recalled-memory bundle (never includes Rules).

    Returns an empty string when nothing relevant is recalled.
    """
    from memory import graph
    from memory.atomic import recall_facts
    from memory.episodic import recall_episodes

    # Rules text is read ONLY to de-duplicate -- rules themselves stay in the
    # always-on P14 provider and must not be repeated in this bundle.
    try:
        rules_text = graph.render_active_rules(ctx) or ""
    except Exception:
        rules_text = ""

    def _seen_in_rules(text: str) -> bool:
        head = (text or "").strip()
        return bool(head) and head in rules_text

    decisions: list[dict] = []
    try:
        for hit in graph.recall_graph(ctx, query, MAX_DECISIONS):
            if hit.get("kind") == "rule":
                continue
            if _seen_in_rules(hit.get("text", "")):
                continue
            decisions.append(hit)
    except Exception:
        decisions = []

    gists: list[dict] = []
    try:
        gists = recall_episodes(ctx, query, MAX_GISTS)
    except Exception:
        gists = []

    atoms: list[dict] = []
    try:
        atoms = recall_facts(ctx, query, MAX_ATOMS)
    except Exception:
        atoms = []

    if not decisions and not gists and not atoms:
        return ""

    lines: list[str] = ["## Recalled context (relevant to your current message)"]

    if decisions:
        lines.append("### Decisions & specs")
        for d in decisions[:MAX_DECISIONS]:
            lines.append(f"- {d.get('text', '').strip()}")

    if gists:
        lines.append("### Recent episodes")
        for g in gists[:MAX_GISTS]:
            lines.append(f"- {g.get('text', '').strip()}")

    if atoms:
        lines.append("### Known facts")
        for a in atoms[:MAX_ATOMS]:
            lines.append(f"- {a.get('text', '').strip()}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Turn-0 seed (fills agent.flashback_maybe_seed)
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"flashback: {msg}", file=sys.stderr, flush=True)


def maybe_seed(session) -> None:
    """Decide-and-seed the turn-0 flashback bundle onto *session*.

    Sets ``session._flashback_block`` to the rendered bundle (seed) or to ""
    (skip). Always logs the gate outcome: seeded / terse-skip / continuation-skip.
    Never raises -- any failure clears the block and returns.
    """
    try:
        messages = getattr(session, "_messages", None) or []
        # The current user message is the most recent one.
        query = ""
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                query = m["content"]
                break

        # --- Gate 1: terse (checked BEFORE the continuation gate) -----------
        if len(query.split()) < TERSE_MIN_TOKENS:
            session._flashback_block = ""
            _log("terse-skip (message below token floor)")
            return

        from memory.recall import get_memory

        root = getattr(session, "project_root", None)
        ctx = get_memory(str(root) if root is not None else None)
        embedder = ctx.embedder

        # --- Gate 2: continuation (only when embeddings are available) ------
        if getattr(embedder, "available", False):
            q_emb = embedder.embed(query)
            prior_texts = _recent_texts(messages, exclude_last=True, limit=CENTROID_WINDOW)
            prior_embs = [e for e in (embedder.embed(t) for t in prior_texts) if e]
            centroid = _centroid(prior_embs)
            if q_emb and centroid is not None:
                sim = _cosine(q_emb, centroid)
                if sim >= CONTINUATION_THRESHOLD:
                    session._flashback_block = ""
                    _log(f"continuation-skip (cosine {sim:.3f} >= {CONTINUATION_THRESHOLD})")
                    return
        # else: FTS-only mode -> continuation gate degraded to terse-only.

        # --- Both gates passed: render + stash the bundle -------------------
        block = render_bundle(ctx, query)
        session._flashback_block = block
        if block:
            n = block.count("\n- ")
            _log(f"seeded ({n} recalled item(s))")
        else:
            _log("seeded (nothing relevant recalled)")
    except Exception as exc:
        try:
            session._flashback_block = ""
        except Exception:
            pass
        _log(f"error: {exc}")


# ---------------------------------------------------------------------------
# Context provider (seam): inject the stashed bundle into the assembled context
# ---------------------------------------------------------------------------


def _flashback_provider(session) -> str:
    """CONTEXT_PROVIDERS entry: emit the turn's stashed flashback block (if any)."""
    return getattr(session, "_flashback_block", "") or ""


_provider_registered = False


def register_flashback_provider() -> None:
    """Append the flashback provider to session.CONTEXT_PROVIDERS (idempotent)."""
    global _provider_registered
    if _provider_registered:
        return
    try:
        import session as _session_mod

        if _flashback_provider not in _session_mod.CONTEXT_PROVIDERS:
            _session_mod.CONTEXT_PROVIDERS.append(_flashback_provider)
    except Exception:
        return
    _provider_registered = True