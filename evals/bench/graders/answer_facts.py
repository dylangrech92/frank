"""answer_facts — card.json fact recall + trap penalties (+ optional coherence).

Spec fields (beyond the common ``kind``/``weight``/``requires``/``gate``):

    recall_weight     float, default 100 — points (of this grader's own
                       0-100 score) allocated to fact recall.
    coherence_weight  float, default 0 — points allocated to a cheap
                       structural "does this read like a narrative, not a
                       fact dump" heuristic. This is a screen, not a
                       judgment: DESIGN.md is explicit that coherence is
                       "screened by graders, confirmed by the human read" —
                       do not treat a high coherence_fraction as proof of a
                       good answer, only as evidence it is not disqualified.
    tiers             optional list[str] — restrict which card facts/traps
                       count (default: every fact/trap in the card).

Per the shared rule in ``graders/__init__.py``: coherence (a secondary
signal) awards nothing when recall (the primary signal) is zero, so an
empty answer cannot buy points by trivially reading as "structured."
"""

from __future__ import annotations

import re

from ._cards import any_regex_matches, load_card

_COHERENCE_CONNECTORS = re.compile(
    r"\b(because|therefore|so that|as a result|which (?:then|in turn)|"
    r"leads? to|causes?|then|first|next|finally)\b",
    re.IGNORECASE,
)


def _coherence_fraction(answer: str) -> float:
    stripped = answer.strip()
    if len(stripped) < 200:
        return 0.0
    signals = 0
    signals += 1  # length threshold already cleared above
    if len(re.findall(r"[.!?]\s|\n", answer)) >= 2:
        signals += 1
    if _COHERENCE_CONNECTORS.search(answer):
        signals += 1
    return signals / 3.0


def grade(spec: dict, ctx) -> dict:
    if ctx.truth_dir is None:
        raise ValueError("answer_facts grader requires a truth dir")
    card = load_card(ctx.truth_dir)
    tiers = spec.get("tiers")
    facts = [f for f in card["facts"] if tiers is None or f.get("tier") in tiers]
    traps = [t for t in card["traps"] if tiers is None or t.get("tier") in tiers]
    if not facts:
        raise ValueError(f"card.json under {ctx.truth_dir} has no facts (tiers filter={tiers})")

    answer = ctx.answer or ""

    matched = [f for f in facts if any_regex_matches(f["any"], answer)]
    matched_ids = {f["id"] for f in matched}
    total_weight = sum(f.get("weight", 1) for f in facts)
    matched_weight = sum(f.get("weight", 1) for f in matched)
    recall_fraction = (matched_weight / total_weight) if total_weight else 0.0

    triggered = [t for t in traps if any_regex_matches(t["any"], answer)]
    penalty = sum(t.get("penalty", 0) for t in triggered)

    recall_weight = spec.get("recall_weight", 100.0)
    coherence_weight = spec.get("coherence_weight", 0.0)
    coherence_fraction = _coherence_fraction(answer) if (coherence_weight and recall_fraction > 0) else 0.0

    score = recall_weight * recall_fraction + coherence_weight * coherence_fraction - penalty
    score = max(0.0, min(100.0, score))

    return {
        "score": score,
        "details": {
            "matched_facts": sorted(matched_ids),
            "missed_facts": sorted(f["id"] for f in facts if f["id"] not in matched_ids),
            "triggered_traps": [t["id"] for t in triggered],
            "recall_fraction": recall_fraction,
            "coherence_fraction": coherence_fraction,
            "penalty": penalty,
        },
    }
