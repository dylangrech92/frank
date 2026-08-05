"""perf_report — grades a performance-investigation answer against
truth/<id>/card.json facts (each optionally carrying an int ``rank``, e.g.
"the #1 hotspot") and traps (decoy hotspots that look plausible but are not
where the time/memory actually goes).

Spec fields (must sum to 100):

    recall_weight    float, default 50 -- fraction of true facts named.
    evidence_weight  float, default 30 -- of the *recalled* facts, how many
                      are backed by an adjacent measurement (a number with a
                      unit -- ms, s, %, MiB, KiB, KB, MB, GB, calls, x -- on
                      the same line or the line immediately before/after the
                      matched claim). A named-but-unmeasured claim ("the
                      database is probably slow") is cheap; this band
                      rewards only claims that cite the profiler's own
                      numbers.
    ranking_weight   float, default 10 -- among recalled facts that have a
                      ``rank`` in the card, are they mentioned in the
                      answer in the right relative order? Scored as the
                      fraction of concordant pairs among all pairs of
                      recalled+ranked facts. Needs >=2 recalled+ranked facts
                      to mean anything; with 0 or 1, ranking is undefined,
                      so it scores 0 (not a vacuous 1.0 -- "no engagement,
                      no credit", see graders/__init__.py).
    decoy_weight     float, default 10 -- no trap hotspots named. Only
                      awarded when recall_fraction > 0: a report that names
                      nothing at all trivially avoids every decoy without
                      having demonstrated anything, so it earns nothing
                      here either.
"""

from __future__ import annotations

import re

from ._cards import any_regex_matches, load_card

_EVIDENCE_RE = re.compile(
    r"\d[\d,\.]*\s*(?:(?:ms|s|MiB|KiB|KB|MB|GB|calls?|x)\b|%)",
    re.IGNORECASE,
)


def _has_adjacent_evidence(pattern_hits_line: int | None, lines: list[str]) -> bool:
    if pattern_hits_line is None:
        return False
    window = range(max(0, pattern_hits_line - 1), min(len(lines), pattern_hits_line + 2))
    return any(_EVIDENCE_RE.search(lines[i]) for i in window)


def _first_matching_line(patterns: list[str], lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        if any_regex_matches(patterns, line):
            return idx
    return None


def grade(spec: dict, ctx) -> dict:
    if ctx.truth_dir is None:
        raise ValueError("perf_report grader requires a truth dir")
    card = load_card(ctx.truth_dir)
    facts = card["facts"]
    traps = card["traps"]
    if not facts:
        raise ValueError(f"card.json under {ctx.truth_dir} has no facts")

    recall_weight = spec.get("recall_weight", 50.0)
    evidence_weight = spec.get("evidence_weight", 30.0)
    ranking_weight = spec.get("ranking_weight", 10.0)
    decoy_weight = spec.get("decoy_weight", 10.0)
    total_weight = recall_weight + evidence_weight + ranking_weight + decoy_weight
    if abs(total_weight - 100.0) > 1e-6:
        raise ValueError(
            f"perf_report weights must sum to 100 (recall={recall_weight} evidence={evidence_weight} "
            f"ranking={ranking_weight} decoy={decoy_weight} total={total_weight})"
        )

    answer = ctx.answer or ""
    lines = answer.splitlines()

    matched_line: dict[str, int | None] = {}
    for f in facts:
        matched_line[f["id"]] = _first_matching_line(f["any"], lines)
    recalled = [f for f in facts if matched_line[f["id"]] is not None]
    recall_fraction = len(recalled) / len(facts)
    recall_score = recall_weight * recall_fraction

    if recalled:
        evidenced = [f for f in recalled if _has_adjacent_evidence(matched_line[f["id"]], lines)]
        evidence_fraction = len(evidenced) / len(recalled)
    else:
        evidenced = []
        evidence_fraction = 0.0
    evidence_score = evidence_weight * evidence_fraction

    ranked_recalled = [f for f in recalled if isinstance(f.get("rank"), int)]
    if len(ranked_recalled) >= 2:
        pairs = 0
        concordant = 0
        for i in range(len(ranked_recalled)):
            for j in range(i + 1, len(ranked_recalled)):
                a, b = ranked_recalled[i], ranked_recalled[j]
                pairs += 1
                true_order = a["rank"] < b["rank"]
                seen_order = matched_line[a["id"]] < matched_line[b["id"]]
                if true_order == seen_order:
                    concordant += 1
        ranking_fraction = concordant / pairs if pairs else 0.0
    else:
        ranking_fraction = 0.0
    ranking_score = ranking_weight * ranking_fraction

    triggered_traps = [t for t in traps if any_regex_matches(t["any"], answer)]
    if recall_fraction > 0:
        decoy_fraction = 0.0 if triggered_traps else 1.0
    else:
        decoy_fraction = 0.0
    decoy_score = decoy_weight * decoy_fraction

    score = max(0.0, min(100.0, recall_score + evidence_score + ranking_score + decoy_score))

    return {
        "score": score,
        "details": {
            "recalled_facts": sorted(f["id"] for f in recalled),
            "missed_facts": sorted(f["id"] for f in facts if f["id"] not in {r["id"] for r in recalled}),
            "recall_fraction": recall_fraction,
            "evidenced_facts": sorted(f["id"] for f in evidenced),
            "evidence_fraction": evidence_fraction,
            "ranking_fraction": ranking_fraction,
            "ranked_recalled_count": len(ranked_recalled),
            "triggered_traps": [t["id"] for t in triggered_traps],
            "decoy_fraction": decoy_fraction,
        },
    }
