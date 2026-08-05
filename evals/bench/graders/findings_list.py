"""findings_list — tier-weighted recall + precision over a free-text list of
findings (audit/discovery-style answers: "here's what I found wrong").

The answer is parsed into discrete claims by line -- numbered ("1.", "1)",
"1:") or bulleted ("-", "*") list items. Each claim is matched against
truth/<id>/card.json's ``items`` (kind "bug" or "dead" = a real, creditable
finding; kind "decoy" = a planted trap that looks real but isn't).

Spec fields:

    tier_weights      dict[str, float] e.g. {"T1": 20, "T2": 30, "T3": 30}
                       -- recall points per tier, keyed to each item's
                       "tier" in card.json.
    precision_weight  float e.g. 20 -- points for claims that are actually
                       correct (not decoys, not noise). tier_weights and
                       precision_weight together must sum to 100.

Recall per tier = (recalled items in that tier) / (total items in that
tier); a tier absent from the card scores 0 for its slice rather than
raising, so a truth author can add T3 items later without every existing
task needing a tier_weights edit -- but a tier declared in tier_weights with
zero items in the card at all is almost certainly a truth-authoring typo,
so that DOES raise.

Precision = (claims that match a real item) / (total claims parsed). Zero
parsed claims scores 0 precision, not a vacuous 1.0 -- "no engagement, no
credit" (see graders/__init__.py): an answer that names nothing cannot be
maximally precise about nothing.
"""

from __future__ import annotations

import re

from ._cards import any_regex_matches, load_card

_FINDING_LINE_RE = re.compile(r"^\s*(?:\d+[\.\):]|[-*])\s*(.+)$")


def _parse_claims(answer: str) -> list[str]:
    claims = []
    for line in answer.splitlines():
        m = _FINDING_LINE_RE.match(line)
        if m:
            claim = m.group(1).strip()
            if claim:
                claims.append(claim)
    return claims


def grade(spec: dict, ctx) -> dict:
    if ctx.truth_dir is None:
        raise ValueError("findings_list grader requires a truth dir")
    card = load_card(ctx.truth_dir)
    items = card["items"]
    if not items:
        raise ValueError(f"card.json under {ctx.truth_dir} has no items")

    tier_weights = spec.get("tier_weights")
    precision_weight = spec.get("precision_weight", 0.0)
    if not tier_weights:
        raise ValueError("findings_list requires a non-empty spec['tier_weights']")
    if abs(sum(tier_weights.values()) + precision_weight - 100.0) > 1e-6:
        raise ValueError(
            f"findings_list tier_weights {tier_weights} + precision_weight {precision_weight} must sum to 100"
        )

    real_items = [it for it in items if it.get("kind") in ("bug", "dead")]
    decoy_items = [it for it in items if it.get("kind") == "decoy"]
    if not real_items:
        raise ValueError(f"card.json under {ctx.truth_dir} has no real (bug/dead) items")

    for tier in tier_weights:
        if not any(it.get("tier") == tier for it in real_items):
            raise ValueError(f"tier_weights declares {tier!r} but no real item in card.json has that tier")

    answer = ctx.answer or ""
    claims = _parse_claims(answer)

    recalled_ids = set()
    for it in real_items:
        if any_regex_matches(it["any"], answer):
            recalled_ids.add(it["id"])

    tier_recall: dict[str, float] = {}
    recall_score = 0.0
    for tier, weight in tier_weights.items():
        tier_items = [it for it in real_items if it.get("tier") == tier]
        tier_recalled = [it for it in tier_items if it["id"] in recalled_ids]
        fraction = (len(tier_recalled) / len(tier_items)) if tier_items else 0.0
        tier_recall[tier] = fraction
        recall_score += weight * fraction

    correct_claims = 0
    for claim in claims:
        if any(any_regex_matches(it["any"], claim) for it in real_items):
            correct_claims += 1
    precision_fraction = (correct_claims / len(claims)) if claims else 0.0
    precision_score = precision_weight * precision_fraction

    triggered_decoys = []
    for claim in claims:
        for decoy in decoy_items:
            if any_regex_matches(decoy["any"], claim):
                triggered_decoys.append(decoy["id"])
                break

    score = max(0.0, min(100.0, recall_score + precision_score))

    return {
        "score": score,
        "details": {
            "claims_parsed": len(claims),
            "recalled_items": sorted(recalled_ids),
            "missed_items": sorted(it["id"] for it in real_items if it["id"] not in recalled_ids),
            "tier_recall": tier_recall,
            "precision_fraction": precision_fraction,
            "triggered_decoys": sorted(set(triggered_decoys)),
        },
    }
