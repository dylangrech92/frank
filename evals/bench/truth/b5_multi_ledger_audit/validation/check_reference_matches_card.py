#!/usr/bin/env python3
"""Self-validation requirement (c): reference_answer.md must match every
card.json item's regex for bug|dead kinds (recall = 100%) and must match
NO decoy item's regex (precision = 100%). This is exactly the "reference
scores full marks" half of the two-sided validity contract (DESIGN.md,
"Design stance: no toy tasks" / "--calibrate").

As an additional, non-required sanity check, also reports how null_answer.txt
(the lint-level calibration baseline) scores against the same card -- it is
expected to hit only a subset of the T1 items and to trip at least one
decoy, demonstrating the difficulty ceiling is not vacuous.
"""
import json
import re
import sys
from pathlib import Path

TRUTH_DIR = Path(__file__).resolve().parent.parent
CARD = TRUTH_DIR / "card.json"
REFERENCE = TRUTH_DIR / "reference_answer.md"
NULL_ANSWER = TRUTH_DIR / "null_answer.txt"


def matches_any(patterns, text) -> list[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(pattern)
    return hits


def check_reference(card: dict, text: str) -> bool:
    ok = True
    print("=== reference_answer.md vs card.json ===")
    for item in card["items"]:
        hits = matches_any(item["any"], text)
        is_decoy = item["kind"] == "decoy"
        if is_decoy:
            passed = len(hits) == 0
            label = "PASS (correctly absent)" if passed else "FAIL (decoy matched!)"
        else:
            passed = len(hits) > 0
            label = f"PASS (matched: {hits[0]!r})" if passed else "FAIL (no regex matched)"
        print(f"  [{item['kind']:5s} {item['tier']:5s}] {item['id']:45s} {label}")
        ok = ok and passed
    return ok


def summarize_null(card: dict, text: str) -> None:
    print()
    print("=== null_answer.txt vs card.json (informational only) ===")
    tier_totals: dict[str, int] = {}
    tier_hits: dict[str, int] = {}
    decoy_hits = 0
    for item in card["items"]:
        hits = matches_any(item["any"], text)
        if item["kind"] == "decoy":
            if hits:
                decoy_hits += 1
                print(f"  decoy tripped: {item['id']}")
            continue
        tier_totals[item["tier"]] = tier_totals.get(item["tier"], 0) + 1
        if hits:
            tier_hits[item["tier"]] = tier_hits.get(item["tier"], 0) + 1
    for tier in sorted(tier_totals):
        print(f"  tier {tier}: {tier_hits.get(tier, 0)}/{tier_totals[tier]} items recalled")
    print(f"  decoys tripped: {decoy_hits}/3")


def main() -> int:
    card = json.loads(CARD.read_text())
    reference_text = REFERENCE.read_text()

    ok = check_reference(card, reference_text)

    if NULL_ANSWER.exists():
        summarize_null(card, NULL_ANSWER.read_text())

    print()
    if ok:
        print("CONFIRMED: reference_answer.md matches every bug|dead item's regex "
              "and matches no decoy's regex -- recall=100%, precision=100% on the "
              "reference solution.")
        return 0
    print("FAILED: reference_answer.md does not cleanly calibrate against card.json.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
