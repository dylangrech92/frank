"""Shared truth/<id>/card.json loading + tolerant regex matching.

Internal helper module (leading underscore) used by the grader kinds that
read a ground-truth card: answer_facts, findings_list, perf_report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class CardError(Exception):
    """card.json is missing, unreadable, or not shaped like a card."""


def load_card(truth_dir: Path) -> dict:
    path = truth_dir / "card.json"
    if not path.is_file():
        raise CardError(f"missing card.json under {truth_dir}")
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CardError(f"malformed card.json under {truth_dir}: {exc}") from exc
    if not isinstance(card, dict):
        raise CardError(f"card.json under {truth_dir} must be a JSON object")
    card.setdefault("facts", [])
    card.setdefault("traps", [])
    card.setdefault("items", [])
    return card


def any_regex_matches(patterns: list[str], text: str) -> bool:
    """Case-insensitive OR-match against *text*.

    Every result is wrapped in ``bool()`` before it can reach a result dict —
    a bare ``re.Match`` is not JSON-serializable, a previously recorded
    failure mode in this project's other regex-based scorers.
    """
    return bool(any(re.search(p, text, re.IGNORECASE) is not None for p in patterns))
