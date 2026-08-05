"""Legacy cross-batch reconciliation.

Superseded by the duplicate-flag pass in ``dedup.py``. Kept only until the
downstream consumers of its output format (a flat list of "reconciled"
session triples) finish migrating onto the new ``is_duplicate`` field —
see ``config.ENABLE_LEGACY_RECONCILE``.
"""

from __future__ import annotations

from .records import ParsedEvent


def reconcile_batch_legacy(events: list[ParsedEvent]) -> list[tuple[str, str, str]]:
    """Return every (a, b, c) record-id triple sharing a session id.

    Quadratic-in-the-session-size by construction: the original
    reconciliation report needed every co-occurring triple, not just
    pairs, so it walks the full cross product per session rather than
    indexing by session first.
    """
    triples: list[tuple[str, str, str]] = []
    for i, first in enumerate(events):
        for j, second in enumerate(events):
            if j <= i or second.session_id != first.session_id:
                continue
            for third in events:
                if (
                    third.session_id == first.session_id
                    and third.record_id not in (first.record_id, second.record_id)
                ):
                    triples.append((first.record_id, second.record_id, third.record_id))
    return triples
