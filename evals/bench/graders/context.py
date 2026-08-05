"""Grading context + filesystem tree diffing.

Deliberately stdlib-only and git-free: the same ``diff_trees`` helper grades
a live runner's git-initialized temp copy, a calibration tree built by
overlaying ``truth/<id>/reference/`` on a plain fixture copy, and a
``_selftest.py`` fixture living nowhere near a git repo. Using ``git diff``
would work for the first case but not the other two.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

# Harness-owned state: never part of what a grader diffs or inspects. Mirrors
# the exclusion the live runner writes into .git/info/exclude (see run.py).
_EXCLUDED_DIRS = {".git", ".coding_agent"}


@dataclass
class GradeContext:
    """Everything one grader invocation needs.

    Built once per task x rep (live runs) or once per reference/null case
    (``--calibrate``), then threaded through every grader in the task's
    ``graders`` list in order. ``results`` accumulates each grader's own
    result dict as it runs, so a later grader can read what an earlier one
    found (see ``spec['requires']`` and ``envelope_guard``'s acceptance
    cross-check, both documented in ``graders/__init__.py``).
    """

    task: dict
    tree: Path
    baseline: Path
    envelope: dict | None
    answer: str
    truth_dir: Path | None
    results: list = field(default_factory=list)


@dataclass
class TreeDiff:
    added: list
    removed: list
    modified: list

    @property
    def changed_paths(self) -> list:
        return sorted(self.added + self.removed + self.modified)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified)


def _snapshot(root: Path) -> dict:
    """Return {relpath (posix) -> sha256 hex} for every file under *root*,
    excluding harness-owned dirs. Empty dict for a missing/empty root."""
    if not root.exists():
        return {}
    out: dict[str, str | None] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            try:
                out[rel] = hashlib.sha256(full.read_bytes()).hexdigest()
            except OSError:
                # Unreadable (broken symlink, permission denied) still counts
                # as present-and-different rather than silently vanishing.
                out[rel] = None
    return out


def diff_trees(baseline: Path, tree: Path) -> TreeDiff:
    """Diff *tree* against *baseline*, excluding .git/ and .coding_agent/."""
    before = _snapshot(baseline)
    after = _snapshot(tree)
    added = sorted(p for p in after if p not in before)
    removed = sorted(p for p in before if p not in after)
    modified = sorted(p for p in after if p in before and after[p] != before[p])
    return TreeDiff(added=added, removed=removed, modified=modified)
