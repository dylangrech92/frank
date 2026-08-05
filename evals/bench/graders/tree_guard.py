"""tree_guard — diffs the post-run tree against the pre-run baseline, always
excluding .git/ and .coding_agent/ (harness-owned state, see
graders/context.py).

Spec fields:

    mode  one of:
      "byte_identical"      full marks iff the diff is empty. The research
                             vertical's contract: "a run that mutates the
                             fixture scores 0 regardless of answer quality" —
                             pair with spec['gate'] = true (see
                             graders/__init__.py) to make that a veto on the
                             whole task, not just this grader's own weight.
      "confined_diff"       full marks iff every changed path is inside
                             spec['allowed_paths'] (prefix match) AND at
                             least one path changed. An empty diff scores 0:
                             confinement isn't satisfied by never attempting
                             the task (the shared "no engagement, no credit"
                             rule — see graders/__init__.py). Violating paths
                             cost points proportionally.
      "pollution_whitelist"  full marks iff none of the newly *added* paths
                             match spec['pollution_patterns'] (fnmatch
                             globs) AND at least one path changed. Meant for
                             greenfield tasks (fixture=None) where there is
                             no fixed allowed-paths list, only known-junk
                             patterns (__pycache__/, *.pyc, *.log, ...) that
                             must not be left behind by the agent's own
                             verification runs.
"""

from __future__ import annotations

import fnmatch

from .context import diff_trees

# Interpreter/tooling residue ignored by byte_identical and confined_diff:
# the harness lints after every edit (.ruff_cache/) and running the task's
# own repro command compiles bytecode (__pycache__/*.pyc) — neither is an
# agent edit, and grading them as violations punishes the agent for doing
# the task. pollution_whitelist deliberately does NOT filter: its patterns
# are the task's explicit leave-no-trace contract, and filtering here would
# make that check vacuous.
_EPHEMERAL_DIRS = frozenset({"__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache"})


def _is_ephemeral(path: str) -> bool:
    if path.endswith(".pyc"):
        return True
    return any(part in _EPHEMERAL_DIRS for part in path.split("/")[:-1])


def _is_within(path: str, allowed: str) -> bool:
    allowed = allowed.rstrip("/")
    return path == allowed or path.startswith(allowed + "/")


def grade(spec: dict, ctx) -> dict:
    mode = spec.get("mode")
    diff = diff_trees(ctx.baseline, ctx.tree)
    changed = diff.changed_paths
    # Used by byte_identical and confined_diff only; pollution_whitelist
    # grades the raw diff (see _EPHEMERAL_DIRS above).
    ephemeral = [p for p in changed if _is_ephemeral(p)]
    considered = [p for p in changed if not _is_ephemeral(p)]

    if mode == "byte_identical":
        ok = not considered
        return {
            "score": 100.0 if ok else 0.0,
            "details": {"mode": mode, "changed_paths": considered, "ignored_ephemeral": ephemeral},
        }

    if mode == "confined_diff":
        allowed = spec.get("allowed_paths") or []
        if not allowed:
            raise ValueError("tree_guard confined_diff mode requires a non-empty spec['allowed_paths']")
        if not considered:
            return {
                "score": 0.0,
                "details": {
                    "mode": mode,
                    "reason": "no changes -- task not attempted",
                    "changed_paths": [],
                    "ignored_ephemeral": ephemeral,
                },
            }
        violations = [p for p in considered if not any(_is_within(p, a) for a in allowed)]
        fraction_ok = 1.0 - (len(violations) / len(considered))
        return {
            "score": max(0.0, min(100.0, 100.0 * fraction_ok)),
            "details": {
                "mode": mode,
                "violations": violations,
                "changed_paths": considered,
                "ignored_ephemeral": ephemeral,
            },
        }

    if mode == "pollution_whitelist":
        patterns = spec.get("pollution_patterns") or []
        if not patterns:
            raise ValueError("tree_guard pollution_whitelist mode requires a non-empty spec['pollution_patterns']")
        if not changed:
            return {
                "score": 0.0,
                "details": {"mode": mode, "reason": "no changes -- task not attempted", "changed_paths": []},
            }
        polluted = [p for p in diff.added if any(fnmatch.fnmatch(p, pat) for pat in patterns)]
        fraction_ok = 1.0 - (len(polluted) / len(diff.added)) if diff.added else 1.0
        return {
            "score": max(0.0, min(100.0, 100.0 * fraction_ok)),
            "details": {"mode": mode, "polluted": polluted, "changed_paths": changed},
        }

    raise ValueError(f"tree_guard: unknown mode {mode!r} (expected byte_identical|confined_diff|pollution_whitelist)")
