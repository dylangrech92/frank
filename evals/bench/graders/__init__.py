"""graders — six grader kinds sharing one signature:

    grade(spec: dict, ctx: GradeContext) -> {"score": float 0..100, "details": ...}

Dispatched by kind through ``grade()`` below, which every caller (run.py's
live protocol, run.py's ``--calibrate``, and ``_selftest.py``) should import
and use instead of calling a kind module directly -- it is the one place
that applies the two spec fields every grader kind honors generically:

    requires   list[int] -- indices into ``ctx.results`` (the grader results
               already produced earlier in this task's ``graders`` list).
               If any required result's ``full_pass`` is not true, THIS
               grader is skipped and scored 0 without running -- e.g. an
               "answer must correctly cite line numbers" grader that only
               makes sense once an earlier "answer identifies the right
               file" grader already passed.
    gate       bool, default false -- if true and this grader's OWN score is
               below spec.get('gate_threshold', 100.0), the entire task's
               total score is forced to 0, not just this grader's weighted
               share. This is how DESIGN.md's B1 contract ("a run that
               mutates the fixture scores 0 regardless of answer quality")
               is expressed: pair kind='tree_guard' with gate=true rather
               than inventing a one-off multiplicative special case.
               run.py's grade_task() is the actual enforcement point (it
               owns weighting across the whole graders list); this module
               only *records* whether a gated grader passed its threshold,
               via full_pass below, so run.py can act on it.

Design rule threaded through every kind module: "no engagement, no credit".
A grader must never let an empty, no-op, or off-task answer score well on a
secondary/defensive bevy of points (coherence, precision, decoy-avoidance,
ranking, ...) just because there was nothing to be wrong about. Concretely:
coherence requires recall_fraction > 0 (answer_facts); confined_diff and
pollution_whitelist require at least one changed path (tree_guard);
precision requires at least one parsed claim (findings_list); decoy_weight
and ranking_weight require recall_fraction > 0 / >=2 ranked recalls
(perf_report). This is what keeps a null/placeholder run's calibration
total near 0 instead of accidentally near 100 by exploiting a band that
rewards absence.

Every grader kind raises ValueError (not a soft 0) on a malformed spec or a
missing/malformed truth fixture -- a task-authoring bug should fail loudly
at calibration time, never silently degrade a score at run time.
"""

from __future__ import annotations

from .context import GradeContext, TreeDiff, diff_trees
from . import (
    acceptance,
    answer_facts,
    envelope_guard,
    findings_list,
    perf_report,
    tree_guard,
)

KNOWN_KINDS = (
    "answer_facts",
    "acceptance",
    "tree_guard",
    "envelope_guard",
    "findings_list",
    "perf_report",
)

_FUNCS = {
    "answer_facts": answer_facts.grade,
    "acceptance": acceptance.grade,
    "tree_guard": tree_guard.grade,
    "envelope_guard": envelope_guard.grade,
    "findings_list": findings_list.grade,
    "perf_report": perf_report.grade,
}

__all__ = ["grade", "KNOWN_KINDS", "GradeContext", "TreeDiff", "diff_trees"]


def grade(spec: dict, ctx: GradeContext) -> dict:
    """Dispatch *spec* to its grader kind, applying the generic ``requires``
    gate first. Always returns a result dict carrying ``_kind`` and
    ``full_pass`` -- callers should not construct result dicts by hand."""
    kind = spec.get("kind")
    func = _FUNCS.get(kind)
    if func is None:
        raise ValueError(f"unknown grader kind {kind!r} (expected one of {KNOWN_KINDS})")

    for idx in spec.get("requires", []):
        if not (0 <= idx < len(ctx.results)):
            raise ValueError(f"grader spec 'requires' index {idx} out of range (have {len(ctx.results)} prior results)")
        if not ctx.results[idx].get("full_pass"):
            result = {
                "score": 0.0,
                "details": {
                    "gated_by_requires": idx,
                    "reason": f"prior grader at index {idx} did not fully pass",
                },
            }
            result["_kind"] = kind
            result["full_pass"] = False
            return result

    result = func(spec, ctx)
    result["_kind"] = kind
    result["full_pass"] = result["score"] >= 100.0 - 1e-9
    return result
