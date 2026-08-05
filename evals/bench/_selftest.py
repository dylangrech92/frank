#!/usr/bin/env python3
"""_selftest — end-to-end check of evals/bench's runner + grading plumbing,
with no LLM and no dependency on sibling build streams' output. All five
other streams write verticals/, fixtures/, and truth/ concurrently with this
one; those directories may be empty, partial, or mid-edit at any moment this
runs. This module fabricates its own tiny task, card, fixture, and truth/
tree entirely inside a temp directory (`tempfile.mkdtemp`) and NEVER writes
under the real verticals/, fixtures/, or truth/ -- the fake task is injected
directly into `tasks.build_tasks()`, bypassing the filesystem scan.

Every check is a plain function that raises AssertionError with the
offending value on failure -- loud, not swallowed -- and prints "OK: ..."
on success. main() runs them all in order and exits 0 only if every one
completes; an uncaught exception propagates and gives a non-zero exit,
which is the point (a passing self-test must mean something).

Usage:
    python3 evals/bench/_selftest.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

import graders  # noqa: E402
import run  # noqa: E402
import tasks  # noqa: E402

# =============================================================================
# Fabricated fixture / truth content
# =============================================================================

FIXTURE_APP_BUGGY = textwrap.dedent(
    """\
    def get_item(items, index):
        return items[index + 1]  # off-by-one bug


    def unused_helper():
        return "dead code, never called"
    """
)

FIXTURE_APP_FIXED = textwrap.dedent(
    """\
    def get_item(items, index):
        return items[index]


    def unused_helper():
        return "dead code, never called"
    """
)

PROBE_CORE_BASIC = textwrap.dedent(
    """\
    import sys
    from pathlib import Path


    def main() -> int:
        tree = Path(sys.argv[1])
        app_path = tree / "app.py"
        if not app_path.is_file():
            return 1
        ns: dict = {}
        exec(app_path.read_text(encoding="utf-8"), ns)
        result = ns["get_item"]([10, 20, 30], 0)
        return 0 if result == 10 else 1


    if __name__ == "__main__":
        sys.exit(main())
    """
)

CARD_JSON = {
    "facts": [
        {"id": "f_offbyone", "tier": "T1", "weight": 70, "any": [r"off[- ]by[- ]one"]},
        {"id": "f_fix", "tier": "T1", "weight": 30, "any": [r"index\s*\+\s*1", r"remov(?:e|ed|ing) the \+\s*1"]},
    ],
    "traps": [
        {"id": "t_redherring", "any": [r"red herring", r"unrelated database issue"], "penalty": 25},
    ],
    "items": [
        {"id": "i_offbyone", "kind": "bug", "tier": "T1", "any": [r"off[- ]by[- ]one"]},
        {"id": "i_deadcode", "kind": "dead", "tier": "T1", "any": [r"unused_helper", r"dead code"]},
        {"id": "i_decoy", "kind": "decoy", "any": [r"memory leak"]},
    ],
}

REFERENCE_ANSWER = (
    "get_item had an off-by-one bug: it indexed items[index + 1] instead of "
    "items[index]. Fixed by removing the + 1.\n"
    "1. off-by-one indexing bug in get_item\n"
    "2. unused_helper is dead code, never called\n"
)

NULL_ANSWER = "I looked at the file and it seems fine, no changes needed."

TASK = {
    "id": "b1_t1_selftest_demo",
    "vertical": "B1",
    "tier": "T1",
    "mode": "code",
    "fixture": "demo",
    "prompt": "Fix the off-by-one bug in get_item and note any dead code you notice.",
    "timeout_s": 60,
    "graders": [
        {"kind": "tree_guard", "weight": 20, "mode": "confined_diff", "allowed_paths": ["app.py"], "gate": True},
        {"kind": "acceptance", "weight": 30, "band_weights": {"core": 100.0}},
        {"kind": "answer_facts", "weight": 20, "recall_weight": 100.0},
        {"kind": "findings_list", "weight": 15, "tier_weights": {"T1": 80.0}, "precision_weight": 20.0},
        {"kind": "envelope_guard", "weight": 15, "expect_verified": True, "requires": [1]},
    ],
}

PERF_CARD = {
    "facts": [
        {"id": "p_hot1", "tier": "T1", "weight": 60, "rank": 1, "any": [r"parse_json"]},
        {"id": "p_hot2", "tier": "T1", "weight": 40, "rank": 2, "any": [r"db_query"]},
    ],
    "traps": [
        {"id": "p_trap", "any": [r"network latency"], "penalty": 10},
    ],
    "items": [],
}

PERF_ANSWER_GOOD = (
    "The top hotspot is parse_json, taking 420ms (95% of calls).\n"
    "The second hotspot is db_query at 30ms, called 270k times.\n"
)

# tier: "multi" (round-2 amendment) -- legal only when the task's own card
# facts/items each carry their own per-item tier (PERF_CARD's facts already
# do: tier "T1" each); the task-level "multi" label just means "not one
# uniform tier". Also exercises a weight-0 pure-gate grader (contributes
# nothing to the weighted total but can still veto it) and, via its truth
# dir having no reference/ subdir, the read-only-mode calibration fallback
# (the pristine fixture copy stands in as the reference tree).
MULTI_TASK = {
    "id": "b4_multi_selftest_perf",
    "vertical": "B4",
    "tier": "multi",
    "mode": "performance_debug",
    "fixture": "demo",
    "prompt": "Profile the demo app and report the top hotspots, in order, with measurements.",
    "timeout_s": 300,
    "graders": [
        {"kind": "tree_guard", "weight": 0, "mode": "byte_identical", "gate": True},
        {"kind": "perf_report", "weight": 100, "recall_weight": 50.0, "evidence_weight": 30.0, "ranking_weight": 10.0, "decoy_weight": 10.0},
    ],
}


def _checkpoint(label: str) -> None:
    print(f"OK: {label}")


def _build_temp_world() -> Path:
    """Lay out <tmp>/fixtures/demo/ and <tmp>/truth/<task_id>/{card.json,
    acceptance/,reference/,reference_answer.md,null_answer.txt} -- a
    complete, isolated stand-in for the real evals/bench/{fixtures,truth}
    trees this module must never touch."""
    tmp = Path(tempfile.mkdtemp(prefix="bench-selftest-"))

    fixture_dir = tmp / "fixtures" / "demo"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "app.py").write_text(FIXTURE_APP_BUGGY, encoding="utf-8")

    truth_dir = tmp / "truth" / TASK["id"]
    truth_dir.mkdir(parents=True)
    (truth_dir / "card.json").write_text(json.dumps(CARD_JSON), encoding="utf-8")

    acceptance_dir = truth_dir / "acceptance"
    acceptance_dir.mkdir()
    (acceptance_dir / "probe_core_basic.py").write_text(PROBE_CORE_BASIC, encoding="utf-8")

    reference_dir = truth_dir / "reference"
    reference_dir.mkdir()
    (reference_dir / "app.py").write_text(FIXTURE_APP_FIXED, encoding="utf-8")

    (truth_dir / "reference_answer.md").write_text(REFERENCE_ANSWER, encoding="utf-8")
    (truth_dir / "null_answer.txt").write_text(NULL_ANSWER, encoding="utf-8")

    perf_truth_dir = tmp / "truth" / "perf_demo"
    perf_truth_dir.mkdir(parents=True)
    (perf_truth_dir / "card.json").write_text(json.dumps(PERF_CARD), encoding="utf-8")

    # MULTI_TASK's truth dir deliberately has NO reference/ subdir -- it is
    # a read-only-mode (performance_debug) task, so --calibrate must fall
    # back to the pristine fixture copy as the reference tree rather than
    # FAIL on the missing directory.
    multi_truth_dir = tmp / "truth" / MULTI_TASK["id"]
    multi_truth_dir.mkdir(parents=True)
    (multi_truth_dir / "card.json").write_text(json.dumps(PERF_CARD), encoding="utf-8")
    (multi_truth_dir / "reference_answer.md").write_text(PERF_ANSWER_GOOD, encoding="utf-8")

    return tmp


# =============================================================================
# tasks.py
# =============================================================================


def check_build_tasks() -> None:
    result = tasks.build_tasks([("selftest_module.py", [TASK])])
    assert result == [TASK], f"build_tasks should return the injected task unchanged, got {result}"

    # A broken entry must collect ALL problems in one pass, not just the
    # first, and must not silently drop a well-formed task alongside it.
    broken = dict(TASK)
    broken["id"] = "not-a-valid-id"
    broken["graders"] = [{"kind": "bogus_kind", "weight": 50}]
    try:
        tasks.build_tasks([("good.py", [TASK]), ("bad.py", [broken])])
        raise AssertionError("build_tasks should have raised TaskValidationError on the broken entry")
    except tasks.TaskValidationError as exc:
        assert len(exc.problems) >= 2, f"expected >=2 collected problems, got {exc.problems}"

    _checkpoint("tasks.build_tasks: accepts a valid injected task, collects every problem on a broken one")


def check_build_tasks_multi_tier() -> None:
    # tier: "multi" + a "..._multi_..." id must validate cleanly.
    result = tasks.build_tasks([("selftest_multi.py", [MULTI_TASK])])
    assert result == [MULTI_TASK], f"build_tasks should accept a multi-tier task unchanged, got {result}"

    _checkpoint("tasks.build_tasks: tier 'multi' + matching id segment validates")


def check_build_tasks_weight_zero_gate() -> None:
    # weight: 0 paired with gate: true is a valid pure veto -- must NOT be
    # flagged, and must not perturb the sum-to-100 check (it contributes 0).
    result = tasks.build_tasks([("selftest_zero_gate.py", [MULTI_TASK])])
    assert result == [MULTI_TASK], result

    # weight: 0 WITHOUT gate: true is dead spec -- FATAL.
    bad = dict(TASK)
    bad["id"] = "b1_t1_selftest_bad_zero_weight"
    bad["graders"] = [
        {"kind": "tree_guard", "weight": 0, "mode": "byte_identical"},
        {"kind": "answer_facts", "weight": 100, "recall_weight": 100.0},
    ]
    try:
        tasks.build_tasks([("bad_zero.py", [bad])])
        raise AssertionError("build_tasks should reject a weight-0 grader without gate: true")
    except tasks.TaskValidationError as exc:
        assert any("weight" in p.lower() and "gate" in p.lower() for p in exc.problems), exc.problems

    _checkpoint("tasks.build_tasks: weight-0 valid only with gate: true, FATAL otherwise")


def check_build_tasks_per_kind_fields() -> None:
    # tree_guard with an unknown/missing mode is FATAL at --list time.
    bad_mode = dict(TASK)
    bad_mode["id"] = "b1_t1_selftest_bad_tree_guard_mode"
    bad_mode["graders"] = [
        {"kind": "tree_guard", "weight": 20, "gate": True},  # no 'mode'
        {"kind": "answer_facts", "weight": 80, "recall_weight": 100.0},
    ]
    try:
        tasks.build_tasks([("bad_mode.py", [bad_mode])])
        raise AssertionError("build_tasks should reject a tree_guard grader with no mode")
    except tasks.TaskValidationError as exc:
        assert any("tree_guard" in p and "mode" in p for p in exc.problems), exc.problems

    # tree_guard mode=confined_diff with no allowed_paths is FATAL.
    bad_confined = dict(TASK)
    bad_confined["id"] = "b1_t1_selftest_bad_confined_diff"
    bad_confined["graders"] = [
        {"kind": "tree_guard", "weight": 20, "mode": "confined_diff", "gate": True},  # no allowed_paths
        {"kind": "answer_facts", "weight": 80, "recall_weight": 100.0},
    ]
    try:
        tasks.build_tasks([("bad_confined.py", [bad_confined])])
        raise AssertionError("build_tasks should reject confined_diff with no allowed_paths")
    except tasks.TaskValidationError as exc:
        assert any("allowed_paths" in p for p in exc.problems), exc.problems

    # envelope_guard with a garbage expect_verified is FATAL.
    bad_envelope = dict(TASK)
    bad_envelope["id"] = "b1_t1_selftest_bad_expect_verified"
    bad_envelope["graders"] = [
        {"kind": "envelope_guard", "weight": 100, "expect_verified": "yes"},
    ]
    try:
        tasks.build_tasks([("bad_envelope.py", [bad_envelope])])
        raise AssertionError("build_tasks should reject a non-bool/None expect_verified")
    except tasks.TaskValidationError as exc:
        assert any("expect_verified" in p for p in exc.problems), exc.problems

    _checkpoint("tasks.build_tasks: per-kind field checks catch tree_guard/envelope_guard authoring bugs at --list time")


# =============================================================================
# graders/
# =============================================================================


def check_tree_guard(baseline: Path, good_tree: Path, polluted_tree: Path) -> None:
    ctx = graders.GradeContext(task=TASK, tree=good_tree, baseline=baseline, envelope=None, answer="", truth_dir=None)
    result = graders.grade({"kind": "tree_guard", "mode": "confined_diff", "allowed_paths": ["app.py"]}, ctx)
    assert result["score"] == 100.0, result
    assert result["_kind"] == "tree_guard" and result["full_pass"] is True, result

    ctx2 = graders.GradeContext(task=TASK, tree=polluted_tree, baseline=baseline, envelope=None, answer="", truth_dir=None)
    result2 = graders.grade({"kind": "tree_guard", "mode": "confined_diff", "allowed_paths": ["app.py"]}, ctx2)
    assert 0.0 < result2["score"] < 100.0, result2

    ctx3 = graders.GradeContext(task=TASK, tree=good_tree, baseline=baseline, envelope=None, answer="", truth_dir=None)
    result3 = graders.grade({"kind": "tree_guard", "mode": "byte_identical"}, ctx3)
    assert result3["score"] == 0.0, result3

    ctx4 = graders.GradeContext(task=TASK, tree=baseline, baseline=baseline, envelope=None, answer="", truth_dir=None)
    result4 = graders.grade({"kind": "tree_guard", "mode": "byte_identical"}, ctx4)
    assert result4["score"] == 100.0, result4

    # No engagement, no credit: an empty diff must score 0 under
    # confined_diff, not a vacuous 100.
    ctx5 = graders.GradeContext(task=TASK, tree=baseline, baseline=baseline, envelope=None, answer="", truth_dir=None)
    result5 = graders.grade({"kind": "tree_guard", "mode": "confined_diff", "allowed_paths": ["app.py"]}, ctx5)
    assert result5["score"] == 0.0, result5

    # Interpreter/tooling residue (__pycache__, .ruff_cache, *.pyc) is not an
    # agent edit: the harness's post-edit lint and the task's own repro
    # command produce it. Pilot regression — every B1 rep lost ~19/20
    # tree_guard points to exactly this residue while its real diff was
    # perfectly confined.
    world = baseline.parent.parent
    residue_tree = world / "agent_residue"
    shutil.copytree(good_tree, residue_tree)
    (residue_tree / "__pycache__").mkdir()
    (residue_tree / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\x00fake-bytecode")
    (residue_tree / ".ruff_cache").mkdir()
    (residue_tree / ".ruff_cache" / "CACHEDIR.TAG").write_text("tag", encoding="utf-8")

    result6 = graders.grade({"kind": "tree_guard", "mode": "confined_diff", "allowed_paths": ["app.py"]}, graders.GradeContext(task=TASK, tree=residue_tree, baseline=baseline, envelope=None, answer="", truth_dir=None))
    assert result6["score"] == 100.0, result6
    assert result6["details"]["violations"] == [], result6
    assert sorted(result6["details"]["ignored_ephemeral"]) == [".ruff_cache/CACHEDIR.TAG", "__pycache__/app.cpython-312.pyc"], result6

    # Residue alone is still "not attempted" under confined_diff, and still
    # byte-identical under byte_identical.
    residue_only = world / "agent_residue_only"
    shutil.copytree(baseline, residue_only)
    (residue_only / "__pycache__").mkdir()
    (residue_only / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\x00fake-bytecode")
    result7 = graders.grade({"kind": "tree_guard", "mode": "confined_diff", "allowed_paths": ["app.py"]}, graders.GradeContext(task=TASK, tree=residue_only, baseline=baseline, envelope=None, answer="", truth_dir=None))
    assert result7["score"] == 0.0, result7
    result8 = graders.grade({"kind": "tree_guard", "mode": "byte_identical"}, graders.GradeContext(task=TASK, tree=residue_only, baseline=baseline, envelope=None, answer="", truth_dir=None))
    assert result8["score"] == 100.0, result8

    # The filter is mode-scoped: pollution_whitelist's patterns are the
    # task's explicit leave-no-trace contract and MUST still see residue.
    result9 = graders.grade({"kind": "tree_guard", "mode": "pollution_whitelist", "pollution_patterns": ["*__pycache__*", "*.pyc"]}, graders.GradeContext(task=TASK, tree=residue_tree, baseline=baseline, envelope=None, answer="", truth_dir=None))
    assert result9["score"] < 100.0, result9
    assert result9["details"]["polluted"] == ["__pycache__/app.cpython-312.pyc"], result9

    _checkpoint("graders.tree_guard: byte_identical + confined_diff + no-engagement-no-credit + ephemeral-residue filter (mode-scoped)")


def check_answer_facts(truth_dir: Path) -> None:
    ctx = graders.GradeContext(task=TASK, tree=truth_dir, baseline=truth_dir, envelope=None, answer=REFERENCE_ANSWER, truth_dir=truth_dir)
    result = graders.grade({"kind": "answer_facts", "recall_weight": 100.0}, ctx)
    assert result["score"] == 100.0, result

    ctx_null = graders.GradeContext(task=TASK, tree=truth_dir, baseline=truth_dir, envelope=None, answer=NULL_ANSWER, truth_dir=truth_dir)
    result_null = graders.grade({"kind": "answer_facts", "recall_weight": 100.0}, ctx_null)
    assert result_null["score"] == 0.0, result_null

    trap_answer = REFERENCE_ANSWER + " This might be a red herring though."
    ctx_trap = graders.GradeContext(task=TASK, tree=truth_dir, baseline=truth_dir, envelope=None, answer=trap_answer, truth_dir=truth_dir)
    result_trap = graders.grade({"kind": "answer_facts", "recall_weight": 100.0}, ctx_trap)
    assert result_trap["score"] < 100.0, result_trap

    _checkpoint("graders.answer_facts: full recall on the reference answer, zero on null, trap penalty applied")


def check_acceptance(baseline: Path, good_tree: Path, truth_dir: Path) -> None:
    ctx_good = graders.GradeContext(task=TASK, tree=good_tree, baseline=baseline, envelope=None, answer="", truth_dir=truth_dir)
    result_good = graders.grade({"kind": "acceptance", "band_weights": {"core": 100.0}}, ctx_good)
    assert result_good["score"] == 100.0, result_good

    ctx_bad = graders.GradeContext(task=TASK, tree=baseline, baseline=baseline, envelope=None, answer="", truth_dir=truth_dir)
    result_bad = graders.grade({"kind": "acceptance", "band_weights": {"core": 100.0}}, ctx_bad)
    assert result_bad["score"] == 0.0, result_bad

    _checkpoint("graders.acceptance: probe passes against the fixed tree, fails against the buggy fixture")


def check_envelope_guard(baseline: Path) -> None:
    envelope_ok = run.synthetic_envelope("some research answer", mutated=False, changed_paths=[], verified=None)
    ctx_ok = graders.GradeContext(task=TASK, tree=baseline, baseline=baseline, envelope=envelope_ok, answer="x", truth_dir=None)
    result_ok = graders.grade({"kind": "envelope_guard", "expect_verified": None}, ctx_ok)
    assert result_ok["score"] == 100.0, result_ok

    envelope_bad = run.synthetic_envelope("x", mutated=True, changed_paths=["app.py"], verified=None)
    ctx_bad = graders.GradeContext(task=TASK, tree=baseline, baseline=baseline, envelope=envelope_bad, answer="x", truth_dir=None)
    result_bad = graders.grade({"kind": "envelope_guard", "expect_verified": None}, ctx_bad)
    assert result_bad["score"] == 0.0, result_bad

    envelope_true = run.synthetic_envelope("x", mutated=True, changed_paths=["app.py"], verified=True)
    ctx_true = graders.GradeContext(task=TASK, tree=baseline, baseline=baseline, envelope=envelope_true, answer="x", truth_dir=None)
    ctx_true.results.append({"_kind": "acceptance", "full_pass": True, "score": 100.0, "details": {}})
    result_true = graders.grade({"kind": "envelope_guard", "expect_verified": True}, ctx_true)
    assert result_true["score"] == 100.0, result_true

    # The envelope lie: verified=True claimed, but the acceptance suite it
    # was supposedly gated on actually failed -- worse than an honest miss.
    ctx_lie = graders.GradeContext(task=TASK, tree=baseline, baseline=baseline, envelope=envelope_true, answer="x", truth_dir=None)
    ctx_lie.results.append({"_kind": "acceptance", "full_pass": False, "score": 40.0, "details": {}})
    result_lie = graders.grade({"kind": "envelope_guard", "expect_verified": True}, ctx_lie)
    assert result_lie["score"] == 0.0, result_lie
    assert result_lie["details"].get("envelope_lie") is True, result_lie

    _checkpoint("graders.envelope_guard: None/True contracts, an honest pass, and the envelope-lie case")


def check_findings_list(truth_dir: Path) -> None:
    findings_answer = (
        "1. off-by-one indexing bug in get_item\n"
        "2. unused_helper is dead code, never called\n"
    )
    ctx = graders.GradeContext(task=TASK, tree=truth_dir, baseline=truth_dir, envelope=None, answer=findings_answer, truth_dir=truth_dir)
    result = graders.grade({"kind": "findings_list", "tier_weights": {"T1": 80.0}, "precision_weight": 20.0}, ctx)
    assert result["score"] == 100.0, result

    ctx_empty = graders.GradeContext(task=TASK, tree=truth_dir, baseline=truth_dir, envelope=None, answer="nothing to report", truth_dir=truth_dir)
    result_empty = graders.grade({"kind": "findings_list", "tier_weights": {"T1": 80.0}, "precision_weight": 20.0}, ctx_empty)
    assert result_empty["score"] == 0.0, result_empty

    _checkpoint("graders.findings_list: tier recall + precision on target, zero on no-engagement")


def check_perf_report(perf_truth_dir: Path) -> None:
    perf_task = {**TASK, "id": "b4_t1_selftest_perf", "vertical": "B4", "tier": "T1", "mode": "performance_debug"}
    spec = {"kind": "perf_report", "recall_weight": 50.0, "evidence_weight": 30.0, "ranking_weight": 10.0, "decoy_weight": 10.0}

    ctx = graders.GradeContext(task=perf_task, tree=perf_truth_dir, baseline=perf_truth_dir, envelope=None, answer=PERF_ANSWER_GOOD, truth_dir=perf_truth_dir)
    result = graders.grade(spec, ctx)
    assert result["score"] == 100.0, result

    no_evidence_answer = "parse_json is slow. db_query is also slow."
    ctx2 = graders.GradeContext(task=perf_task, tree=perf_truth_dir, baseline=perf_truth_dir, envelope=None, answer=no_evidence_answer, truth_dir=perf_truth_dir)
    result2 = graders.grade(spec, ctx2)
    assert result2["details"]["evidence_fraction"] == 0.0, result2
    assert result2["score"] < 100.0, result2

    decoy_answer = PERF_ANSWER_GOOD + "There might also be network latency issues.\n"
    ctx3 = graders.GradeContext(task=perf_task, tree=perf_truth_dir, baseline=perf_truth_dir, envelope=None, answer=decoy_answer, truth_dir=perf_truth_dir)
    result3 = graders.grade(spec, ctx3)
    assert result3["details"]["decoy_fraction"] == 0.0, result3
    assert result3["score"] < 100.0, result3

    _checkpoint("graders.perf_report: recall + evidence-adjacency + ranking + decoy penalty")

    # Regression: _EVIDENCE_RE's trailing \b broke on '%' when it's followed
    # by whitespace/punctuation/end-of-line ('%' is not a word char, so no
    # word/non-word transition ever forms there) -- percent-backed evidence
    # was silently uncreditable despite the grader's own docstring listing
    # '%' as an accepted unit. A percent claim right next to the fact
    # mention must now be credited...
    percent_evidence_answer = (
        "The top hotspot is parse_json, at 2.7% of total runtime.\n"
        "The second hotspot is db_query, at (14.5%) of total runtime.\n"
    )
    ctx4 = graders.GradeContext(task=perf_task, tree=perf_truth_dir, baseline=perf_truth_dir, envelope=None, answer=percent_evidence_answer, truth_dir=perf_truth_dir)
    result4 = graders.grade(spec, ctx4)
    assert result4["details"]["evidence_fraction"] == 1.0, result4
    assert result4["score"] == 100.0, result4

    # ...while a bare number with no recognized unit next to the mention
    # still earns nothing -- the fix must not turn every digit into
    # "evidence".
    bare_number_answer = (
        "The top hotspot is parse_json, roughly 2.7 of total (no unit given).\n"
        "The second hotspot is db_query, roughly 14 too.\n"
    )
    ctx5 = graders.GradeContext(task=perf_task, tree=perf_truth_dir, baseline=perf_truth_dir, envelope=None, answer=bare_number_answer, truth_dir=perf_truth_dir)
    result5 = graders.grade(spec, ctx5)
    assert result5["details"]["evidence_fraction"] == 0.0, result5
    assert result5["score"] < 100.0, result5

    _checkpoint("graders.perf_report: percent-backed evidence is credited (round-2 regression fix), bare numbers are not")


# =============================================================================
# run.py: aggregation, calibration, envelope parsing, DNF, spawn detection
# =============================================================================


def check_grade_task_aggregation(baseline: Path, good_tree: Path, polluted_tree: Path, truth_dir: Path) -> None:
    envelope = run.synthetic_envelope(REFERENCE_ANSWER, mutated=True, changed_paths=["app.py"], verified=True)

    ctx = graders.GradeContext(task=TASK, tree=good_tree, baseline=baseline, envelope=envelope, answer=REFERENCE_ANSWER, truth_dir=truth_dir)
    result = run.grade_task(TASK, ctx)
    assert result["total"] >= 99.99, result
    assert result["gate_failed_index"] is None, result

    # Gate veto: tree_guard (index 0, gate=True) fails against the polluted
    # tree, which must zero the WHOLE task total, not just its own 20-point
    # share -- even though every other grader here would otherwise pass.
    ctx2 = graders.GradeContext(task=TASK, tree=polluted_tree, baseline=baseline, envelope=envelope, answer=REFERENCE_ANSWER, truth_dir=truth_dir)
    result2 = run.grade_task(TASK, ctx2)
    assert result2["gate_failed_index"] == 0, result2
    assert result2["total"] == 0.0, result2

    # requires gating: envelope_guard (index 4) requires acceptance (index 1)
    # to have fully passed. Against the untouched buggy baseline, acceptance
    # fails, so envelope_guard must be gated to 0 via 'requires' rather than
    # scored on its own verified-flag logic.
    ctx3 = graders.GradeContext(task=TASK, tree=baseline, baseline=baseline, envelope=envelope, answer=REFERENCE_ANSWER, truth_dir=truth_dir)
    result3 = run.grade_task(TASK, ctx3)
    envelope_guard_result = result3["results"][4]
    assert envelope_guard_result["details"].get("gated_by_requires") == 1, envelope_guard_result
    assert envelope_guard_result["score"] == 0.0, envelope_guard_result

    _checkpoint("run.grade_task: weighted aggregation + gate veto + requires gating")


def check_weight_zero_gate(baseline: Path, multi_truth_dir: Path) -> None:
    # Gate passes: tree is byte-identical to baseline (untouched, as a
    # correct read-only-mode run should be) and the answer scores 100 on
    # perf_report -- the weight-0 gate contributes nothing to the sum but
    # must not veto either.
    ctx_pass = graders.GradeContext(
        task=MULTI_TASK, tree=baseline, baseline=baseline, envelope=None,
        answer=PERF_ANSWER_GOOD, truth_dir=multi_truth_dir,
    )
    result_pass = run.grade_task(MULTI_TASK, ctx_pass)
    assert result_pass["gate_failed_index"] is None, result_pass
    assert result_pass["total"] >= 99.99, result_pass

    # Gate vetoes: the tree got mutated (violates byte_identical), so even
    # though the answer is still perfect, the weight-0 gate must force the
    # WHOLE total to 0 -- proving a weight-0 entry still gates despite
    # contributing nothing to the weighted sum.
    mutated_tree = baseline.parent / "selftest_gate_mutated"
    shutil.copytree(baseline, mutated_tree)
    (mutated_tree / "app.py").write_text(FIXTURE_APP_FIXED, encoding="utf-8")
    ctx_veto = graders.GradeContext(
        task=MULTI_TASK, tree=mutated_tree, baseline=baseline, envelope=None,
        answer=PERF_ANSWER_GOOD, truth_dir=multi_truth_dir,
    )
    result_veto = run.grade_task(MULTI_TASK, ctx_veto)
    assert result_veto["gate_failed_index"] == 0, result_veto
    assert result_veto["total"] == 0.0, result_veto

    _checkpoint("run.grade_task: weight-0 gate entry vetoes the whole total without contributing to the score sum")


def check_calibration(world: Path) -> None:
    original_fixtures_dir = run.FIXTURES_DIR
    original_truth_dir = run.TRUTH_DIR
    run.FIXTURES_DIR = world / "fixtures"
    run.TRUTH_DIR = world / "truth"
    try:
        result = run._calibrate_one(TASK)
        assert result["status"] == "PASS", result
        assert result["reference_total"] >= 99.99, result
        assert result["null_total"] <= 5.0, result

        missing_task = {**TASK, "id": "b1_t1_no_truth_dir_at_all"}
        skip_result = run._calibrate_one(missing_task)
        assert skip_result["status"] == "SKIP", skip_result
    finally:
        run.FIXTURES_DIR = original_fixtures_dir
        run.TRUTH_DIR = original_truth_dir

    _checkpoint("run._calibrate_one: reference scores ~100, null scores <=5, missing truth dir is a loud SKIP")


def check_calibration_no_reference(world: Path) -> None:
    # MULTI_TASK's truth dir has no reference/ subdir (a read-only-mode
    # contract: performance_debug never mutates the tree). --calibrate must
    # fall back to the pristine fixture copy as the reference tree instead
    # of FAILing on the missing directory, and the synthesized reference
    # envelope must land on verified=None (nothing mutated) -- which is
    # exactly what the weight-0 tree_guard gate and envelope_guard-less
    # perf_report task need to reach a 100 reference score.
    original_fixtures_dir = run.FIXTURES_DIR
    original_truth_dir = run.TRUTH_DIR
    run.FIXTURES_DIR = world / "fixtures"
    run.TRUTH_DIR = world / "truth"
    try:
        result = run._calibrate_one(MULTI_TASK)
        assert result["status"] == "PASS", result
        assert result["reference_total"] >= 99.99, result
        assert result["null_total"] <= 5.0, result
    finally:
        run.FIXTURES_DIR = original_fixtures_dir
        run.TRUTH_DIR = original_truth_dir

    _checkpoint("run._calibrate_one: no reference/ dir falls back to the pristine fixture as the reference tree")


def check_parse_envelope() -> None:
    good_stdout = 'some log line\n{"envelope": 1, "status": "ok", "answer": "hi"}\n'
    parsed = run.parse_envelope(good_stdout)
    assert parsed == {"envelope": 1, "status": "ok", "answer": "hi"}, parsed

    trailing_noise = good_stdout + "trailing non-json noise\n"
    parsed2 = run.parse_envelope(trailing_noise)
    assert parsed2 == {"envelope": 1, "status": "ok", "answer": "hi"}, parsed2

    assert run.parse_envelope('{"status": "ok"}\n') is None
    assert run.parse_envelope("{not valid json at all\n") is None
    assert run.parse_envelope("") is None

    _checkpoint("run.parse_envelope: last JSON-object line, tolerates trailing noise and malformed input")


def check_dnf_classification() -> None:
    dnf, reason = run._classify_dnf(None, timed_out=True, timeout_reason="hard timeout after 60s")
    assert dnf is True and reason == "hard timeout after 60s", (dnf, reason)

    dnf, reason = run._classify_dnf(None, timed_out=False, timeout_reason=None)
    assert dnf is True and "no parseable" in reason, (dnf, reason)

    dnf, reason = run._classify_dnf({"status": "error", "error": "boom"}, timed_out=False, timeout_reason=None)
    assert dnf is True and "boom" in reason, (dnf, reason)

    dnf, reason = run._classify_dnf({"status": "ok"}, timed_out=False, timeout_reason=None)
    assert dnf is False and reason is None, (dnf, reason)

    _checkpoint("run._classify_dnf: hard timeout, unparseable envelope, status=error, and the healthy case")


def check_spawn_agents_detection() -> None:
    # Addendum coverage: a canned stderr transcript with, and without, a
    # spawn_agents tool-call line -- the exact shape agent.py's telemetry
    # print produces (ui.tool_call() is plain text without --pretty).
    stderr_with = (
        "config: /tmp/x/config.json\n"
        "mode: code\n"
        'Tool call: spawn_agents({"count": 2, "task": "fan out"})\n'
        "Tool result: spawn_agents -> ok\n"
    )
    spawned, line = run.detect_spawn_agents(stderr_with)
    assert spawned is True, (spawned, line)
    assert line is not None and "spawn_agents(" in line, (spawned, line)

    stderr_without = (
        "config: /tmp/x/config.json\n"
        "mode: code\n"
        'Tool call: read_file({"path": "app.py"})\n'
        'Tool call: write_file({"path": "app.py", "content": "..."})\n'
    )
    spawned2, line2 = run.detect_spawn_agents(stderr_without)
    assert spawned2 is False and line2 is None, (spawned2, line2)

    _checkpoint("run.detect_spawn_agents: matches a real spawn_agents tool-call line, absent otherwise")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    world = _build_temp_world()
    try:
        baseline = world / "fixtures" / "demo"
        truth_dir = world / "truth" / TASK["id"]
        perf_truth_dir = world / "truth" / "perf_demo"
        multi_truth_dir = world / "truth" / MULTI_TASK["id"]

        good_tree = world / "agent_good"
        shutil.copytree(baseline, good_tree)
        (good_tree / "app.py").write_text(FIXTURE_APP_FIXED, encoding="utf-8")

        polluted_tree = world / "agent_polluted"
        shutil.copytree(baseline, polluted_tree)
        (polluted_tree / "app.py").write_text(FIXTURE_APP_FIXED, encoding="utf-8")
        (polluted_tree / "junk.log").write_text("noise", encoding="utf-8")

        check_build_tasks()
        check_build_tasks_multi_tier()
        check_build_tasks_weight_zero_gate()
        check_build_tasks_per_kind_fields()
        check_tree_guard(baseline, good_tree, polluted_tree)
        check_answer_facts(truth_dir)
        check_acceptance(baseline, good_tree, truth_dir)
        check_envelope_guard(baseline)
        check_findings_list(truth_dir)
        check_perf_report(perf_truth_dir)
        check_grade_task_aggregation(baseline, good_tree, polluted_tree, truth_dir)
        check_weight_zero_gate(baseline, multi_truth_dir)
        check_calibration(world)
        check_calibration_no_reference(world)
        check_parse_envelope()
        check_dnf_classification()
        check_spawn_agents_detection()
    finally:
        shutil.rmtree(world, ignore_errors=True)

    print("\nALL SELFTEST CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
