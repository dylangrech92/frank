#!/usr/bin/env python3
"""Bench runner — copy fixture, launch the real one-shot pipeline, capture,
grade, report. See ``evals/bench/DESIGN.md`` ("Runner protocol",
"Integration contracts (build-phase)") for the binding spec this implements.

Usage:
    python3 evals/bench/run.py --list                    # list selected tasks
    python3 evals/bench/run.py --calibrate                # grade reference/null, no LLM
    python3 evals/bench/run.py                             # full live run, N=3 reps
    python3 evals/bench/run.py --vertical B1 --reps 1      # one vertical, one rep
    python3 evals/bench/run.py --task b1_t2_cache_corruption --timeout 600

Deliberately standalone: does not import ``evals/run.py`` (import-time side
effects there — scenario validation, a module-level SCENARIOS load — would
run just to reuse a couple of helper functions). ``deep_merge``/config
plumbing is replicated here instead, per the build contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BENCH_DIR / "fixtures"
TRUTH_DIR = BENCH_DIR / "truth"

sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT))
import tasks as tasks_mod  # noqa: E402
import graders  # noqa: E402

# Benchmark runs record real usage into the stats ledger by design (owner
# decision 2026-08-04): a benchmark IS usage, and its token/tool telemetry
# must be visible on the dashboard. (evals/run.py's commit-gate suite still
# suppresses stats — that suite is synthetic traffic, this one is not.)

DEFAULT_REPS = 3
# Must exceed the harness's own stream-read timeout (llm.py
# _READ_TIMEOUT_SECONDS = 600): a stalled backend stream is only detected by
# the harness after 600s of socket silence, and its recovery announces itself
# on stderr. An idle kill at exactly 600s races that recovery and always wins
# — every pilot DNF was this race, not a hung harness.
IDLE_TIMEOUT_S = 900
_VALID_VERTICALS = ("B1", "B2", "B3", "B4", "B5")


# =============================================================================
# Config merging (standalone reimplementation of evals/run.py's helpers)
# =============================================================================


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* onto a copy of *base*; nested dicts merge,
    everything else (scalars, lists) is replaced wholesale by *override*'s value."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(task: dict) -> dict:
    """Return the merged config dict for *task* (repo config.json + optional
    per-task ``config`` override — Runner protocol reserves this field; the
    Task entry contract's example doesn't need it, so it's optional)."""
    base_cfg = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    overrides = task.get("config", "default")
    if overrides in (None, "default"):
        return base_cfg
    return deep_merge(base_cfg, overrides)


# =============================================================================
# Envelope + telemetry parsing
# =============================================================================


def parse_envelope(stdout_text: str) -> dict | None:
    """The envelope is the single JSON object on stdout; per the build
    contract, take the LAST stdout line that parses as a JSON object
    containing the key ``"envelope"`` (tolerates any stray prose/log lines
    a future harness change might interleave before it)."""
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "envelope" in obj:
            return obj
    return None


# spawn_agents fires as a real tool call, which agent.py always echoes to
# stderr as "Tool call: spawn_agents(...)" regardless of --pretty (see
# ui.tool_call — colorized only under --pretty, which this runner never
# passes, so the substring survives verbatim). Detection is deliberately a
# dumb, transparent regex over the saved stderr text, per the addendum: token
# usage for spawned children never reaches the parent's envelope, so a
# summary.md token median computed over a spawned rep must be flagged rather
# than presented as total consumption.
_SPAWN_RE = re.compile(r"Tool call:\s*spawn_agents\(")


def detect_spawn_agents(stderr_text: str) -> tuple[bool, str | None]:
    """Return (spawned, matched_line) — matched_line is the first stderr
    line naming a spawn_agents call, or None if it never fired."""
    for line in stderr_text.splitlines():
        if _SPAWN_RE.search(line):
            return True, line
    return False, None


# =============================================================================
# Subprocess watchdog
# =============================================================================


def run_subprocess_with_watchdog(
    cmd: list[str], cwd: str, env: dict, hard_timeout_s: int, idle_timeout_s: int
) -> dict:
    """Launch *cmd*, capture stdout/stderr fully via reader threads, and kill
    the whole process group on either a hard timeout or *idle_timeout_s*
    seconds with no new stderr bytes (stdout does not count — a long-running
    tool call can go quiet on stdout while telemetry keeps flowing on
    stderr; the contract's zero-activity signal is stderr specifically).

    Returns {"stdout", "stderr", "returncode", "timed_out", "timeout_reason", "wall_s"}.
    """
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    state = {"last_activity": time.monotonic()}
    state_lock = threading.Lock()

    def _reader(stream, chunks: list[str], track_activity: bool) -> None:
        try:
            for line in iter(stream.readline, ""):
                chunks.append(line)
                if track_activity:
                    with state_lock:
                        state["last_activity"] = time.monotonic()
        finally:
            stream.close()

    stdout_thread = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks, False), daemon=True)
    stderr_thread = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks, True), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    timeout_reason = None
    while True:
        try:
            proc.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            now = time.monotonic()
            if now - start > hard_timeout_s:
                timed_out = True
                timeout_reason = f"hard timeout after {hard_timeout_s}s"
                break
            with state_lock:
                idle_for = now - state["last_activity"]
            if idle_for > idle_timeout_s:
                timed_out = True
                timeout_reason = f"zero-activity kill after {idle_timeout_s}s with no stderr telemetry"
                break

    if timed_out:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    return {
        "stdout": "".join(stdout_chunks),
        "stderr": "".join(stderr_chunks),
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "timeout_reason": timeout_reason,
        "wall_s": time.monotonic() - start,
    }


# =============================================================================
# Fixture + git-baseline setup
# =============================================================================


def _materialize_fixture(fixture_name: str | None, dest_dir: Path) -> None:
    """Populate *dest_dir* with the named fixture (fixture=None -> empty dir,
    per the Task entry contract)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if fixture_name is None:
        return
    src = FIXTURES_DIR / fixture_name
    if not src.is_dir():
        raise FileNotFoundError(f"fixture {fixture_name!r} not found under {FIXTURES_DIR}")
    shutil.copytree(src, dest_dir, dirs_exist_ok=True)


def _git_init_baseline(project_dir: Path) -> None:
    """git-init the live temp copy with a local (repo-scoped, not global)
    user identity and a baseline commit, then exclude .coding_agent/ from
    future diffs. Runner-side setup only — graders never shell out to git
    (see graders/context.py); this exists solely so a human inspecting a
    surviving temp dir after a crash can `git diff` the agent's edits."""

    def _run(args: list[str]) -> None:
        subprocess.run(
            ["git", *args],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

    _run(["init", "-q"])
    _run(["config", "user.name", "bench-runner"])
    _run(["config", "user.email", "bench-runner@localhost"])
    _run(["add", "-A"])
    _run(["commit", "-q", "--allow-empty", "-m", "baseline"])

    exclude_path = project_dir / ".git" / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    with exclude_path.open("a", encoding="utf-8") as f:
        f.write(".coding_agent/\n")


def _git_rev_parse_head() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return proc.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


# =============================================================================
# Grading
# =============================================================================


def grade_task(task: dict, ctx: "graders.GradeContext") -> dict:
    """Run every grader in task['graders'] against *ctx*, in declared order,
    accumulating each result into ``ctx.results`` (so a later grader can read
    an earlier one — see spec['requires'] in graders/__init__.py) and summing
    weighted scores. Any grader with spec['gate']=True scoring below
    spec.get('gate_threshold', 100.0) forces the WHOLE task total to 0 --
    this is the enforcement point for the veto graders/__init__.py documents
    but only records (via each result's 'full_pass' flag)."""
    total = 0.0
    gate_failed_index = None
    for i, spec in enumerate(task["graders"]):
        result = graders.grade(spec, ctx)
        ctx.results.append(result)
        weight = spec.get("weight", 0.0)
        total += result["score"] * weight / 100.0
        if spec.get("gate") and result["score"] < spec.get("gate_threshold", 100.0):
            gate_failed_index = i

    if gate_failed_index is not None:
        total = 0.0

    return {
        "total": max(0.0, min(100.0, total)),
        "gate_failed_index": gate_failed_index,
        "results": list(ctx.results),
    }


# =============================================================================
# Calibration (--calibrate, no LLM)
# =============================================================================


def synthetic_envelope(answer_text: str, mutated: bool, changed_paths: list[str], verified) -> dict:
    """Build a minimal, honest envelope for a synthetic (non-live) grading
    pass, matching main.py's real schema (main.py:_build_envelope) closely
    enough that every grader kind can read it uniformly."""
    return {
        "envelope": 1,
        "status": "ok",
        "error": None,
        "answer": answer_text,
        "verified": verified,
        "declared_unverified": False,
        "files_changed": list(changed_paths) if mutated else [],
        "verification_runs": [],
        "report": None,
        "artifacts_dir": None,
        "trace": None,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "llm_calls": 0},
        "session_id": "calibrate",
        "duration_s": 0.0,
    }


def _calibrate_one(task: dict) -> dict:
    """Grade the reference solution and a null baseline against *task*'s own
    graders, no LLM involved. A missing truth/<id>/ dir is a loud SKIP, not a
    failure -- most tasks won't have truth fixtures built yet mid-build."""
    task_id = task["id"]
    truth_dir = TRUTH_DIR / task_id
    if not truth_dir.is_dir():
        return {"task_id": task_id, "status": "SKIP", "reason": f"no truth dir at {truth_dir}"}

    reference_dir = truth_dir / "reference"
    reference_answer_path = truth_dir / "reference_answer.md"
    has_reference_dir = reference_dir.is_dir()
    if not reference_answer_path.is_file():
        return {"task_id": task_id, "status": "FAIL", "reason": f"missing {reference_answer_path}"}

    try:
        reference_answer = reference_answer_path.read_text(encoding="utf-8")
        null_answer_path = truth_dir / "null_answer.txt"
        null_answer = null_answer_path.read_text(encoding="utf-8") if null_answer_path.is_file() else ""

        scratch = Path(tempfile.mkdtemp(prefix=f"bench-calibrate-{task_id}-"))
        try:
            baseline_dir = scratch / "baseline"
            _materialize_fixture(task.get("fixture"), baseline_dir)

            # reference/ is a COMPLETE solution tree per contract ("fixture
            # with the fix applied; a full build for the greenfield task"),
            # not a diff to overlay -- copied as-is so a fix that deletes a
            # fixture file is representable. reference/ is only required
            # where the reference differs from the fixture (code-mode
            # fixes, greenfield builds); a task with no reference/ is a
            # read-only-mode contract (research / performance_debug) whose
            # ideal post-run tree IS the untouched fixture, so the pristine
            # fixture copy stands in as the reference tree.
            ref_tree = scratch / "reference_tree"
            if has_reference_dir:
                shutil.copytree(reference_dir, ref_tree)
            else:
                _materialize_fixture(task.get("fixture"), ref_tree)

            null_tree = scratch / "null_tree"
            _materialize_fixture(task.get("fixture"), null_tree)  # untouched copy

            ref_diff = graders.diff_trees(baseline_dir, ref_tree)
            ref_mutated = not ref_diff.is_empty
            # The reference is the ideal answer, so its truthful envelope is
            # derived from what it actually did to the tree (matching
            # turn/outcome.py's real tri-state contract) rather than from
            # whatever a task's own envelope_guard spec happens to declare
            # -- that would grade the synthetic envelope circularly against
            # the very field envelope_guard checks. Mutated => a code-mode
            # fix landed clean (verified True); untouched => the read-only
            # contract (verified None).
            ref_envelope = synthetic_envelope(
                reference_answer,
                mutated=ref_mutated,
                changed_paths=ref_diff.changed_paths,
                verified=True if ref_mutated else None,
            )
            ref_ctx = graders.GradeContext(
                task=task, tree=ref_tree, baseline=baseline_dir,
                envelope=ref_envelope, answer=reference_answer, truth_dir=truth_dir,
            )
            ref_result = grade_task(task, ref_ctx)

            # Uniform null rule: an untouched tree + a wrong/empty answer is
            # what an honest no-op run's envelope looks like regardless of
            # mode, so verified=None/files_changed=[] here always. A null
            # run that also lies about verification would be a *different*,
            # deliberately separate,
            # honesty-focused calibration case this suite does not need:
            # envelope_guard's null-case score here measures "did nothing,
            # said nothing false", which is a legitimately high score on
            # THAT one grader even though the task total must stay low
            # (driven down by the content graders, not by envelope_guard).
            null_envelope = synthetic_envelope(null_answer, mutated=False, changed_paths=[], verified=None)
            null_ctx = graders.GradeContext(
                task=task, tree=null_tree, baseline=baseline_dir,
                envelope=null_envelope, answer=null_answer, truth_dir=truth_dir,
            )
            null_result = grade_task(task, null_ctx)

            ok = ref_result["total"] >= 99.99 and null_result["total"] <= 5.0
            return {
                "task_id": task_id,
                "status": "PASS" if ok else "FAIL",
                "reference_total": ref_result["total"],
                "null_total": null_result["total"],
                "reference_results": ref_result["results"],
                "null_results": null_result["results"],
            }
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 -- reported as a FAIL row, never crashes the whole --calibrate run
        return {"task_id": task_id, "status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}


def _print_calibration_table(rows: list[dict]) -> bool:
    width = max([len("TASK")] + [len(r["task_id"]) for r in rows])
    header = f"{'TASK'.ljust(width)}  STATUS  REFERENCE     NULL   DETAIL"
    print(header)
    print("-" * len(header))
    ok = True
    for r in rows:
        status = r["status"]
        if status == "FAIL":
            ok = False
        ref = f"{r['reference_total']:.1f}" if "reference_total" in r else ""
        null = f"{r['null_total']:.1f}" if "null_total" in r else ""
        detail = r.get("reason", "")
        print(f"{r['task_id'].ljust(width)}  {status.ljust(6)}  {ref:>9}  {null:>6}   {detail}")
    return ok


# =============================================================================
# Live run (one rep)
# =============================================================================


def _classify_dnf(envelope: dict | None, timed_out: bool, timeout_reason: str | None) -> tuple[bool, str | None]:
    """Decide DNF status from the watchdog outcome + envelope-parse result.
    Pulled out of run_one_rep as its own pure function so _selftest.py can
    exercise every DNF branch without launching a real subprocess. A DNF
    always stays in the table (see "What is measured") -- this only decides
    the flag, never drops the row."""
    if timed_out:
        return True, timeout_reason
    if envelope is None:
        return True, "no parseable JSON envelope found on stdout"
    if envelope.get("status") == "error":
        return True, f"envelope status=error: {envelope.get('error')}"
    return False, None


def run_one_rep(task: dict, rep_idx: int, timeout_override: int | None, results_dir: Path) -> dict:
    """Run one task x rep against the real one-shot pipeline and grade it.
    Always returns a row dict, even on a DNF (timeout / unparseable envelope
    / envelope status=='error') -- a DNF stays in the table, it is never
    dropped (per "What is measured"). Grading still runs on whatever is
    gradable (the tree is always gradable; answer-based graders naturally
    score low against an empty/missing answer)."""
    task_id = task["id"]
    project_dir = Path(tempfile.mkdtemp(prefix=f"bench-{task_id}-r{rep_idx}-"))
    baseline_dir = Path(tempfile.mkdtemp(prefix=f"bench-{task_id}-r{rep_idx}-baseline-"))
    cfg_scratch = Path(tempfile.mkdtemp(prefix=f"bench-{task_id}-r{rep_idx}-cfg-"))
    try:
        _materialize_fixture(task.get("fixture"), project_dir)
        _materialize_fixture(task.get("fixture"), baseline_dir)
        _git_init_baseline(project_dir)

        cfg = build_config(task)
        cfg_path = cfg_scratch / "config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        cmd = [
            sys.executable, str(REPO_ROOT / "main.py"),
            "--config", str(cfg_path),
            "--mode", task["mode"],
            "--json", "-p", task["prompt"],
        ]
        env = dict(os.environ)

        hard_timeout_s = timeout_override if timeout_override is not None else task["timeout_s"]
        run_result = run_subprocess_with_watchdog(cmd, str(project_dir), env, hard_timeout_s, IDLE_TIMEOUT_S)

        envelope = parse_envelope(run_result["stdout"])
        spawned, spawned_line = detect_spawn_agents(run_result["stderr"])
        dnf, dnf_reason = _classify_dnf(envelope, run_result["timed_out"], run_result["timeout_reason"])

        answer = (envelope or {}).get("answer") or ""
        truth_dir = TRUTH_DIR / task_id
        ctx = graders.GradeContext(
            task=task,
            tree=project_dir,
            baseline=baseline_dir,
            envelope=envelope,
            answer=answer,
            truth_dir=truth_dir if truth_dir.is_dir() else None,
        )
        grade_result = grade_task(task, ctx)

        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / f"{task_id}.r{rep_idx}.stdout.txt").write_text(run_result["stdout"], encoding="utf-8")
        (results_dir / f"{task_id}.r{rep_idx}.stderr.txt").write_text(run_result["stderr"], encoding="utf-8")

        llm_cfg = cfg.get("llm", {})
        return {
            "task_id": task_id,
            "vertical": task["vertical"],
            "tier": task["tier"],
            "rep": rep_idx,
            "suite_version": tasks_mod.SUITE_VERSION,
            "dnf": dnf,
            "dnf_reason": dnf_reason,
            "spawned": spawned,
            "spawned_detail": spawned_line,
            "grader_results": grade_result["results"],
            "gate_failed_index": grade_result["gate_failed_index"],
            "total_score": grade_result["total"],
            "envelope": envelope,
            "wall_s": run_result["wall_s"],
            "duration_s_envelope": (envelope or {}).get("duration_s"),
            "config_snapshot": {
                "model": llm_cfg.get("model"),
                "base_url": llm_cfg.get("base_url"),
                "context_limit": llm_cfg.get("context_limit"),
                "head_sha": _git_rev_parse_head(),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)
        shutil.rmtree(baseline_dir, ignore_errors=True)
        shutil.rmtree(cfg_scratch, ignore_errors=True)


# =============================================================================
# Reporting
# =============================================================================


def append_run_row(results_dir: Path, row: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.median(values)


def write_summary(results_dir: Path, rows: list[dict], suite_version: int) -> None:
    lines = [
        "# Bench summary",
        "",
        f"- suite_version: {suite_version}",
        f"- results_dir: {results_dir}",
        f"- generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
    ]

    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)

    verticals = sorted({row["vertical"] for row in rows})
    for vertical in verticals:
        lines.append(f"## {vertical}")
        lines.append("")
        lines.append(
            "| task | tier | min | median | max | tokens (median) | llm_calls (median) | "
            "wall_s (median) | DNFs | honesty violations | spawned reps |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

        task_ids = sorted(t for t in by_task if by_task[t][0]["vertical"] == vertical)
        for task_id in task_ids:
            group = by_task[task_id]
            tier = group[0]["tier"]
            scores = [r["total_score"] for r in group]
            median_score = statistics.median(scores)
            dnf_count = sum(1 for r in group if r["dnf"])
            spawned_count = sum(1 for r in group if r["spawned"])
            honesty_violations = sum(
                1
                for r in group
                for gr in r["grader_results"]
                if gr.get("details", {}).get("envelope_lie")
            )

            token_totals = []
            llm_calls = []
            for r in group:
                usage = (r.get("envelope") or {}).get("usage") or {}
                pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
                if pt is not None and ct is not None:
                    token_totals.append(pt + ct)
                if usage.get("llm_calls") is not None:
                    llm_calls.append(usage.get("llm_calls"))

            tok_median = _median(token_totals)
            tok_str = "n/a" if tok_median is None else f"{tok_median:.0f}"
            # Addendum (verified 2026-08-04, "What is measured" hygiene bullet):
            # spawn_agents children are separate main.py subprocesses whose
            # token usage never reaches the parent's envelope. Any token
            # median drawn from a rep where spawn_agents fired understates
            # real consumption and must say so, not be presented as a total.
            if spawned_count > 0:
                tok_str += " (parent-only, under-counted)"

            calls_median = _median(llm_calls)
            calls_str = "n/a" if calls_median is None else f"{calls_median:.1f}"
            wall_median = _median([r["wall_s"] for r in group])
            wall_str = "n/a" if wall_median is None else f"{wall_median:.1f}"

            lines.append(
                f"| {task_id} | {tier} | {min(scores):.1f} | {median_score:.1f} | {max(scores):.1f} "
                f"| {tok_str} | {calls_str} | {wall_str} "
                f"| {dnf_count}/{len(group)} | {honesty_violations} | {spawned_count}/{len(group)} |"
            )
            if median_score >= 99.99:
                lines.append(
                    f"- NOTE: `{task_id}` median score is 100 — single-run saturation; "
                    "per the Freeze protocol's difficulty gate, this task needs a harder tier."
                )
        lines.append("")

    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_answers(results_dir: Path, rows: list[dict]) -> None:
    lines = ["# Bench answers (verbatim)", ""]
    for row in sorted(rows, key=lambda r: (r["task_id"], r["rep"])):
        answer = (row.get("envelope") or {}).get("answer")
        lines.append(f"## {row['task_id']} rep {row['rep']}")
        lines.append("")
        lines.append(f"- dnf: {row['dnf']}" + (f" ({row['dnf_reason']})" if row["dnf"] else ""))
        lines.append(f"- total_score: {row['total_score']:.1f}")
        lines.append(f"- spawned: {row['spawned']}")
        lines.append("")
        lines.append("```")
        lines.append(answer if answer is not None else "(no answer)")
        lines.append("```")
        lines.append("")
    (results_dir / "answers.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================


def _select_tasks(all_tasks: list[dict], vertical: str | None, task_id: str | None) -> list[dict]:
    selected = all_tasks
    if vertical:
        selected = [t for t in selected if t["vertical"] == vertical]
    if task_id:
        selected = [t for t in selected if t["id"] == task_id]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the coding_agent capability benchmark (evals/bench)")
    parser.add_argument("--list", action="store_true", help="List selected tasks and exit")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Grade each selected task's reference solution and a null baseline against its own graders (no LLM)",
    )
    parser.add_argument("--vertical", default=None, metavar="B1..B5", help="Only select tasks in this vertical")
    parser.add_argument("--task", default=None, metavar="TASK_ID", help="Only select this task id")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help=f"Repetitions per task (default: {DEFAULT_REPS})")
    parser.add_argument(
        "--results-dir", default=None, metavar="DIR",
        help="Where to write runs.jsonl/summary.md/answers.md (default: evals/bench/results/<timestamp>)",
    )
    parser.add_argument("--timeout", type=int, default=None, metavar="SECONDS", help="Override every selected task's timeout_s")
    args = parser.parse_args()

    if args.vertical and args.vertical not in _VALID_VERTICALS:
        parser.error(f"--vertical must be one of {_VALID_VERTICALS}, got {args.vertical!r}")
    if args.reps < 1:
        parser.error("--reps must be >= 1")

    all_tasks = tasks_mod.load_tasks()
    selected = _select_tasks(all_tasks, args.vertical, args.task)

    if args.list:
        if not selected:
            print("No tasks match (verticals/ may be empty/partial, or the filter matched nothing).")
            return 0
        width = max(len(t["id"]) for t in selected)
        for t in sorted(selected, key=lambda t: t["id"]):
            fixture = t["fixture"] or "(empty dir)"
            print(f"{t['id'].ljust(width)}  {t['vertical']}/{t['tier']}  mode={t['mode']}  fixture={fixture}  timeout_s={t['timeout_s']}")
        return 0

    if args.calibrate:
        if not selected:
            print("No tasks to calibrate (verticals/ may be empty/partial, or the filter matched nothing).")
            return 0
        rows = [_calibrate_one(t) for t in sorted(selected, key=lambda t: t["id"])]
        ok = _print_calibration_table(rows)
        return 0 if ok else 1

    if not selected:
        print("No tasks match --vertical/--task filter.", file=sys.stderr)
        return 1

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = Path(args.results_dir) if args.results_dir else (BENCH_DIR / "results" / timestamp)

    rows: list[dict] = []
    for task in sorted(selected, key=lambda t: t["id"]):
        for rep_idx in range(args.reps):
            print(f"running: {task['id']} rep {rep_idx + 1}/{args.reps} ...", file=sys.stderr)
            row = run_one_rep(task, rep_idx, args.timeout, results_dir)
            append_run_row(results_dir, row)
            rows.append(row)
            status = "DNF" if row["dnf"] else f"{row['total_score']:.1f}"
            print(f"  -> {status}", file=sys.stderr)

    write_summary(results_dir, rows, tasks_mod.SUITE_VERSION)
    write_answers(results_dir, rows)
    print(f"results written to: {results_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
