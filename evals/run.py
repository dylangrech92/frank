#!/usr/bin/env python3
"""Repeatable eval harness runner for coding_agent (H4).

Replays the scenarios declared in ``evals/scenarios.py`` against a real
``main.py`` REPL session (live LLM endpoint, config-driven) or, in
``--smoke`` mode, against a canned stub subprocess so the check-evaluation
and table-rendering plumbing can be verified offline with no network calls.

Usage:
    python3 evals/run.py                  # run every scenario (live)
    python3 evals/run.py --list            # list scenario names + descriptions
    python3 evals/run.py --only oversize   # run scenarios whose name contains 'oversize'
    python3 evals/run.py --timeout 600     # override the per-scenario timeout
    python3 evals/run.py --smoke           # offline plumbing check, no LLM calls

Exit code is 0 iff every selected scenario passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(REPO_ROOT))
from scenarios import SCENARIOS  # noqa: E402

DEFAULT_TIMEOUT = 420


def _validate_scenarios(scenarios: list[dict]) -> None:
    """Fail loudly at import time if a live (non-inline) scenario has no valid mode.

    A live scenario launches main.py, which now requires ``--mode`` for every
    agent-launching run (research/code/test/performance_debug fix the exact
    tool set for the whole process — see modes.py). A scenario missing
    ``mode`` would otherwise either crash main.py's argv parsing at run time
    with an error far from its cause, or — worse, if a blanket default were
    used instead — silently hand the scenario the wrong toolset and change
    what it measures without anyone noticing. Refusing to run at all is
    strictly better than guessing, so this runs unconditionally on import
    (including under --list), not just before a live run.
    """
    from modes import mode_names

    valid_modes = set(mode_names())
    problems: list[str] = []
    for s in scenarios:
        if s.get("inline"):
            continue
        mode = s.get("mode")
        if mode is None:
            problems.append(f"scenario {s['name']!r} is live (has 'turns', no 'inline') but has no 'mode' key")
        elif mode not in valid_modes:
            problems.append(
                f"scenario {s['name']!r} has mode {mode!r}, not one of modes.mode_names() {sorted(valid_modes)}"
            )
    if problems:
        for p in problems:
            print(f"FATAL: {p}", file=sys.stderr)
        raise SystemExit(1)


_validate_scenarios(SCENARIOS)


# =============================================================================
# Config merging
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


def build_config(scenario: dict) -> dict:
    """Return the merged config dict for *scenario* (repo config.json + overrides)."""
    base_cfg = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    overrides = scenario.get("config", "default")
    if overrides in (None, "default"):
        return base_cfg
    return deep_merge(base_cfg, overrides)


# =============================================================================
# Fixture setup
# =============================================================================


def _generate_bigfile(path: Path, target_bytes: int) -> None:
    """Write a ~*target_bytes*-byte python-ish text file to *path*."""
    header = '"""Generated oversize fixture module for evals — not meant to be imported."""\n\n'
    chunks = [header]
    total = len(header)
    i = 0
    while total < target_bytes:
        line = f"def padding_func_{i}(x):\n    return x + {i}  # padding line to inflate file size\n\n"
        chunks.append(line)
        total += len(line)
        i += 1
    path.write_text("".join(chunks), encoding="utf-8")


def materialize_setup(setup: dict, project_dir: Path) -> None:
    """Write every fixture file declared in *setup* into *project_dir*."""
    for rel_path, content in setup.items():
        if rel_path == "bigfile_bytes":
            continue
        target = project_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    if "bigfile_bytes" in setup:
        _generate_bigfile(project_dir / "bigfile.py", int(setup["bigfile_bytes"]))


# =============================================================================
# Check evaluation
# =============================================================================


def _stream_text(check: dict, stdout_text: str, stderr_text: str) -> str:
    return stdout_text if check.get("stream") == "stdout" else stderr_text


def run_checks(
    scenario: dict, stdout_text: str, stderr_text: str, project_dir: Path
) -> tuple[list[str], list[str]]:
    """Evaluate every check in *scenario* against the captured run.

    Returns:
        ``(failed, notes)`` — human-readable failure descriptions for checks
        that did not pass, and informational note strings for any
        ``regex-note`` checks (which never contribute to failure).
    """
    failed: list[str] = []
    notes: list[str] = []

    for check in scenario.get("checks", []):
        kind = check["kind"]

        if kind == "regex-present":
            text = _stream_text(check, stdout_text, stderr_text)
            if re.search(check["pattern"], text) is None:
                failed.append(
                    f"regex-present[{check['stream']}]: pattern {check['pattern']!r} not found"
                )

        elif kind == "regex-absent":
            text = _stream_text(check, stdout_text, stderr_text)
            if re.search(check["pattern"], text) is not None:
                failed.append(
                    f"regex-absent[{check['stream']}]: pattern {check['pattern']!r} unexpectedly found"
                )

        elif kind == "ordered":
            text = _stream_text(check, stdout_text, stderr_text)
            pos = 0
            for pattern in check["patterns"]:
                match = re.search(pattern, text[pos:])
                if match is None:
                    failed.append(
                        f"ordered[{check['stream']}]: pattern {pattern!r} not found "
                        f"after position {pos}"
                    )
                    break
                pos += match.end()

        elif kind == "regex-note":
            text = _stream_text(check, stdout_text, stderr_text)
            present = re.search(check["pattern"], text) is not None
            notes.append(
                f"note[{check['stream']}] {check['pattern']!r}: "
                f"{'PRESENT' if present else 'ABSENT'}"
            )

        elif kind == "file-lines-min":
            file_path = project_dir / check["path"]
            if not file_path.exists():
                failed.append(f"file-lines-min: {check['path']} does not exist")
            else:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                nonblank = [line for line in text.splitlines() if line.strip() != ""]
                if len(nonblank) < check["min"]:
                    failed.append(
                        f"file-lines-min: {check['path']} has {len(nonblank)} "
                        f"non-blank line(s), need >= {check['min']}"
                    )

        else:
            failed.append(f"unknown check kind {kind!r}")

    return failed, notes


# =============================================================================
# Scenario execution
# =============================================================================


def build_live_cmd(scenario: dict, cfg_path: Path, smoke: bool) -> list[str]:
    """Build the child-process argv for one live (non-inline) scenario run.

    Every live scenario must declare a ``mode`` (validated at import time by
    ``_validate_scenarios``); that mode is threaded through as ``--mode`` for
    both the smoke-stub branch and the real ``main.py`` branch so the two
    stay symmetric and neither path can silently launch modeless.
    """
    mode = scenario["mode"]
    if smoke:
        return [
            sys.executable,
            str(EVALS_DIR / "_smoke_stub.py"),
            "--config",
            str(cfg_path),
            "--scenario",
            scenario["name"],
            "--mode",
            mode,
        ]
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--config",
        str(cfg_path),
        "--mode",
        mode,
    ]


def run_live_scenario(scenario: dict, timeout: int, smoke: bool) -> dict:
    """Run one non-inline scenario in a fresh temp project dir and score it."""
    project_dir = Path(tempfile.mkdtemp(prefix=f"evalproj-{scenario['name']}-"))
    try:
        materialize_setup(scenario.get("setup", {}) or {}, project_dir)

        merged_cfg = build_config(scenario)
        cfg_path = project_dir / "_eval_config.json"
        cfg_path.write_text(json.dumps(merged_cfg), encoding="utf-8")

        stdin_text = "\n".join(scenario["turns"]) + "\n"

        cmd = build_live_cmd(scenario, cfg_path, smoke)

        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(project_dir),
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout_text, stderr_text = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            # TimeoutExpired carries the partial output as *bytes* even when
            # the run was started with text=True (CPython quirk) — decode.
            def _as_text(data: bytes | str | None) -> str:
                if data is None:
                    return ""
                if isinstance(data, bytes):
                    return data.decode("utf-8", errors="replace")
                return data

            stdout_text = _as_text(exc.stdout)
            stderr_text = _as_text(exc.stderr)
            timed_out = True

        failed, notes = run_checks(scenario, stdout_text, stderr_text, project_dir)
        if timed_out:
            failed = [f"scenario timed out after {timeout}s"] + failed

        return {
            "name": scenario["name"],
            "stdout": stdout_text,
            "stderr": stderr_text,
            "failed": failed,
            "notes": notes,
            "passed": not failed,
        }
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def run_inline_scenario(scenario: dict, timeout: int) -> dict:
    """Run a dispatch-level ``inline`` scenario as a standalone script."""
    script_path = EVALS_DIR / scenario["inline"]
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing_pp if existing_pp else "")

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        stdout_text, stderr_text = proc.stdout, proc.stderr
        passed = proc.returncode == 0
        failed = [] if passed else [f"inline script exited with code {proc.returncode}"]
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        failed = [f"inline scenario timed out after {timeout}s"]
        passed = False

    return {
        "name": scenario["name"],
        "stdout": stdout_text,
        "stderr": stderr_text,
        "failed": failed,
        "notes": [],
        "passed": passed,
    }


# =============================================================================
# Reporting
# =============================================================================


def write_transcript(results_dir: Path, name: str, stdout_text: str, stderr_text: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{name}.out.txt").write_text(stdout_text, encoding="utf-8")
    (results_dir / f"{name}.err.txt").write_text(stderr_text, encoding="utf-8")


def print_table(results: list[dict]) -> None:
    name_width = max([len("SCENARIO")] + [len(r["name"]) for r in results])
    header = f"{'SCENARIO'.ljust(name_width)}  RESULT  FAILED CHECKS"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        first_line = r["failed"][0] if r["failed"] else ""
        print(f"{r['name'].ljust(name_width)}  {status.ljust(6)}  {first_line}")
        for extra in r["failed"][1:]:
            print(f"{' ' * name_width}          {extra}")
        for note in r.get("notes", []):
            print(f"{' ' * name_width}          {note}")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Run coding_agent eval scenarios")
    parser.add_argument(
        "--only", default=None, metavar="SUBSTR",
        help="Only run scenarios whose name contains SUBSTR",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List scenario names and descriptions, then exit",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-scenario subprocess timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Offline plumbing check: replace the main.py invocation with a "
            "canned stub subprocess so check-evaluation and table-rendering "
            "are exercised without a live LLM endpoint"
        ),
    )
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in scenarios if args.only in s["name"]]

    if args.list:
        if not scenarios:
            print("No scenarios match.")
            return 0
        name_width = max(len(s["name"]) for s in scenarios)
        for s in scenarios:
            print(f"{s['name'].ljust(name_width)}  {s['description']}")
        return 0

    if not scenarios:
        print("No scenarios match --only filter.", file=sys.stderr)
        return 1

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = EVALS_DIR / "results" / timestamp

    results: list[dict] = []
    for scenario in scenarios:
        print(f"running: {scenario['name']} ...", file=sys.stderr)
        if scenario.get("inline"):
            result = run_inline_scenario(scenario, args.timeout)
        else:
            result = run_live_scenario(scenario, args.timeout, args.smoke)

        write_transcript(results_dir, scenario["name"], result["stdout"], result["stderr"])
        results.append(result)

    print()
    print_table(results)
    print()
    print(f"transcripts written to: {results_dir}")

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
