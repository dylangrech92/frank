"""tasks — aggregates the frozen task list from every vertical stream's
``verticals/<v>.py`` module, validating each entry against the contract
schema fixed in DESIGN.md's "Integration contracts (build-phase)" section.

``SUITE_VERSION`` lives here per the Freeze protocol: any later change to a
fixture, prompt, or grader bumps it, and every result row records it.

Deliberately does NOT validate at import time (unlike the module-level
``TASKS = load_tasks()`` precedent in ``evals/run.py``): ``verticals/`` is
five other build streams' work in progress, and ``_selftest.py`` needs to
exercise ``build_tasks()`` against a fabricated in-memory task list without
a real filesystem scan of that directory ever running (a broken or
half-written sibling module must not be able to crash the self-test). Call
``load_tasks()`` explicitly (``run.py`` does, at the start of ``main()``).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import modes

SUITE_VERSION = 2

BENCH_DIR = Path(__file__).resolve().parent
VERTICALS_DIR = BENCH_DIR / "verticals"

_VALID_VERTICALS = {"B1", "B2", "B3", "B4", "B5"}
_VALID_TIERS = {"T1", "T2", "T3", "multi"}
_KNOWN_GRADER_KINDS = (
    "answer_facts",
    "acceptance",
    "tree_guard",
    "envelope_guard",
    "findings_list",
    "perf_report",
)
_ID_RE = re.compile(r"^(b[1-5])_(t[1-3]|multi)_[a-z0-9]+(?:_[a-z0-9]+)*$")
_TREE_GUARD_MODES = ("byte_identical", "confined_diff", "pollution_whitelist")


class TaskValidationError(ValueError):
    """One or more task entries failed contract validation.

    ``problems`` carries every violation found (not just the first), each
    prefixed with its source module and, where applicable, the task id --
    a broken vertical module should report everything wrong in one pass,
    not dole it out one round-trip at a time.
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(problems))


def _load_vertical_modules() -> list[tuple[str, list]]:
    """Return [(source_name, TASKS_list), ...] for every non-underscore
    ``.py`` file directly under ``verticals/``. Tolerates a missing or
    empty directory (returns []) -- siblings may not have landed yet."""
    if not VERTICALS_DIR.is_dir():
        return []

    sources: list[tuple[str, list]] = []
    for path in sorted(VERTICALS_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        module_name = f"_bench_vertical_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not build an import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 -- re-wrapped, not swallowed
            raise ImportError(f"vertical module {path.name} failed to import: {exc}") from exc

        tasks_list = getattr(module, "TASKS", None)
        if tasks_list is None:
            raise ImportError(f"vertical module {path.name} has no module-level TASKS list")
        if not isinstance(tasks_list, list):
            raise ImportError(f"vertical module {path.name}: TASKS must be a list, got {type(tasks_list).__name__}")
        sources.append((path.name, tasks_list))
    return sources


def _validate_task(entry: object, source: str, seen_ids: dict[str, str], problems: list[str]) -> None:
    if not isinstance(entry, dict):
        problems.append(f"{source}: task entry is not a dict (got {type(entry).__name__})")
        return

    task_id = entry.get("id")
    prefix = f"{source} task {task_id!r}" if isinstance(task_id, str) else f"{source} task <no id>"

    if not isinstance(task_id, str) or not task_id:
        problems.append(f"{prefix}: 'id' must be a non-empty string")
    else:
        m = _ID_RE.match(task_id)
        if not m:
            problems.append(f"{prefix}: 'id' {task_id!r} does not match '<vertical>_<tier>_<slug>' (e.g. b1_t2_cache_corruption)")
        if task_id in seen_ids:
            problems.append(f"{prefix}: duplicate id, already declared in {seen_ids[task_id]}")
        else:
            seen_ids[task_id] = source

    vertical = entry.get("vertical")
    if vertical not in _VALID_VERTICALS:
        problems.append(f"{prefix}: 'vertical' must be one of {sorted(_VALID_VERTICALS)}, got {vertical!r}")
    elif isinstance(task_id, str) and _ID_RE.match(task_id) and not task_id.lower().startswith(vertical.lower() + "_"):
        problems.append(f"{prefix}: id prefix does not match vertical {vertical!r}")

    tier = entry.get("tier")
    if tier not in _VALID_TIERS:
        problems.append(f"{prefix}: 'tier' must be one of {sorted(_VALID_TIERS)}, got {tier!r}")
    elif isinstance(task_id, str) and _ID_RE.match(task_id):
        tier_part = task_id.split("_")[1]
        if tier_part != tier.lower():
            problems.append(f"{prefix}: id tier segment {tier_part!r} does not match tier {tier!r}")

    mode = entry.get("mode")
    if mode not in modes.MODES:
        problems.append(f"{prefix}: 'mode' must be one of {sorted(modes.MODES)}, got {mode!r}")

    fixture = entry.get("fixture")
    if fixture is not None and not isinstance(fixture, str):
        problems.append(f"{prefix}: 'fixture' must be a string or None, got {type(fixture).__name__}")

    prompt = entry.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        problems.append(f"{prefix}: 'prompt' must be a non-empty string")

    timeout_s = entry.get("timeout_s")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s <= 0:
        problems.append(f"{prefix}: 'timeout_s' must be a positive int, got {timeout_s!r}")

    graders = entry.get("graders")
    if not isinstance(graders, list) or not graders:
        problems.append(f"{prefix}: 'graders' must be a non-empty list")
        return

    total_weight = 0.0
    for i, g in enumerate(graders):
        gprefix = f"{prefix} graders[{i}]"
        if not isinstance(g, dict):
            problems.append(f"{gprefix}: must be a dict, got {type(g).__name__}")
            continue
        kind = g.get("kind")
        if kind not in _KNOWN_GRADER_KINDS:
            problems.append(f"{gprefix}: 'kind' must be one of {_KNOWN_GRADER_KINDS}, got {kind!r}")

        gate = g.get("gate", False)
        if not isinstance(gate, bool):
            problems.append(f"{gprefix}: 'gate' must be a bool, got {gate!r}")
            gate = False  # fall back so the weight check below still runs sensibly

        # A grader's weight must be positive (it contributes to the weighted
        # total), OR exactly 0 paired with gate: true (a pure veto -- scores
        # nothing but can still zero the whole task via the gate mechanic).
        # weight 0 without gate: true is dead spec: it neither scores nor
        # gates, so it's a validation error rather than a silent no-op.
        weight = g.get("weight")
        is_number = isinstance(weight, (int, float)) and not isinstance(weight, bool)
        if not is_number:
            problems.append(f"{gprefix}: 'weight' must be a positive number (or exactly 0 with gate: true), got {weight!r}")
        elif weight > 0:
            total_weight += float(weight)
        elif weight == 0:
            if not gate:
                problems.append(
                    f"{gprefix}: 'weight' is 0 but 'gate' is not true -- a weight-0 grader must carry "
                    "gate: true (a pure veto) or else use a positive weight"
                )
        else:
            problems.append(f"{gprefix}: 'weight' must be a positive number (or exactly 0 with gate: true), got {weight!r}")

        requires = g.get("requires", [])
        if not isinstance(requires, list) or not all(isinstance(r, int) and not isinstance(r, bool) for r in requires):
            problems.append(f"{gprefix}: 'requires' must be a list of ints, got {requires!r}")
        else:
            for r in requires:
                if not (0 <= r < i):
                    problems.append(f"{gprefix}: 'requires' index {r} must point to an earlier grader in this same list (0..{i - 1})")

        # Minimal per-kind required-field validation: catch authoring bugs
        # at --list time instead of mid-calibration/mid-run (the graders
        # themselves keep their own defensive checks too -- belt and
        # suspenders, not a replacement).
        if kind == "tree_guard":
            mode = g.get("mode")
            if mode not in _TREE_GUARD_MODES:
                problems.append(f"{gprefix}: tree_guard 'mode' must be one of {_TREE_GUARD_MODES}, got {mode!r}")
            elif mode == "confined_diff" and not g.get("allowed_paths"):
                problems.append(f"{gprefix}: tree_guard mode=confined_diff requires a non-empty 'allowed_paths'")
            elif mode == "pollution_whitelist" and not g.get("pollution_patterns"):
                problems.append(f"{gprefix}: tree_guard mode=pollution_whitelist requires a non-empty 'pollution_patterns'")
        elif kind == "envelope_guard" and "expect_verified" in g:
            if g["expect_verified"] not in (True, False, None):
                problems.append(f"{gprefix}: envelope_guard 'expect_verified' must be True, False, or None, got {g['expect_verified']!r}")

    if abs(total_weight - 100.0) > 1e-6:
        problems.append(f"{prefix}: positive graders weights must sum to 100, got {total_weight}")


def build_tasks(sources: list[tuple[str, list | None]]) -> list[dict]:
    """Pure aggregation + validation over already-loaded ``[(source, TASKS), ...]``
    pairs. Raises ``TaskValidationError`` (carrying every problem found) on any
    contract violation. Does no filesystem I/O -- callers load ``sources``
    themselves (``load_tasks()`` via ``_load_vertical_modules()`` for the real
    suite; ``_selftest.py`` by constructing ``sources`` in memory)."""
    problems: list[str] = []
    seen_ids: dict[str, str] = {}
    all_tasks: list[dict] = []

    for source, tasks_list in sources:
        if tasks_list is None:
            continue
        for entry in tasks_list:
            _validate_task(entry, source, seen_ids, problems)
            if isinstance(entry, dict):
                all_tasks.append(entry)

    if problems:
        raise TaskValidationError(problems)

    return all_tasks


def load_tasks() -> list[dict]:
    """Real entry point: scan ``verticals/``, aggregate, validate. On a
    contract violation, print every problem to stderr and exit(1) -- a
    broken task list must never silently produce a partial suite."""
    try:
        sources = _load_vertical_modules()
    except ImportError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        return build_tasks(sources)
    except TaskValidationError as exc:
        for problem in exc.problems:
            print(f"FATAL: {problem}", file=sys.stderr)
        raise SystemExit(1) from exc
