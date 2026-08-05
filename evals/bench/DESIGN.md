# Capability Benchmark — design

A scored benchmark that runs the **real harness against the real configured
model** on a frozen set of hard tasks, asserts correctness with deterministic
graders, and records tokens and wall time. It answers one question over time:
*is the harness+model getting more capable, faster, and cheaper — measured, not
felt?*

This is a separate subsystem from the regression evals (`evals/run.py` +
`scenarios.py`). Those are binary pass/fail commit gates for harness
*mechanisms*; this benchmark produces graded *scores* for harness+model
*capability*. A benchmark regression must never block a commit, and a commit
gate must never be graded on a curve — the two lifecycles stay separate.

## Problem

Harness changes land with per-mechanism proof but there is no standing measure
of whole-system capability: whether bug fixing, project delivery, code
discovery, performance diagnosis, and code auditing are actually improving —
or silently regressing — release over release, and at what token/time cost.

## Design stance: no toy tasks

**A perfect score is a defect in the suite, not a triumph of the system.**
The suite exists to expose headroom. Two-sided validity is enforced at freeze
time and forever after:

- **Grader validity (floor):** every grader, run against a committed reference
  solution, must award full marks; run against the untouched fixture / an
  empty answer, it must award zero. Automated as `--calibrate` (no LLM
  involved). This kills vacuous checks — an assertion derived from the system
  under test always passes, so every expectation comes from a ground-truth
  card written from outside the code.
- **Difficulty validity (ceiling):** at freeze time, the live harness+model
  must score **materially below ceiling on every vertical** (target band:
  30–70% on the hard tier, near-zero on the frontier tier). If a pilot run
  saturates a vertical, the vertical gains harder tiers *before* the suite is
  frozen.
- **Standing saturation rule:** if any tier's median correctness hits 100%
  across two consecutive counted runs, the suite is extended with a harder
  tier and the version is bumped. Scores are only comparable within a suite
  version.

Every task is tiered:

| Tier | Intent | Expected today |
|---|---|---|
| T1 baseline | sanity anchor — a broken harness fails it | pass |
| T2 hard | the real signal band | partial |
| T3 frontier | known-hard shapes; today's ceiling | mostly fail |

T3 tasks are built from **recorded live failure shapes** (documented in the
project's run history), not invented difficulty: alias/re-export indirection
that defeats LSP call hierarchies, wrong-output bugs whose cause is far from
the symptom, multi-part specs where trailing sub-tasks get dropped,
bottlenecks that only dominate at scale. Difficulty is evidence-anchored.

## What is measured

Every run is one invocation of the real one-shot pipeline:

```
python3 main.py --mode <mode> --json -p "<frozen prompt>"
```

run with `cwd` = a fresh temp copy of the task's fixture. The `--json`
envelope (`main.py:_build_envelope`) already carries everything the metrics
need:

| Metric | Source |
|---|---|
| Correctness (0–100 per task) | graders run **after** the agent exits, against the answer + the resulting tree |
| Prompt / completion tokens | `envelope.usage.prompt_tokens` / `completion_tokens` |
| LLM round-trips | `envelope.usage.llm_calls` |
| Wall time | `envelope.duration_s` (subprocess wall time as cross-check) |
| Envelope honesty | graded verdict vs `envelope.verified` / `files_changed` (see below) |
| DNF | timeout or `status: "error"` — stays in the table, never dropped |

**Envelope honesty is a first-class column.** The harness's purpose is to be
trusted by an orchestrator, so a run where `verified: true` but the hidden
acceptance suite fails is recorded as an *envelope lie* — a worse outcome than
an honest failure. Symmetrically: research-vertical runs must report
`verified: null` with an empty `files_changed` and a byte-identical tree.

Measurement hygiene:

- Benchmark runs record real usage into the stats ledger by design (owner
  decision 2026-08-04, reversing the initial suppress-stats stance): a
  benchmark is real usage and its telemetry must be dashboard-visible.
  Bench sessions are identifiable by their `bench-<task>-r<rep>` session
  naming. (`evals/run.py`'s commit-gate suite still suppresses stats — that
  traffic is synthetic.)
- Memory is always on and per-project (`.coding_agent/memory.db` inside the
  project root; the CLI deliberately has no way to disable it). Rep isolation
  comes from the runner, not a flag: every rep runs in a fresh temp copy of
  the fixture, so per-project memory starts cold by construction and
  consolidation output dies with the temp dir. Cold-start memory behavior
  (orientation seeding, consolidation cost) is part of what the benchmark
  measures — it is the same cost every fresh project pays.
- Graders treat `.coding_agent/` as harness-owned state: the byte-identical
  tree check on research runs, the delivery pollution probes, and the
  no-collateral diff checks all exclude it.
- **Token accounting counts the parent session only — verified 2026-08-04.**
  `spawn_agents` children are separate `main.py` subprocesses with their own
  sessions (`tools/spawn_agents.py:250` launches without `--json`; the file
  never touches `usage` or `turn_report`), and `_build_envelope` reads only
  the parent's `turn_report` — child tokens never reach the envelope.
  Consequence: the runner records a `spawned` flag per rep (detected from the
  parent's stderr telemetry naming a `spawn_agents` call), and `summary.md`
  annotates token medians computed over any flagged rep as **parent-only,
  under-counted**. Fixing the harness to aggregate child usage (sum into
  `usage`, or a separate `usage.children` field) is a product decision left
  to the owner — the benchmark documents the semantics it measures rather
  than silently changing them.
- N=3 reps per task. Tokens/time aggregate as median; correctness is reported
  as min/median/max — correctness *variance* is itself capability signal.
- Timeouts are per-task (default 30 min; delivery 90 min) plus a
  zero-activity kill (no stderr telemetry for 15 min — deliberately longer
  than the harness's own 600s stream-read timeout so the runner never races
  the harness's stall recovery; the pilot's five DNFs were all that race).

## Architecture

```
evals/bench/
  DESIGN.md            this file
  run.py               runner: copy fixture → launch → capture → grade → report
  tasks.py             frozen task list (SUITE_VERSION lives here)
  graders/             deterministic graders (stdlib-only)
  fixtures/<name>/     committed fixture projects the agent sees
  truth/<task_id>/     ground-truth cards, hidden acceptance suites,
                       reference solutions — NEVER copied into the fixture
  results/<ts>/        per-run JSON + answers.md + summary.md (git-ignored)
```

- **Fixtures** are real, hand-authored mini-applications (Python,
  stdlib-only — no installs, no network), committed in-repo. Each is copied
  to a fresh temp dir per run; the agent never sees `truth/`.
- **Ground truth lives outside the fixture.** Hidden acceptance suites are
  executed from `truth/` against the post-run tree; they are never present in
  the tree the agent can read, so they cannot be gamed or show up as diff
  noise.
- **Graders are deterministic and stdlib-only.** Regex-based answer graders
  wrap every match in `bool()` (a `re.Match` in a result dict is not JSON
  serializable — a recorded, repeated failure). Regexes tolerate wording
  variance (backticks/quotes around symbols, "could not be reproduced"
  phrasings — all previously recorded scorer misses).
- **Scorer verdicts are screening, not truth.** Every run's raw answer is
  collated into `results/<ts>/answers.md`; the human read of that file is part
  of the counted-run protocol. A scorer disagreement is resolved by the
  answer text, and the regex gets patched under a version bump.
- No LLM judge in v1. Deterministic graders + mandatory human read. A judge
  adds a second model dependency to a suite that must be self-contained, and
  the recorded history of regex scorers already shows how often automated
  grading misreads answers — a judge fails less visibly.

## The five verticals

### B1 — Bug fixing (`--mode code`, 3 tasks)

Fixture `queueworks/` (~1.5k LOC): a job-queue library with workers, retry
policy, and a persistent store. Prompt = a user-style bug report with a repro
command. `code` mode is the evidenced choice: `qa` has no edit tools
(`modes.py`), while `code` carries editors + `run_command` + `verify_scratch`.

- **T1 crash:** clear traceback, single-site fix.
- **T2 wrong output, action at a distance:** the cause (an in-place mutation
  of a module-cached list) lives two modules away from the symptom (a
  corrupted chronological report). No crash, no traceback.
- **T3 interacting pair:** the reported symptom has *two* cooperating causes;
  fixing only the visible one shifts the symptom instead of clearing it. The
  hidden suite also probes a nearby behavior a naive fix breaks.

Scoring per task (0–100): diagnosis 20 (answer names the causal file +
mechanism, per-card regexes), fix 50 (hidden acceptance suite pass fraction),
no-collateral 20 (hidden regression probes + diff confined to plausible
files), honesty 10 (`verified` matches the graded outcome; edits actually ran
a verification).

### B2 — Project delivery (`--mode code`, 1 task, 90-min cap)

No fixture — greenfield in an empty dir from a single frozen spec:
a dependency-aware job scheduler CLI (schedule parsing, DAG resolution with
cycle detection, simulated-clock run, crash-safe state persistence and
recovery, log rotation, query commands, a documented exit-code contract,
README + CHANGELOG). Ten numbered requirements, several deliberately
interacting (the persistence requirement constrains how the DAG runner must
be structured; the exit-code contract applies to every command including the
ones added last).

Hidden acceptance suite: ~40 behavioral probes in three bands — core (each
requirement's happy path), edge (empty inputs, unicode names, boundary
values, malformed schedule lines, corrupt state file recovery), adversarial
(exact error-contract adherence, idempotent re-run, the trailing
documentation sub-tasks that historically get dropped). Score = weighted
pass fraction (core 50, edge 30, adversarial 20). Tree pollution (state/log
files left behind by the agent's own verification runs) is graded against —
a recorded live failure shape.

### B3 — Discovery (`--mode research`, 6 single-question tasks)

Fixture `relay/` (~1.6k LOC, 39 files — landed denser than the ~2.5k
sketch: indirection per line beat padding): an event-pipeline application built
specifically to defeat shallow search: string-keyed dispatch tables,
decorator-based handler registration, a compatibility shim re-exporting
symbols under legacy names (LSP call-hierarchy traversal stops at aliased
re-exports — a recorded blind spot), callbacks passed as values, one dynamic
import. Every question has a **named trap**: the plausible-wrong answer a
grep-level investigation lands on.

- T1 (×1): where is symbol X implemented; who calls it directly.
- T2 (×3): the complete caller set of Y including registry and aliased paths;
  an end-to-end trace of event Z from entry to sink; which callees of W can
  raise and what happens to the error.
- T3 (×2): which config flag governs retry behavior and what breaks if it is
  removed (cross-file synthesis); every code path that can write to the
  dead-letter file (exhaustive enumeration, recall-scored).

Scoring per question: required-fact recall (each fact = tolerant regex set)
70, trap avoidance 20 (naming the trap answer as truth zeroes this band),
coherence 10 (ordered-narrative check: the answer presents a chain, not a
file dump; screened by graders, confirmed by the human read). Envelope
checks: `verified: null`, `files_changed: []`, byte-identical tree — a
research run that mutates the fixture scores 0 regardless of answer quality.

### B4 — Performance diagnosis (`--mode performance_debug`, 2 tasks)

Fixture `grind/` (~1k LOC): a data-crunching pipeline with a deterministic
workload entry point (`python3 workload.py --scale N`). Planted, per the
ground-truth card:

- **T1:** one dominant CPU hotspot, visible at the top of any profile.
- **T2 (CPU task):** the true second cost is a stdlib-attributed pattern
  (per-row `json.loads` / re-compilation inside a loop) — the profile blames
  library frames; credit requires mapping attribution back to the calling
  site. A cold decoy (an ugly nested loop on a never-hot path) is planted;
  naming it as a bottleneck costs precision.
- **T2 (memory task):** an unbounded memoization cache — growth, not speed;
  requires the memory profiler, not the CPU one.
- **T3:** an O(n²) accumulation that is *invisible at the default scale* and
  dominant at 10× — the prompt asks what must be fixed before a 10× growth,
  so credit requires either profiling at a second scale or measured
  extrapolation. Naming it from code-smell alone without measurement earns
  half credit; the mode's contract is a measured report.

Scoring: true-positive hotspots named with file+function 50, measured
evidence attached to each claim (digits + units near the claim — %, seconds,
call counts, KiB) 30, ranking correct 10, decoy avoidance 10.

### B5 — Silent bugs & dead code (`--mode research`, 1 task)

Fixture `ledger/` (~2k LOC): a batch CSV-import/aggregate/report application.
The prompt demands a **numbered findings list** (id, file, one-line claim) —
parseable output is part of the capability under test. Ground-truth card:

- **T1 (lint-adjacent):** a swallowed exception hiding I/O errors; a dead
  function; an unreachable branch; an unused config flag.
- **T2 (cross-file reasoning):** a mutable default argument accumulating
  state across batches; an in-place sort corrupting a shared cache; float
  equality on currency values.
- **T3 (intent-level):** a cache key built from the wrong field (stale
  results only when two entities share a name); a pagination off-by-one that
  drops the final record only when the page size divides the count; a
  double-applied timezone conversion visible only across a DST boundary.
- **Decoys (precision traps):** handlers reachable only via `getattr`
  dispatch (dead-looking, alive); a deliberate re-raise wrapper that
  pattern-matches "swallowed exception".

Scoring: recall per tier (T1 20, T2 30, T3 30) + precision 20 — every claim
in the findings list is matched against card + decoy entries; unmatched
claims and decoy hits cost precision. A 50-item "maybe" dump must lose to a
short, correct report.

## Runner protocol

Per task × rep: copy `fixtures/<name>` → temp dir (empty dir for B2) →
launch the one-shot command with the merged config (`config.json` +
per-task overrides, reusing `deep_merge`/config plumbing from `evals/run.py`)
→ capture stdout (envelope), stderr (telemetry) → run the task's graders
from `truth/` against answer + tree → append one JSON row to
`results/<ts>/runs.jsonl` (envelope, grades, suite version, config snapshot:
model, endpoint, context limit, HEAD sha) → collate `answers.md` and
`summary.md` (per-vertical table: correctness min/med/max per tier, tokens,
llm_calls, wall, honesty violations, DNFs).

CLI: `--vertical`, `--task`, `--reps`, `--timeout`, `--calibrate` (graders
vs reference solutions + null baselines, no LLM), `--list`.

A full counted run (13 tasks × 3 reps, local-model speeds) is an overnight
job by design; the filters exist for iteration, but only full runs at N=3 are
counted for the longitudinal record.

## Integration contracts (build-phase)

Six independent build streams share `evals/bench/`; these contracts keep
their outputs composable. File ownership is exclusive — no two streams write
the same path, and no stream edits this file (contract problems are reported
back, not patched around).

### Layout & ownership

```
evals/bench/
  run.py  graders/  tasks.py  _selftest.py  .gitignore    runner stream
  verticals/<v>.py            one module per vertical stream (b1.py … b5.py)
  fixtures/<name>/            owned by the vertical stream that declared it
  truth/<task_id>/            owned by the vertical stream that owns the task
```

`tasks.py` aggregates dynamically: it imports every non-underscore module in
`verticals/` and concatenates their `TASKS` lists, failing loudly on a broken
module. `_`-prefixed modules are ignored (runner self-test stubs).

### Task entry (`verticals/<v>.py`, module-level `TASKS` list)

```python
{
    "id": "b1_t2_cache_corruption",   # <vertical>_<tier|multi>_<slug>, unique
    "vertical": "B1", "tier": "T2",
    "mode": "code",                    # key of modes.MODES
    "fixture": "queueworks",           # dir under fixtures/, or None = empty dir
    "prompt": "…frozen verbatim…",
    "timeout_s": 1800,
    "graders": [{"kind": "answer_facts", "weight": 20}, …],  # weights sum to 100
}
```

Amendments settled during the build (integration pass, 2026-08-04):

- **`tier: "multi"`** is valid for a task that spans tiers inside one answer
  (B4's hotspot report, B5's audit) — legal only when the task's card
  facts/items each carry their own tier, because per-tier scoring then
  happens at item level. Id pattern: `<vertical>_(t1|t2|t3|multi)_<slug>`.
  The difficulty gate reads per-tier components from these tasks' grader
  details, not the task-level label.
- **Weight-0 grader entries are pure gates**: valid only with `gate: true`
  (the runner's gate mechanic zeroes the whole task when a gated grader
  misses its threshold). A weight-0 entry without `gate: true` is a
  validation error — it would neither score nor veto, i.e. dead spec.
  Positive weights alone must sum to 100.
- **Per-kind spec fields are normative**, documented in each grader module's
  docstring. The load-bearing ones: `tree_guard` requires `mode`
  (`byte_identical` | `confined_diff` + `allowed_paths` |
  `pollution_whitelist` + `pollution_patterns`); `envelope_guard` takes
  `expect_verified` (True for code-mode tasks with an acceptance suite —
  the default None demands a byte-identical tree and is only correct for
  read-only modes); `acceptance` takes `band_weights`.
- **Calibration without `reference/`**: read-only-mode tasks (research /
  performance_debug) mutate nothing, so the ideal post-run tree IS the
  pristine fixture — `--calibrate` uses the fixture copy as the reference
  tree and `reference_answer.md` as the answer. `reference/` is required
  only where the reference differs from the fixture (code-mode tasks,
  greenfield builds).

### `truth/<task_id>/` layout

- `card.json` — ground-truth card driving the answer graders.
- `acceptance/probe_<band>_<slug>.py` — executable probes run post-hoc from
  outside the tree: `python3 probe_x.py <post_run_tree>`; exit 0 = pass;
  stdlib-only. Band ∈ `core|edge|adversarial` and carries the grader's band
  weighting (delivery: 50/30/20; single-band tasks use `core` only).
- `reference/` — a complete correct solution tree (fixture with the fix
  applied; a full build for the greenfield task). `--calibrate` must score
  it 100.
- `reference_answer.md` — an ideal answer text: every card fact must match
  it, no trap may match it. This is what the answer graders calibrate to
  100 against (a reference tree alone cannot calibrate an answer grader).
- `null_answer.txt` (optional) — a plausible-wrong answer for the zero
  calibration check; default null = untouched fixture + empty answer.
- `validation/` (optional) — the author's evidence scripts (bug
  demonstrations, profiler runs). Never read by graders; kept for audit.

### `card.json`

```json
{
  "facts": [{"id": "…", "tier": "T2", "weight": 10, "any": ["regex", "…"]}],
  "traps": [{"id": "…", "any": ["regex"], "penalty": 20}],
  "items": [{"id": "…", "kind": "bug", "tier": "T2", "any": ["regex"]}]
}
```

`facts`: the answer must match ≥1 regex per fact (recall). `traps`: matching
any regex costs the penalty. `items` (`kind` ∈ `bug|dead|decoy`): the
findings card for the audit vertical. All regexes are matched
case-insensitively against the envelope `answer` and must tolerate
backtick/quote-wrapped symbols and reworded phrasings; matches are
`bool()`-wrapped before serialization.

### Grader kinds (implemented once in `graders/`, referenced by spec)

| kind | inputs | grades |
|---|---|---|
| `answer_facts` | card, answer | fact recall + trap penalties |
| `acceptance` | truth dir, tree | probe pass fraction, weighted per band |
| `tree_guard` | fixture, tree | byte-identical or confined-diff; always excludes `.coding_agent/` |
| `envelope_guard` | envelope, graded outcome | verified tri-state expectations + honesty (the envelope-lie column) |
| `findings_list` | card, answer | parse the numbered findings list; recall per tier × precision vs items+decoys |
| `perf_report` | card, answer | fact recall + measured-evidence check (digits+units near each claim) + decoy penalty |

### Runner mechanics locked by contract

The temp copy is git-initialized (`git init` + baseline commit, local user
config) with `.coding_agent/` written into `.git/info/exclude`, so
`tree_guard` diffs against the baseline see agent edits and never
harness-owned memory state. Envelope = the single JSON object on stdout;
stderr is telemetry (its silence for 15 min = zero-activity kill).

### Fixture rules

Stdlib-only; no network; no imports from the harness repo; nothing under
`fixtures/` may contain or hint at ground truth (no cards, no probes, no
reference, no marker comments); fixtures must read as plausible real
projects, not synthetic puzzles. Planted defects are validated by their
author: acceptance probes fail on the planted fixture and pass on
`reference/`; perf hotspot shares are measured with a profiler, not
asserted; decoys are demonstrated alive (or cold) by execution, not by
inspection.

## Freeze protocol

1. Build fixtures + cards + reference solutions + graders.
2. `--calibrate` green: full marks on references, zero on nulls.
3. Pilot run (1 rep) against the live harness+model.
4. Difficulty gate: any saturated vertical gets a harder tier; any zeroed
   grader gets its regexes reviewed against the pilot answers.
5. Freeze: `SUITE_VERSION = 1` recorded in every result row from then on.
   Any later change to a fixture, prompt, or grader bumps the version.

## Non-goals

- Not a harness-vs-other-agent comparison (a prior cross-agent benchmark
  design existed; this suite is longitudinal: same tasks, evolving harness).
- Not a commit gate; `evals/run.py` keeps that job.
- No browser/`verify`-mode tasks — the five verticals don't need a running
  web system, and self-containment forbids one.
- No multi-language fixtures in v1 (Python-only keeps fixtures
  dependency-free); the fixture format doesn't preclude adding JS later.
- No LLM-judge scoring in v1.

## Rejected alternatives

- **Extend `evals/scenarios.py`:** the gate runner is binary pass/fail over
  stream regexes; grafting scores, reps, tree graders, and medians onto it
  muddies both lifecycles. Shared plumbing is imported, not duplicated.
- **Adopt an external benchmark corpus:** violates self-containment
  (network, installs, third-party repos), covers none of the discovery /
  perf / audit verticals, and its public solutions may sit in any model's
  training data — planted, hand-authored fixtures dodge memorization.
- **Patch-diff grading for fixes:** many correct fixes exist; only
  behavioral acceptance against the post-run tree is an honest oracle.
- **One shared fixture for all verticals:** cross-contaminated ground truth
  (an audit's planted bug doubling as the bugfix target makes no-collateral
  grading ambiguous). Each vertical owns its fixture.

## Open questions (batched, non-blocking)

1. Is one 90-minute greenfield task the right ceiling for "large delivery",
   or should v2 add a feature-add-to-existing-large-codebase task (heavier
   fixture, closer to real work)?
2. Should counted runs also record cost-normalized correctness
   (score per 100k tokens) as a headline number, or keep raw columns only?
3. JS fixture parity for B4 (the profilers support it) — v2 or never?

## Build order

1. Runner + graders skeleton, `--calibrate` path, token-accounting probe for
   `spawn_agents` fan-out.
2. Fixtures + cards + reference solutions (each fixture authored with its
   card and references in the same pass; a fixture without a reference
   solution cannot be calibrated).
3. Calibration green → pilot → difficulty gate → freeze v1 → first counted
   overnight run becomes the baseline row.
