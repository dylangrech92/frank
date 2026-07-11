# Benchmark Design — coding_agent harness vs Claude Code

Status: DESIGN ONLY — no implementation yet.
Scope: measure whether this harness's tooling (LSP nav, DAP debugging, diagnostics,
lean context discipline) buys real-world results against Claude Code, both on equal
model footing and against frontier models.

## 1. The three arms

Every arm compares the same task suite, run headlessly, one side vs the other:

| Arm | This harness | Claude Code | Question answered |
|-----|--------------|-------------|-------------------|
| A1  | Qwen3.6:27b (local ollama) | Qwen3.6:27b (same ollama, via Anthropic-API translation proxy) | Same brain, whose harness is better? |
| A2  | Qwen3.6:27b | Sonnet 5 (API) | Is local-27B + strong tooling competitive with mid-frontier + CC? |
| A3  | Qwen3.6:27b | Opus 4.8 (API) | How far is the gap to the top? |

**A1 feasibility gate (must pass before any runs count):** Claude Code speaks the
Anthropic Messages API only. Running Qwen under CC requires a translation proxy
(LiteLLM proxy or claude-code-router via `ANTHROPIC_BASE_URL`). Pre-flight: verify
tool calls round-trip correctly through the proxy on 3 trivial tasks (read a file,
edit a file, run a command). If tool-calling is mangled by translation, that is
recorded as a finding, not silently worked around — CC's incompatibility with
local models is itself a result, but a broken proxy is not.

## 2. Task suite

~12 tasks, Python only (matches the harness's pyright/LSP tooling — a stated scope
limit, not a hidden one). Four categories, three tasks each:

| Category | Shape | Primary metrics exercised |
|----------|-------|---------------------------|
| Greenfield | Build a small tool/module from a written spec with acceptance criteria | 1,2,3,4,6 |
| Feature-add | Add a feature to an existing ~2–5 kLOC fixture repo with a test suite | 1,2,3,4,6 |
| Bugfix | Seeded defect with a reported symptom; ground-truth cause written down beforehand | 1,2,5,6 |
| Refactor | Behaviour-preserving change under constraint ("tests must stay green") | 1,2,3,4,6 |

Task authoring rules:
- Each task ships with: a **prompt file** (identical text for both sides), a
  **fixture repo snapshot** (pinned commit, vendored), and a **hidden acceptance
  suite** never visible to the agent (kept outside the working tree, applied after
  the run).
- Bugfix tasks: defects are seeded by hand (off-by-one, wrong invariant, race,
  swallowed error), and a ground-truth card records the root cause, the correct
  fix location, and a hidden regression test. The symptom description given to
  the agent mimics a real bug report — no line numbers, no file names.
- Prompts state acceptance criteria explicitly ("done when X runs and Y passes")
  so neither side is penalised for guessing scope.

## 3. Metrics — definitions and measurement source

### M1. Tokens used
- Harness: sum of `prompt_tokens` / `completion_tokens` from `llm.py` usage
  telemetry across every call in the run.
- Claude Code: `claude -p ... --output-format json` reports `usage` (including
  cache-read tokens) and cost per session.
- Report **four columns**: prompt tokens, completion tokens, cache-read tokens
  (CC only; harness has no server-side cache), API call count.
- Cross-model caveat (A2/A3): different tokenizers make raw counts non-comparable;
  for those arms the headline is **$ cost per completed task** (API list price vs
  $0 local, with local electricity/time captured by M2) and call count.
- A1 is the clean comparison: both sides' counts come from the same ollama
  `usage` fields.

### M2. Time to completion
- Wall clock from process spawn to process exit of the headless run.
- Known confound in A2/A3 (local 27B on this Mac vs Anthropic API latency) —
  reported, not corrected. A1 is hardware-fair.
- Controls: same machine, nothing else running, one benchmark run at a time,
  ollama `keep_alive -1` and one warm-up call before the timed run so model
  load time isn't billed to the first task.
- Timeout policy: hard cap 30 min per run; kill early only after 10 min of zero
  activity (no output, no file writes, no network) — a killed run scores DNF.

### M3. Code quality — did it work first time?
Graded 0–3 on a **fresh checkout of the delivered diff**, zero human edits:
- 0 = does not import / syntax errors
- 1 = imports and lints clean (ruff + pyright), but visible tests fail
- 2 = the repo's own visible test suite passes
- 3 = hidden acceptance suite also passes ("works as intended")
Binary "first-run pass" = score 3. No partial credit for "almost".

### M4. LOC delivered
- `git diff --shortstat` on the delivered work: **net LOC** (insertions −
  deletions) plus files touched.
- Only scored when M3 = 3; a small diff that doesn't work is N/A, not a win.
  Less is better — Law 2, net-negative LOC is the success signal on refactors.

### M5. Bugfix correctness
Scored against the ground-truth card, 0–3:
- +1 **diagnosis**: the agent's final summary/commit message names the actual
  root cause (not just the symptom).
- +1 **fix**: hidden regression test passes AND the change lands at/upstream of
  the true cause (a symptom-patch that greens the test scores 0 here — judged
  against the card, by hand).
- +1 **no collateral**: full hidden suite still green, no unrelated files touched.

### M6. Taste — senior developer score
- **Blinded pairwise review**: for each task, the two diffs (+ final summaries)
  are anonymised (strip tool signatures, commit trailers, model names),
  labelled A/B in randomised order.
- Primary judge: Dylan, scoring each diff 1–10 on a fixed rubric — naming &
  idioms, minimality, cohesion/placement, error-handling honesty, test quality —
  plus a forced A/B preference.
- Secondary: a 3-model LLM judge panel with the same rubric. Bias caveat is
  explicit: any Anthropic judge model favours familiar output style, and in
  A2/A3 the judge family is also a participant. LLM panel is reported as a
  secondary signal only and never overrides the human score.

## 4. Run protocol

Per (task × arm × side), **N = 3 repetitions** (agents are nondeterministic;
report median, show min–max range):

1. Copy fixture repo to a fresh temp worktree; `git init`/pin so the diff is clean.
2. Launch headless:
   - Harness: `python3 main.py -p "$(cat prompt.md)"` with repo
     `config.json` pinned to Qwen3.6:27b.
   - Claude Code: `claude -p "$(cat prompt.md)" --output-format json
     --dangerously-skip-permissions` in a sandboxed HOME — **no CLAUDE.md, no
     memory, no MCP servers, default toolset** — model pinned per arm.
3. Capture: full transcript, exit code, wall time, token JSON, `git diff`.
4. Post-run scoring: apply hidden suite in a fresh venv, compute M1–M5
   mechanically; queue the diff pair for M6 blinded review.
5. Everything lands in `evals/results/bench-<timestamp>/<arm>/<task>/<side>-<rep>/`
   as raw artifacts + one `metrics.json` per run.

Fairness controls (checklist, verified before the first counted run):
- Identical prompt bytes both sides.
- Both sides fully autonomous — no permission prompts (harness auto-allow config;
  CC skip-permissions in a throwaway sandbox dir).
- No memory / retrieval on either side.
- Tool parity is **not** equalised — the harness's LSP/DAP tools and CC's tools
  are each side's identity; that difference is the thing under test.
- Same machine, serial execution, warm model.

## 5. Aggregation & reporting

No single composite score — a six-metric scorecard per arm, tasks as rows,
median-of-3 as the cell value. Headline numbers per arm:

- **Pass rate** (M3 = 3, or M5 ≥ 2 for bugfix tasks) — the gate everything else
  hangs off.
- **Tokens per solved task** and (A2/A3) **$ per solved task**.
- **Median wall time per solved task**.
- **Median net LOC per solved task**.
- **Taste win rate** (blinded A/B preferences) + mean rubric score.

DNFs and 0-scores stay in the table — dropping failures is how benchmarks lie.

## 6. Threats to validity (acknowledged up front)

- **Proxy fidelity (A1):** the translation layer may degrade CC+Qwen tool
  calling. Mitigated by the pre-flight gate; residual degradation is disclosed.
- **Tokenizer mismatch (A2/A3):** raw token counts across model families are
  apples-to-oranges; $ cost and call count carry those arms.
- **Hardware confound (M2, A2/A3):** local GPU vs API latency — disclosed, not
  corrected.
- **Prompt caching:** CC gets server-side cache reads; reported in their own
  column so they can't hide inside "prompt tokens".
- **Judge bias (M6):** human primary, blinded, randomised order; LLM panel
  secondary with the conflict of interest stated.
- **Small N:** 3 reps × 3 tasks per category bounds noise but won't reach
  significance on close calls — report ranges, don't over-claim.
- **Task authorship bias:** tasks are written by the harness's author; mitigate
  by fixing the task set and acceptance suites **before** the first run and
  never editing them after (any task change resets all results for that task).

## 7. Build order (when implementation is green-lit)

1. A1 pre-flight proxy gate (cheapest kill-switch for the whole design).
2. One task per category + the runner/metrics plumbing (extends `evals/run.py`
   patterns: temp dir, transcript capture, but new — external CC invocation,
   metrics.json, hidden-suite scoring).
3. Pilot: full protocol on those 4 tasks, A1 only → fix protocol bugs.
4. Author remaining 8 tasks, freeze the suite.
5. Full A1 → A2 → A3 runs, then blinded M6 review, then the report.
