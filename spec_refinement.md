# Spec refinement — dependable-senior-coder pass (H-series)

Five refinements identified from the 2026-07-06 live-test sessions (P17 + slice G).
Design philosophy carried over from P17: reactive, evidence-gated mechanisms that
fire only when a smell appears — never standing per-turn instructions.

## H1 — Post-mutation verification nudge

**Gap:** the agent edits files, receives injected LSP diagnostics, and declares done
without running anything. A senior runs the tests.

**Mechanism:** in `agent.py`, when a turn is about to end (model returned no tool
calls) and `_TURN_MUTATIONS`-tracked file mutations occurred this turn without any
subsequent `run_tests`/`run_command` call, inject one system-side line before
accepting the final answer: the agent must either verify or explicitly state the
change is unverified. Fires at most once per turn.

**Done when:** a live session that edits a file and answers immediately receives the
nudge and either runs the tests or says "unverified"; a turn that already ran tests
gets no nudge.

## H2 — Stale-read edit guard

**Gap:** multiple agents share one directory (by design), but write tools happily
overwrite files modified on disk after this session last read them.

**Mechanism:** track per-session `{path: mtime/hash}` at `read_file` time; on
`replace_one`/`replace_many`/`update_file`/`delete_file`, if the file changed on disk
since the last read (or was never read), return `file-changed-on-disk` /
`not-read-yet` error with a re-read hint instead of writing.

**Done when:** an offline probe that reads, externally modifies, then edits gets the
error; re-reading clears it; `create_file` (new files) is unaffected.

## H3 — State-assumptions prompt rule

**Gap:** working rules cover evidence, retries, and scope — not ambiguity. Silent
interpretation picks are invisible until they are wrong.

**Mechanism:** one line in `SYSTEM_PROMPT`'s working rules: when the request is
ambiguous, state the assumption being proceeded on in the answer.

**Done when:** an ambiguous live prompt yields an explicit "assuming X" in the reply.

## H4 — Repeatable eval harness

**Gap:** dependability claims rest on hand-rolled one-off scripts; prompt/guard
changes have no regression score.

**Mechanism:** `evals/` directory with a stdlib-only runner that replays the proven
scenarios against the live endpoint (config-driven): load_tool gate, JSON leniency,
oversize guard (central via run_command + read_file paging), loop-guard steer
(dispatch-level), compaction coherence, scope decline, search-focus nudge. Each
scenario = prompt(s) + pass-markers grepped from transcript/stderr. Output: pass/fail
table.

**Done when:** `python3 evals/run.py` executes all scenarios against the configured
endpoint and prints a scored table matching today's known-good results.

## H5 — Graph-memory usage nudge

**Gap:** P14's decision/spec graph layer is injected into context but rarely written
to — nothing prompts the model to record decisions during long tasks, so
cross-session resumes rely on transcript summaries alone.

**Mechanism:** reactive trigger in `agent.py`: after a turn whose mutations span 3+
files and no `record_decision`/`record_spec` call happened this turn, append a hint
to the final tool result suggesting recording the driving decision. At most once per
session.

**Done when:** a multi-file-edit live session surfaces the hint exactly once and a
single-file edit never does.

## Sequencing

H2 + H4 first (touch tools/evals only). H1 + H3 + H5 after the slice-G commit lands
(they edit `agent.py`/`main.py`). Every slice: offline probe first, then live
verification on the standard endpoint, then commit.
