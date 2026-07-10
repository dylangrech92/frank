# Eval harness

A repeatable regression harness that replays a fixed set of scenarios against
a real agent session so prompt, tool, and guard changes get a pass/fail score
instead of relying on one-off manual checks.

## Running

```
python3 evals/run.py                  # run every scenario
python3 evals/run.py --list           # list scenario names + descriptions
python3 evals/run.py --only oversize  # run scenarios whose name contains "oversize"
python3 evals/run.py --timeout 600    # override the per-scenario timeout (default 420s)
```

Each run drives a real `main.py` session against the endpoint configured in
`config.json` (or a scenario's config overrides, deep-merged on top of it),
so a full run takes real minutes, not seconds — every scenario spawns a
fresh temporary project directory, writes any fixture files it declares,
pipes its prompt turns into the agent's stdin, and captures stdout/stderr.

Exit code is `0` iff every scenario passes, `1` otherwise. A final pass/fail
table is printed, plus the path to the raw transcripts for that run.

`evals/results/` holds one timestamped subdirectory per run
(`<scenario>.out.txt` / `<scenario>.err.txt`); it is git-ignored, since
transcripts are run artifacts, not source.

### Offline smoke test

`python3 evals/run.py --smoke` swaps the real agent invocation for a canned
stub subprocess, so the check-evaluation and table-rendering plumbing can be
verified without hitting any live endpoint. It is a plumbing check only —
it does not validate real agent behavior.

## Adding a scenario

Scenarios are declared in `evals/scenarios.py` as plain dicts appended to
`SCENARIOS`. A scenario needs:

- `name` / `description` — unique id and one-line summary.
- `turns` — a list of single-line prompt strings, sent one per line to the
  agent's stdin in order.
- `config` — `"default"` to use the repo's `config.json` unmodified, or a
  dict that gets deep-merged on top of it (e.g. to shrink `llm.context_limit`
  so compaction or oversize-guard paths trigger reliably).
- `setup` — a dict of relative-path -> file content to write into the fresh
  temp project directory before the run. The special key `bigfile_bytes: N`
  generates a ~N-byte fixture file named `bigfile.py` instead of literal
  content.
- `checks` — a list of assertions evaluated against captured stdout/stderr
  (and, for `file-lines-min`, files written into the temp project dir):
  - `{'stream': ..., 'kind': 'regex-present', 'pattern': ...}` — must match.
  - `{'stream': ..., 'kind': 'regex-absent', 'pattern': ...}` — must not match.
  - `{'stream': ..., 'kind': 'ordered', 'patterns': [...]}` — each pattern
    must be found in order, later patterns searched only after the previous
    match.
  - `{'stream': ..., 'kind': 'regex-note', 'pattern': ...}` — informational
    only, reported in the table but never fails the scenario.
  - `{'kind': 'file-lines-min', 'path': ..., 'min': N}` — the file must exist
    (relative to the temp project dir) with at least N non-blank lines.

A scenario can instead set `inline: '<script>.py'` (a script under `evals/`)
to run a deterministic check with no network involved at all — the script is
run directly and must exit `0` to pass. Two flavours:

- **Dispatch-level** checks poke a single subsystem directly (no LLM at all).
  This is how loop-guard steering is covered, since a cooperative model will
  not reliably repeat an identical failing call twice in a live session.
- **End-to-end** checks drive the real turn loop (`agent.handle_user_message`)
  with a scripted stub LLM (see `evals/_stub.py`), so a whole guarantee is
  exercised offline and deterministically. This is how the escalation ladder
  and the compaction ladder are covered — a live model's verbosity (and thus
  whether compaction even triggers) is far too variable to gate on.
