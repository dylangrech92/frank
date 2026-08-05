"""B2 -- Project delivery.

One greenfield, no-fixture task (90-minute cap): build a dependency-aware
job scheduler CLI from a single frozen, multi-part spec. Eleven numbered
requirements, several deliberately interacting -- crash-safe persistence
constrains how the simulated-clock runner must be structured, dependency
ordering interacts with priority preemption within a tick, cancellation
retroactively revokes a dependency's completed-ness for its dependents,
an execution budget composes with both cancellation and dependency
eligibility under a strict three-way precedence, and the uniform
exit-code contract applies to every subcommand including the ones
documented last. Difficulty axis is multi-part spec adherence under
length plus feature interaction: nothing here is individually hard, but
satisfying requirement N naively tends to violate requirement M, and the
cancel/budget/dependency precedence has no correct implementation that
falls out of any single requirement read in isolation.

Hidden acceptance suite: ~53 behavioral probes across three bands --
core (each requirement's happy path, including cancel and
--max-concurrent), edge (empty/whitespace schedules, unicode job ids,
boundary --until values, malformed schedule lines, corrupt-state and
schedule-hash-mismatch recovery -- now also exercised through `cancel`),
adversarial (exact error-message/exit-code contracts, permanent- vs
transient-dependency blocking, byte-identical dry-run and idempotent
re-run, retroactive dependency revocation on cancellation, the exact
three-way skip-reason precedence, self-dependency and duplicate-id
rejection, the trailing documentation sub-tasks that historically get
dropped). Weighted core/edge/adversarial 30/30/40 in this task's own
graders -- adversarial and edge together must carry enough mass that
acing core alone cannot reach a passing acceptance score, tightened from
the 50/30/20 split DESIGN.md describes as the vertical's general shape.
"""

TASKS = [
    {
        "id": "b2_t3_job_scheduler_cli",
        "vertical": "B2",
        "tier": "T3",
        "mode": "code",
        "fixture": None,
        "prompt": (
            """Build a command-line job scheduler. Deliver it as a single executable Python 3
script named exactly `jobsched.py`, placed at the root of the project
directory. Use only the Python 3 standard library — no pip installs, no
third-party packages, no network access. Invoke it as `python3 jobsched.py
<subcommand> ...`; do not split it into a package or add an install step.

`jobsched` never reads the real system clock. All timing is a simulated
integer "tick" counter supplied by the caller — this keeps every run fully
deterministic given the same schedule file and flags.

There are five subcommands: `validate`, `run`, `status`, `history`, `cancel`.

## 1. Schedule file format and parsing

A schedule file is a UTF-8 text file. Each line is either:
- blank (whitespace-only) — ignored,
- a comment — first non-whitespace character is `#` — ignored,
- a job definition with exactly 5 fields separated by the `|` character:

```
job_id | priority | interval | depends_on | command
```

Parse each job line by splitting on `|` into at most 5 pieces: split on the
first 4 `|` characters only, so the 5th field (`command`) is everything
remaining on the line and may itself contain `|` characters. Trim leading and
trailing whitespace from each of the first 4 fields before validating them;
do not trim `command`'s internal content beyond stripping one leading and one
trailing whitespace character run at its edges the same way the other fields
are trimmed.

Field rules, validated in this order for each job line:
1. The line must split into exactly 5 fields this way. Otherwise: parse
   error `expected 5 fields separated by '|', got N`.
2. `job_id` must match `^[A-Za-z0-9_-]+$` (ASCII only). Otherwise: parse
   error `invalid job id '<value>' (must match [A-Za-z0-9_-]+)`.
3. `priority` must parse as a base-10 integer (may be negative or zero).
   Otherwise: parse error `priority must be an integer`.
4. `interval` must parse as a base-10 integer that is `>= 1`. Otherwise:
   parse error `interval must be a positive integer`.
5. `depends_on` is either the single character `-` (meaning no
   dependencies) or a comma-separated list of job ids, each trimmed of
   surrounding whitespace, each non-empty. A malformed list (e.g. an empty
   entry between two commas) is: parse error `invalid depends_on list`.

A parse error aborts validation/run with: `error: line <L>: <reason>` on
stderr, exit code 2, where `<L>` is the 1-indexed physical line number in the
file (blank and comment lines still count toward the line number).

After every line parses individually, two whole-file checks run, in this
order:
6. Duplicate `job_id`: scanning top-down, the first line whose `job_id`
   repeats an earlier line's `job_id` is reported using that later line's
   number: `error: line <L>: duplicate job id '<value>'`, exit 2.
7. Every id named in any `depends_on` list must be a `job_id` defined
   somewhere in the file. The first such reference found (scanning jobs
   top-down, each job's `depends_on` list left-to-right) that names an
   undefined id is reported as a DAG error (not a parse error — see
   Section 2), not tied to a specific line number.

An empty schedule file (zero job lines after ignoring blank/comment lines)
is valid: zero jobs, zero edges.

## 2. Dependency graph validation (DAG)

After all lines parse and duplicate/undefined-reference checks (Section 1,
rules 6–7) pass, the dependency graph must be acyclic. Build a directed edge
from each job to each id in its `depends_on` list.

- Undefined dependency reference: `error: dag: job '<id>' depends on unknown
  job '<dep>'` on stderr, exit code 3.
- Cycle: find any cycle; report it starting from the lexicographically
  smallest job id that participates in that cycle, following `depends_on`
  edges (job -> dependency) around the cycle exactly once back to the
  starting id: `error: dag: cycle detected: <id1> -> <id2> -> ... -> <id1>`
  on stderr, exit code 3.

`validate <schedule_file>` runs exactly these checks (Sections 1–2) and
nothing else. On success it prints `OK: <n> jobs, <m> edges` to stdout (n =
job count, m = total number of dependency edges, i.e. the sum over all jobs
of the length of their `depends_on` list) and exits 0. If the schedule file
does not exist or cannot be read, print `error: schedule file not found:
<path>` to stderr and exit 1. Any unexpected internal exception anywhere in
`jobsched` (all five subcommands) must be caught, printed as `internal
error: <message>` on stderr, and exit with code 70 — never let a raw
traceback determine the process exit code.

## 3. The simulated-clock `run` command

```
jobsched.py run <schedule_file> --until N [--dry-run]
                 [--state-file PATH] [--log-dir PATH] [--force-recover]
                 [--max-concurrent K]
```

`--until N` is required; `N` must parse as an integer `>= 1`, otherwise:
`error: --until requires a positive integer` on stderr, exit code 6 (see
Section 9 for the exit-code contract and why this must not collide with
exit 2). `--max-concurrent K` is optional; when given, `K` must parse as an
integer `>= 1`, otherwise: `error: --max-concurrent requires a positive
integer` on stderr, exit code 6. When omitted, the number of jobs that may
execute in a tick is unlimited (the original, pre-`--max-concurrent`
behavior). `--state-file` defaults to `.jobsched/state.json`; `--log-dir`
defaults to `.jobsched/logs` (so the default log file is
`.jobsched/logs/run.log`). Both are resolved relative to the current working
directory when not absolute.

`run` first parses and validates the schedule exactly as `validate` does
(Sections 1–2), using the same exit codes and messages on failure. On a
schedule error, `run` must not create, modify, or touch `--state-file`,
`--log-dir`, or any file under them in any way.

Ticks are simulated as the integer range `0` to `N-1` inclusive. A job with
`interval` I is "due" at tick `t` when `t % I == 0`.

State loading, before simulating any tick:
- No state file at `--state-file`: start fresh (no prior completed tick, no
  prior job history).
- State file present but not parseable as JSON, or missing required keys:
  without `--force-recover`, print `error: state file corrupt: <path> (use
  --force-recover to reset)` on stderr, exit code 4. With `--force-recover`,
  discard it and start fresh, as if no state file existed.
- State file present and parseable: compute the SHA-256 hex digest of the
  current schedule file's exact bytes and compare it to the state's stored
  `schedule_hash`. On mismatch, without `--force-recover`: print `error:
  schedule has changed since last run (use --force-recover to reset state)`
  on stderr, exit code 5. With `--force-recover`: discard the state and
  start fresh (the fresh state will record the new hash).

Determining the starting tick:
```
if state.in_progress_tick is not None:
    start = state.in_progress_tick
elif state.last_completed_tick is not None:
    start = state.last_completed_tick + 1
else:
    start = 0
if start > N - 1:
    print "already complete: last_completed_tick=<value-or-'none'>"
    exit 0   # no state, log, or directory touched
```

Otherwise, for each tick `t` from `start` to `N - 1` in order:
1. If not `--dry-run`: persist state (see Section 5 for the atomic-write
   rule) with `in_progress_tick = t` before executing anything in this tick.
2. Compute the due jobs at tick `t` (`t % interval == 0`). Order them by
   `priority` descending; break ties by `job_id` ascending (lexicographic).
3. Walk the ordered due jobs one at a time, classifying each with the
   *first* matching rule below. A job may match more than one rule; only
   the first match is ever recorded, and this precedence is exact and does
   not vary by any other condition:
   a. **Cancelled** (Section 10): if the job has been cancelled, record it
      as `skipped`, reason `cancelled`, and move on to the next due job
      without evaluating eligibility or budget at all.
   b. **Dependency-ineligible**: a due job is eligible to run only if every
      id in its `depends_on` list currently has status `completed` (from an
      earlier tick, or from earlier in this same tick's walk — dependencies
      satisfied earlier in the same tick's ordered walk do count) *and* is
      not itself cancelled (a cancelled job's earlier `completed` status
      never counts as satisfying a dependent — see Section 10). If not
      eligible: record it as `skipped`, reason `waiting-on-dependency`.
   c. **Execution budget exhausted**: only reachable when `--max-concurrent
      K` was given. Track `budget_used`, the count of jobs that have
      actually executed so far in this tick, starting at 0 at the start of
      every tick and incremented by one per executed job regardless of
      whether it completes or fails (Section 4). If `budget_used >= K`:
      record the job as `skipped`, reason `budget-exceeded`.
   d. **Execute** it (Section 4) — the job matched none of the rules above.
   A job skipped for `waiting-on-dependency` or `budget-exceeded` is
   transient and is retried at its own next due tick, exactly as before
   this section's rules a/c existed. A job skipped for `cancelled` is never
   retried (Section 10).
   - `--dry-run` performs this identical a/b/c/d classification against an
     in-memory scratch copy of state seeded from what was loaded at
     startup — never against the real on-disk state — and never spawns any
     subprocess. Within the dry-run scratch copy only, a job that "would
     run" is treated as `completed` for the purpose of evaluating later
     ticks' dependency eligibility, and counts toward that tick's budget,
     within the same dry-run invocation, so multi-tick dry-run plans stay
     coherent. Nothing computed during `--dry-run` is written anywhere.
4. If not `--dry-run`: after every due job in tick `t` is resolved, append
   one log line per resolved due job, in the same order they were resolved,
   to the run log (Section 6), then persist state (Section 5) with
   `last_completed_tick = t` and `in_progress_tick = null`.
5. Print one line to stdout for this tick:
   - real run: `tick <t>: <c> completed, <f> failed, <s> skipped` (counts
     among this tick's due jobs only; `s` sums all three skip reasons —
     `cancelled`, `waiting-on-dependency`, and `budget-exceeded` alike).
   - dry-run: `tick <t> (dry-run): <r> would-run, <s> would-skip` (`r` =
     due jobs that would run, `s` = due jobs that would be skipped for any
     of the three reasons above; non-due jobs are not mentioned).

After the loop, print `done: <N> ticks simulated` (real run) or `done: <N>
ticks planned (dry-run, no changes made)` (dry-run) and exit 0.

A job's own failure (Section 4) is never a `jobsched` scheduling error: as
long as `run` reaches the end of its requested ticks without a schedule,
state, or usage error, it exits 0 — regardless of how many individual job
commands failed. `run`'s exit code reports whether the scheduler itself
functioned, not whether every job succeeded; use `status` (Section 7) to
inspect individual job outcomes.

## 4. Executing a job

Execute `command` with `subprocess.run(command, shell=True)`, inheriting
`jobsched`'s own working directory. Discard the job command's stdout and
stderr entirely — `jobsched` never forwards or logs a job's own output, only
its outcome. Status is `completed` if the subprocess exit code is 0,
otherwise `failed` (both count as "ran"). Record: `run_count` incremented by
1, `last_run_tick = t`, `last_status` set to `completed` or `failed`,
`last_exit_code` set to the subprocess's exit code. A job that has never
been executed (never due-and-eligible) has no entry recorded for it at all.
`skipped` outcomes never update a job's recorded `run_count`,
`last_run_tick`, `last_status`, or `last_exit_code` — those fields only
change on an actual execution attempt.

A job whose most recent execution attempt was `failed` does not satisfy any
dependent's eligibility check (Section 3, step 3) — dependents stay
`skipped` at every future tick until that job is attempted again (at its own
next due tick) and succeeds.

## 5. Crash-safe state persistence

State is one JSON object, UTF-8 encoded:

```json
{
  "version": 1,
  "schedule_hash": "<sha256 hex digest of the schedule file's bytes>",
  "last_completed_tick": null,
  "in_progress_tick": null,
  "jobs": {
    "<job_id>": {
      "run_count": 0,
      "last_run_tick": 0,
      "last_status": "completed",
      "last_exit_code": 0
    }
  }
}
```

`last_completed_tick` is `null` until at least one tick has fully completed,
then the highest completed tick number. `jobs` only contains keys for job
ids that have been executed at least once (Section 4).

Every write to `--state-file` is atomic: write the full JSON document to a
temp file in the same directory as `--state-file`, then replace the target
path with it in one filesystem operation (e.g. `os.replace`) — never leave a
half-written state file on disk. Create `--state-file`'s parent directory
(and `--log-dir`) lazily, only at the moment a real (non-dry-run) `run`
first needs to write to them. `validate`, `status`, `history`, and any
`--dry-run` invocation of `run` must never create `--state-file`,
`--log-dir`, or their parent directories.

If a previous `run` process was killed mid-tick, its last successful state
write left `in_progress_tick` set to that tick's number (written in step 1
of Section 3, before execution) without a matching follow-up write clearing
it (step 4). The next `run` invocation detects this (`in_progress_tick is
not None`, per the starting-tick algorithm in Section 3) and re-attempts
that entire tick from scratch. Because a crash may have happened after a
job's command already ran, `jobsched` guarantees exactly-once bookkeeping
per tick that completes cleanly, and at-least-once re-attempt of a tick that
was interrupted — job commands are expected to tolerate being re-invoked
after a crash. This is a documented, intentional trade-off, not a bug.

## 6. Run log and log rotation

Real (non-dry-run) `run` invocations append to a run log at
`<log-dir>/run.log`, one line per resolved due job, in the order resolved,
appended once per tick as part of that tick's step 4 (Section 3). Each line
is 5 tab-separated fields terminated by `\\n`:

```
<tick>\\t<job_id>\\t<status>\\t<exit_code>\\t<reason>\\n
```

- `status` is `completed`, `failed`, or `skipped`.
- `exit_code` is the subprocess's integer exit code for `completed`/`failed`
  lines, or the single character `-` for `skipped` lines.
- `reason` is the single character `-` for `completed`/`failed` lines, or
  one of `cancelled`, `waiting-on-dependency`, `budget-exceeded` for
  `skipped` lines — whichever precedence rule (Section 3, step 3) produced
  that skip.

After appending a tick's lines, if `<log-dir>/run.log` now exceeds 10000
bytes, rotate before persisting state (Section 5) for that tick: delete
`run.log.5` if present, then rename `run.log.4` -> `run.log.5`, `run.log.3`
-> `run.log.4`, `run.log.2` -> `run.log.3`, `run.log.1` -> `run.log.2`,
`run.log` -> `run.log.1` (each rename only if the source exists), then start
a new empty `run.log`. At most 5 rotated files (`run.log.1` .. `run.log.5`)
are ever kept; older segments are deleted, never merged.

## 7. Query commands

```
jobsched.py status [--state-file PATH]
jobsched.py history [--log-dir PATH] [--limit N]
```

`status` reads only `--state-file`; `history` reads only `--log-dir`. Each
defaults its path the same way `run` does (`.jobsched/state.json` /
`.jobsched/logs`). Neither command writes anything, ever, and neither
accepts `--force-recover`.

`status`: if `--state-file` does not exist, print `no state` and exit 0. If
it exists but is not valid state JSON, print `error: state file corrupt:
<path> (use --force-recover to reset)` on stderr and exit 4 (there is no
recovery from a read-only command; re-run `run --force-recover` or `cancel
--force-recover` to fix it). Otherwise print `last_completed_tick: <value>`
(the tick number, or the literal `none` if nothing has completed yet), then
one line per job present in the state's `jobs` map, sorted by `job_id`
ascending: `<job_id>: status=<display_status> run_count=<run_count>
last_exit_code=<display_exit_code>`, where:

- `display_status` is the literal `cancelled` if the job has been cancelled
  (Section 10), regardless of whatever `last_status` it carries from before
  cancellation (or the absence of one, if it was cancelled before ever
  executing); otherwise it is `last_status` (`completed` or `failed`)
  unchanged.
- `display_exit_code` is the literal `none` if the job has no recorded
  `last_exit_code` (never executed); otherwise the integer `last_exit_code`
  unchanged, following the same `none`-for-absent convention already used
  for `last_completed_tick`.

A job appears in `jobs` (and therefore in this output) once it has either
executed at least once (Section 4) or been cancelled at least once
(Section 10) — whichever happens first. A job that has done neither is not
listed.

`history`: `--limit` defaults to 20 and must parse as an integer `>= 1`,
otherwise `error: --limit requires a positive integer` on stderr, exit 6. If
neither `run.log` nor any `run.log.1`..`run.log.5` exists under `--log-dir`,
print `no history` and exit 0. Otherwise, read log lines newest-first: all
lines of `run.log` in reverse order, then all lines of `run.log.1` in
reverse order, then `run.log.2`, and so on through `run.log.5` (only
existing files are read); print up to `--limit` of them in that
newest-first order, one per output line, reformatted as:
`tick=<tick> job=<job_id> status=<status> exit_code=<exit_code|-> reason=<reason|->`
(the same field values as the raw log line, `-` preserved as-is where the
raw line had it). Exit 0.

## 8. `--force-recover`

`run` and `cancel` (Section 10) both accept `--force-recover`; no other
subcommand does. It has no effect unless the state file is corrupt
(Section 3) or its `schedule_hash` no longer matches (Section 3) — in both
cases it makes the subcommand discard the existing state and proceed as if
starting fresh (a newly computed `schedule_hash`, and for `run`, starting
from tick 0), instead of exiting 4 or 5. For `cancel`, a force-recovered
fresh state still receives the cancellation being requested in the same
invocation — `cancel --force-recover` never exits 4 or 5 on account of the
very state it is about to discard. Passing `--force-recover` when the state
is already valid and current has no effect at all, for either subcommand.

## 9. Exit-code contract (every subcommand)

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | schedule file not found or unreadable (`validate`, `run`, `cancel`) |
| 2 | schedule parse error (Section 1) |
| 3 | DAG error: unknown dependency or cycle (Section 2) |
| 4 | state file corrupt and `--force-recover` not given (`run`, `status`, `cancel`) |
| 5 | schedule/state mismatch and `--force-recover` not given (`run`, `cancel`) |
| 6 | CLI usage error: missing/invalid required argument, unknown flag, unknown or missing subcommand, or (`cancel` only) a job id not defined in the schedule file |
| 70 | unexpected internal error (always caught; never a bare traceback) |

This table applies uniformly to `validate`, `run`, `status`, `history`, and
`cancel`, with no exceptions and no subcommand-specific carve-outs. In
particular: argument-parsing libraries that default to exit code 2 for
usage problems (e.g. a missing required positional argument) must be
overridden or intercepted so that ALL usage problems — on every subcommand,
including `validate` — surface as exit code 6, never colliding with exit
code 2 (reserved exclusively for schedule parse errors). A due job's
command exiting non-zero during `run` is not a usage, schedule, state, or
internal error and never changes `run`'s own exit code (Section 3).

## 10. The `cancel` subcommand

```
jobsched.py cancel <schedule_file> <job_id> [--state-file PATH] [--force-recover]
```

`cancel` marks `job_id` as permanently cancelled in persisted state.
`<schedule_file>` is parsed and validated exactly as `validate` does
(Sections 1–2), using the same exit codes and messages on failure — this is
also how an unknown `job_id` is diagnosed: it must be a job defined in
`<schedule_file>`, regardless of whether it has ever appeared in
`--state-file`. If `job_id` is not a job defined in the (successfully
parsed) schedule, print `error: cancel: unknown job id '<job_id>'` on
stderr and exit 6.

State is loaded and, when necessary, force-recovered using exactly the
rules `run` uses (Section 3's "State loading" list): a missing state file
starts fresh; corrupt state without `--force-recover` exits 4 with the same
message `run` uses; corrupt state with `--force-recover` discards it;
a schedule/state hash mismatch without `--force-recover` exits 5 with the
same message `run` uses; with `--force-recover` it discards the state.
`cancel` never inspects or changes `last_completed_tick` or
`in_progress_tick`, and never touches any job's entry besides `job_id`'s.

Once state is loaded (or freshly created), record the cancellation: if
`job_id` already has an entry in state's `jobs` map, set its `cancelled`
field to `true`, leaving `run_count`, `last_run_tick`, `last_status`, and
`last_exit_code` exactly as they were. If `job_id` has no entry yet (it has
never been executed), create one with `run_count: 0`, `last_run_tick:
null`, `last_status: null`, `last_exit_code: null`, `cancelled: true`.
Persist state atomically (Section 5's atomic-write rule). Cancelling an
already-cancelled job is not an error: it succeeds again, idempotently,
re-writing the same `cancelled: true` state. Print `cancelled: <job_id>` to
stdout and exit 0. `cancel` never touches `--log-dir` or the run log, and
never creates `--log-dir` or its parent directory.

Effect on `run` (real or `--dry-run`), from the moment cancellation is
persisted onward:

- **Direct effect.** At every tick where a cancelled job is due, it is
  recorded `skipped`, reason `cancelled` — checked before dependency
  eligibility and before the execution budget (Section 3, step 3), so a
  cancelled job is reported `cancelled` even if it also has unmet
  dependencies or the tick's budget is already exhausted. `cancelled`
  skips never update the job's `run_count`, `last_run_tick`,
  `last_status`, or `last_exit_code`.
- **Retroactive effect on dependents.** A cancelled job's earlier
  `completed` status (if any) never again satisfies a dependent's
  eligibility check, from the moment cancellation is persisted onward. This
  re-evaluation happens at each dependent's own next due tick after
  cancellation — a dependent that already ran successfully in an
  already-completed earlier tick is not undone; only future eligibility
  checks are affected. A job cancelled with no `last_status` at all (never
  executed) likewise can never satisfy a dependent's eligibility.
- Cancelling a job has no effect on the jobs *it* depends on — cancellation
  never cascades to a cancelled job's own dependencies.

`status` (Section 7) displays a cancelled job's `status` as the literal
`cancelled`, superseding whatever `last_status` it carries, and its
`last_exit_code` as the literal `none` if it was cancelled before it ever
executed.

## 11. Documentation

Deliver `README.md` at the project root, documenting: the exact CLI name
and invocation form (`python3 jobsched.py <subcommand> ...`); all five
subcommands (including `cancel`) with every flag they accept (including
`run --max-concurrent`) and each flag's default; the schedule file format
(Section 1) including a worked example schedule with at least 3 jobs and at
least one dependency; the complete exit-code table from Section 9
reproduced in full (all 8 rows); the three skip reasons a `run` may record
(`cancelled`, `waiting-on-dependency`, `budget-exceeded`) and the
precedence between them (Section 3); and where runtime state and logs are
written by default.

Deliver `CHANGELOG.md` at the project root with at least one dated or
versioned entry describing the initial release and listing the major
capabilities implemented (schedule parsing, DAG validation, simulated-clock
run, crash-safe persistence, log rotation, dry-run, query commands, job
cancellation, execution budgeting, exit-code contract).

## Deliverable files

The only files you may create or modify are: `jobsched.py`, `README.md`,
`CHANGELOG.md`, and optionally `.gitignore`. Runtime state and logs
(everything under `.jobsched/`, wherever `--state-file`/`--log-dir` point by
default or by flag) are the only files `jobsched.py` itself is allowed to
create when it runs, and only ever under a path you chose via
`--state-file`/`--log-dir` or their `.jobsched/` defaults — never at the
project root and never anywhere else. Before you finish, make sure your
working directory contains nothing beyond `jobsched.py`, `README.md`,
`CHANGELOG.md`, an optional `.gitignore`, and — only if you leave it in
place instead of gitignoring or removing it — a `.jobsched/` directory
holding only files this same spec describes. Leave no other trace of your
own testing (temporary schedule files, `__pycache__/`, stray state or log
files outside `.jobsched/`, editor backups, etc.)."""
        ),
        "timeout_s": 5400,
        "graders": [
            {
                "kind": "acceptance",
                "weight": 80,
                "band_weights": {"core": 30, "edge": 30, "adversarial": 40},
            },
            {
                "kind": "tree_guard",
                "weight": 10,
                "mode": "confined_diff",
                "allowed_paths": [
                    "jobsched.py",
                    "README.md",
                    "CHANGELOG.md",
                    ".gitignore",
                    ".jobsched",
                ],
            },
            {
                "kind": "envelope_guard",
                "weight": 10,
                "expect_verified": True,
            },
        ],
    },
]
