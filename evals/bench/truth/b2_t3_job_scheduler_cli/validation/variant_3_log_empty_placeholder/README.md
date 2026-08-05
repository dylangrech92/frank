# jobsched

A dependency-aware job scheduler CLI with a simulated clock, crash-safe
persistence, log rotation, and query commands.

## Running it

```
python3 jobsched.py <subcommand> ...
```

`jobsched.py` is a single, stdlib-only Python 3 script. No install step, no
third-party packages, no network access.

`jobsched` never reads the real system clock — all timing is a simulated
integer "tick" counter you supply on the command line, so every run is
deterministic given the same schedule file and flags.

## Subcommands

### `validate <schedule_file>`

Parses and validates a schedule file: field syntax, then the dependency
graph (unknown references, cycles). Prints `OK: <n> jobs, <m> edges` on
success. Does not touch any runtime state.

### `run <schedule_file> --until N [--dry-run] [--state-file PATH] [--log-dir PATH] [--force-recover]`

Simulates ticks `0` through `N-1` against the schedule.

| Flag | Default | Meaning |
|---|---|---|
| `--until N` | *(required)* | simulate ticks `0..N-1` |
| `--dry-run` | off | compute and print the plan; make no filesystem changes |
| `--state-file PATH` | `.jobsched/state.json` | where run state is persisted |
| `--log-dir PATH` | `.jobsched/logs` | directory holding `run.log` and its rotated segments |
| `--force-recover` | off | discard a corrupt or schedule-mismatched state file and start fresh |

Re-running with the same schedule and `--until` is idempotent: a run that
finds it has already reached the requested tick prints
`already complete: last_completed_tick=<t>` and exits `0` without touching
state or logs. A job's own non-zero exit is never a scheduler error — `run`
exits `0` as long as the scheduler itself completed the requested ticks;
inspect individual job outcomes with `status`.

### `status [--state-file PATH]`

Prints `last_completed_tick: <t or none>`, then one line per job that has
been executed at least once, sorted by job id:
`<job_id>: status=<completed|failed> run_count=<n> last_exit_code=<code>`.
Jobs never executed are omitted. Prints `no state` if `--state-file` does
not exist.

### `history [--log-dir PATH] [--limit N]`

Prints up to `--limit` (default `20`) run-log events, newest first, as
`tick=<t> job=<id> status=<status> exit_code=<code|-> reason=<reason|->`.
Reads across rotated log segments. Prints `no history` if no log file
exists yet.

## Schedule file format

A UTF-8 text file. Blank lines and lines whose first non-whitespace
character is `#` are ignored. Every other line defines one job with exactly
5 fields separated by `|`:

```
job_id | priority | interval | depends_on | command
```

- `job_id`: matches `[A-Za-z0-9_-]+`, unique in the file.
- `priority`: any integer. Among jobs due in the same tick, higher priority
  runs first; ties break by `job_id` ascending.
- `interval`: a positive integer. The job is due at tick `t` whenever
  `t % interval == 0`.
- `depends_on`: `-` for no dependencies, or a comma-separated list of other
  `job_id`s. A due job only runs once every id in its `depends_on` list has
  most recently completed successfully (from an earlier tick, or earlier in
  the same tick's priority-ordered walk); otherwise it is skipped for that
  tick and retried at its next due tick.
- `command`: the rest of the line (may itself contain `|`). Run via the
  shell; its own stdout/stderr are discarded, only its exit code is kept.

Worked example:

```
backup  | 5 | 2 | -      | ./scripts/backup.sh
report  | 3 | 2 | backup | ./scripts/report.sh
cleanup | 1 | 4 | report | ./scripts/cleanup.sh
```

`backup` and `report` are due every 2 ticks, `cleanup` every 4. Because
`backup` has the highest priority it always runs first in a shared tick,
satisfying `report`'s dependency in that same tick; `cleanup` then runs
whenever both `report` (and transitively `backup`) have completed.

## State and logs

Real (non-`--dry-run`) runs persist state atomically to `--state-file`
(default `.jobsched/state.json`) and append one line per resolved job to
`--log-dir/run.log` (default `.jobsched/logs/run.log`), rotating it to
`run.log.1` .. `run.log.5` once it exceeds 10000 bytes. `validate`,
`status`, `history`, and `--dry-run` runs never create `.jobsched/` or
anything under it.

If `run` is killed mid-tick, the next `run` invocation detects the
incomplete tick (via `in_progress_tick` in the state file) and re-attempts
it from scratch — exactly-once bookkeeping per tick that completes cleanly,
at-least-once re-attempt of a tick that was interrupted.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | schedule file not found or unreadable (`validate`, `run`) |
| 2 | schedule parse error |
| 3 | DAG error: unknown dependency or cycle |
| 4 | state file corrupt and `--force-recover` not given (`run`, `status`) |
| 5 | schedule/state mismatch and `--force-recover` not given (`run`) |
| 6 | CLI usage error: missing/invalid required argument, unknown flag, unknown or missing subcommand |
| 70 | unexpected internal error (always caught; never a bare traceback) |

This table applies uniformly to all four subcommands. A due job's command
exiting non-zero during `run` never changes `run`'s own exit code.
