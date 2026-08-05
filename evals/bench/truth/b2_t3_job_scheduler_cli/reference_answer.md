Built `jobsched.py`, a single stdlib-only Python 3 CLI implementing all
eleven requirements from the spec, plus `README.md` and `CHANGELOG.md`.

- **Schedule parsing** (`validate`, and the first stage of `run`): a
  `job_id | priority | interval | depends_on | command` line format, parsed
  in a fixed per-line validation order (field count, job id charset,
  priority, interval, depends_on syntax), followed by two whole-file passes
  — duplicate job ids, then unknown-dependency references — so a later
  line's syntax error is always reported before an earlier line's
  duplicate, matching the spec's stated pass ordering. `validate` prints
  `OK: <n> jobs, <m> edges` and exits 0; a bad line exits 2 with
  `error: line <L>: <reason>`.

- **DAG validation**: unknown dependency references and cycles both exit
  3. Cycles are reported starting from the lexicographically smallest
  participating job id, so the message is deterministic regardless of
  which node the cycle search visits first.

- **Simulated-clock `run`**: an integer tick loop (`--until N` simulates
  ticks `0..N-1`); at each tick, due jobs (`t % interval == 0`) are ordered
  by priority descending, ties by job id ascending, and executed only once
  every dependency has most recently completed — including dependencies
  satisfied earlier in the same tick's ordered walk. A job's own failure
  never fails `run` itself; it only blocks that job's dependents until it
  succeeds again. Within a tick, three skip reasons compose under a strict
  precedence: `cancelled` beats `waiting-on-dependency` beats
  `budget-exceeded` (the last only possible with `--max-concurrent`) — a job
  matching more than one is reported under the first match only, so the
  reason string is deterministic.

- **Crash-safe persistence**: state is one atomically-written JSON file
  (`os.replace` from a temp file), with an `in_progress_tick` marker set
  before a tick's jobs run and cleared only after that tick's log lines and
  state are both durably written. If `run` is killed mid-tick, the next
  invocation detects the dangling marker and re-attempts that tick from
  scratch — documented as at-least-once re-attempt on crash, exactly-once
  bookkeeping on a clean run.

- **Log rotation**: one tab-separated line per resolved job appended to
  `run.log`; once it exceeds 10000 bytes it rotates to `run.log.1..5`,
  oldest segment dropped.

- **`--dry-run`**: the identical due/priority/dependency computation
  against an in-memory scratch copy of state — never touches disk, never
  spawns a subprocess, and multi-tick plans stay coherent because a
  would-run job is treated as completed for evaluating later ticks within
  the same dry-run invocation.

- **Idempotent re-run**: a `run` whose persisted state already covers the
  requested `--until` prints `already complete: last_completed_tick=<t>`
  and exits 0 without touching state or logs.

- **Query commands**: `status` (last completed tick + one line per job that
  has ever been executed or cancelled) and `history` (newest-first log
  events across rotated segments, `--limit`-bounded), both read-only.
  `status` displays a cancelled job as `status=cancelled` regardless of its
  prior `last_status`, and prints the literal `none` for `last_exit_code`
  on a job cancelled before it ever ran.

- **`cancel`**: a fifth subcommand that permanently marks a job cancelled in
  persisted state (same atomic-write and corrupt/mismatch/`--force-recover`
  rules as `run`). A cancelled job is skipped by every future `run`
  (real or `--dry-run`) ahead of the dependency check, and its
  completed-ness is retroactively revoked for dependents from the moment of
  cancellation — even though its historical `last_status` is left untouched.
  Re-cancelling an already-cancelled job succeeds again with no error.

- **`run --max-concurrent`**: an optional per-tick cap on how many jobs may
  actually execute; jobs due beyond the budget are skipped with reason
  `budget-exceeded` and retried at their own next due tick, composing with
  cancellation and dependency eligibility under the precedence documented
  above. Honored identically by `--dry-run`.

- **Exit-code contract**: 0/1/2/3/4/5/6/70, applied identically across all
  five subcommands — including overriding argparse's own usage-error exit
  code so it never collides with 2 (reserved for schedule parse errors).

- **Documentation**: `README.md` covers the CLI invocation, all five
  subcommands and flags (including `cancel` and `--max-concurrent`), the
  schedule format with a worked 3-job example, the full exit-code table,
  and default state/log locations. `CHANGELOG.md` records the 1.0.0 initial
  release.

No files beyond `jobsched.py`, `README.md`, and `CHANGELOG.md` are
committed; `.jobsched/` (state + logs) is only ever written at runtime,
under the paths the spec designates, never at the project root.
