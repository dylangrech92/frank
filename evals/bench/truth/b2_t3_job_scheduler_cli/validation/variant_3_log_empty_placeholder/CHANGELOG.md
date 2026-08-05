# Changelog

## 1.0.0 — initial release

- Schedule file parsing: `job_id | priority | interval | depends_on |
  command`, with a precise field-by-field validation order and per-line
  error reporting.
- Dependency graph (DAG) validation: unknown-dependency detection and
  cycle detection with a deterministic, minimal cycle report.
- Simulated-clock `run`: integer tick loop, due-job computation, and
  priority-ordered execution with per-job dependency eligibility.
- Crash-safe state persistence: atomic writes, an `in_progress_tick`
  crash marker, and automatic recovery of an interrupted tick on the next
  `run`.
- Run log with automatic rotation once the active segment exceeds 10000
  bytes, keeping up to 5 rotated segments.
- `--dry-run` planning mode: identical due/priority/dependency logic,
  computed entirely in memory, with no filesystem side effects.
- Idempotent re-run: a `run` that has already reached the requested tick
  is a no-op.
- `status` and `history` query commands for inspecting persisted state and
  the run log without mutating anything.
- A uniform exit-code contract (0, 1, 2, 3, 4, 5, 6, 70) applied
  identically across `validate`, `run`, `status`, and `history`.
