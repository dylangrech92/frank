"""Adversarial: within one tick, cancelled/dependency/budget classification
follows a strict, exact-first-match precedence (Section 3, step 3), and all
three skip reasons plus a completion and a failure can coexist in a single
tick. Five jobs, one tick, --max-concurrent 2, exercise every branch at
once:

  p (prio 10, no deps)              -> executes, succeeds  (budget_used: 0->1)
  q (prio  9, no deps, exits 1)     -> executes, fails      (budget_used: 1->2)
  r (prio  8, no deps)              -> pre-cancelled        -> skipped/cancelled
  s (prio  7, depends_on r)         -> r never completed    -> skipped/waiting-on-dependency
  t (prio  6, no deps)              -> budget already 2/2   -> skipped/budget-exceeded

r's cancellation must be checked BEFORE s's dependency eligibility and
BEFORE t's budget (Section 3 step 3's fixed a/b/c/d order) -- a scheduler
that evaluates these in any other order, or that lets r's skip consume a
budget slot, produces a different (wrong) outcome for s or t.

Requirement 10 adversarial case (exact skip-reason precedence).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(
        work,
        "jobs.txt",
        "p | 10 | 1 | -  | exit 0\n"
        "q |  9 | 1 | -  | exit 1\n"
        "r |  8 | 1 | -  | exit 0\n"
        "s |  7 | 1 | r  | exit 0\n"
        "t |  6 | 1 | -  | exit 0\n",
    )

    proc = _lib.run_cli(work, ["cancel", "jobs.txt", "r"])
    if proc.returncode != 0:
        _lib.fail(f"cancel: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1", "--max-concurrent", "2"])
    if proc.returncode != 0:
        _lib.fail(f"run: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    expected_summary = "tick 0: 1 completed, 1 failed, 3 skipped"
    if expected_summary not in proc.stdout:
        _lib.fail(f"expected {expected_summary!r} in stdout, got {proc.stdout!r}")

    log = _lib.read_log(work)
    lines = [ln for ln in log.splitlines() if ln]
    expected_lines = [
        "0\tp\tcompleted\t0\t-",
        "0\tq\tfailed\t1\t-",
        "0\tr\tskipped\t-\tcancelled",
        "0\ts\tskipped\t-\twaiting-on-dependency",
        "0\tt\tskipped\t-\tbudget-exceeded",
    ]
    if lines != expected_lines:
        _lib.fail(f"expected log lines (in resolution order) {expected_lines}, got {lines}")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
