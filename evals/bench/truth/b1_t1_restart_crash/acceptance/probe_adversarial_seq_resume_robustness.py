#!/usr/bin/env python3
"""Adversarial acceptance probe for b1_t1_restart_crash.

Targets two plausible-but-wrong near-misses of the real fix, neither of
which the core/edge probes or the other adversarial probe exercise:

Stage 1 -- a fix that resumes the sequence counter from only PENDING (or
PENDING + RETRY_SCHEDULED) jobs, rather than every job still sitting in the
store regardless of status. A completed-but-not-yet-purged job still holds
its seq number; a status-filtered resume undercounts it exactly the same
way the original len()-based bug does.

Stage 2 -- a fix shaped like ``max(job.seq for job in ...)`` without a
``default=0``, which crashes with ValueError the first time a process
restarts against a store that has been fully purged (an empty ``self._jobs``
after a period of no failures/pending work is a completely normal state,
not an edge case).

Usage: python3 probe_adversarial_seq_resume_robustness.py <tree_path>
Exit 0 = pass, non-zero = fail. Stdlib only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def _import_queueworks_from(tree_path: str):
    base = tempfile.mkdtemp(prefix="b1-t1-adv-seq-")
    work_copy = os.path.join(base, "tree")
    shutil.copytree(tree_path, work_copy)
    if work_copy not in sys.path:
        sys.path.insert(0, work_copy)
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_adversarial_seq_resume_robustness.py <tree_path>", file=sys.stderr)
        return 2
    tree_path = os.path.abspath(sys.argv[1])
    state_dir = _import_queueworks_from(tree_path)

    try:
        from queueworks.models import Job, JobStatus, Priority
        from queueworks.store import JobStore
    except Exception as exc:
        print(f"FAIL: could not import queueworks: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # -- Stage 1: a completed-but-unpurged job must still reserve its seq --
    state_path_1 = os.path.join(state_dir, "stage1_state.json")
    try:
        store = JobStore(state_path_1)
        job_pending = Job(task_name="noop", priority=Priority.NORMAL, seq=2, status=JobStatus.PENDING)
        job_retry = Job(
            task_name="noop",
            priority=Priority.NORMAL,
            seq=3,
            status=JobStatus.RETRY_SCHEDULED,
            attempts=1,
        )
        # Highest seq in the store belongs to a job that finished but was
        # never purged -- the common case in between purge cycles.
        job_completed_unpurged = Job(
            task_name="noop", priority=Priority.NORMAL, seq=4, status=JobStatus.COMPLETED
        )
        store.add(job_pending)
        store.add(job_retry)
        store.add(job_completed_unpurged)

        # Fresh process restart against the same state file.
        restarted_store = JobStore(state_path_1)
        existing_seqs = {j.seq for j in restarted_store.all_jobs()}
        next_val = restarted_store.next_seq()
    except Exception as exc:
        print(
            f"FAIL: stage 1 (completed-unpurged seq reservation) raised "
            f"{type(exc).__name__}: {exc} (expected it to resume cleanly)",
            file=sys.stderr,
        )
        return 1

    if next_val in existing_seqs:
        print(
            f"FAIL: stage 1 -- next_seq() returned {next_val}, which collides with an "
            f"existing job's seq (existing seqs: {sorted(existing_seqs)}). A resumed "
            "counter must account for every job still in the store, not just "
            "pending/retry-scheduled ones -- a completed-but-unpurged job still "
            "occupies its seq.",
            file=sys.stderr,
        )
        return 1

    # -- Stage 2: a fully-purged (empty) store must not crash on restart --
    state_path_2 = os.path.join(state_dir, "stage2_state.json")
    try:
        store2 = JobStore(state_path_2)
        j = Job(task_name="noop", priority=Priority.NORMAL, seq=1, status=JobStatus.COMPLETED)
        store2.add(j)
        removed = store2.purge_completed()
        if removed != 1:
            print(f"FAIL: stage 2 setup -- expected purge_completed() to remove 1 job, removed {removed}", file=sys.stderr)
            return 1

        # Fresh process restart against a store file with zero jobs left.
        restarted_empty_store = JobStore(state_path_2)
        next_val_2 = restarted_empty_store.next_seq()
    except Exception as exc:
        print(
            f"FAIL: stage 2 (empty-store restart) raised {type(exc).__name__}: {exc} "
            "(a fully-purged store is a normal state, not a crash condition)",
            file=sys.stderr,
        )
        return 1

    if next_val_2 != 1:
        print(
            f"FAIL: stage 2 -- expected the first seq handed out after a fully-purged "
            f"restart to be 1, got {next_val_2}",
            file=sys.stderr,
        )
        return 1

    print("PASS: seq resume accounts for all job statuses and survives a fully-purged restart")
    return 0


if __name__ == "__main__":
    sys.exit(main())
