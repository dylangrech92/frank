"""Adversarial: cancellation retroactively revokes a job's completed-ness
for its dependents, even though its `last_status` field stays literally
"completed" forever as a historical record. This is the single hardest
interaction in the spec: an implementation that computes eligibility from
`last_status == "completed"` (the obvious, natural reading of Section 4
alone) passes every other probe and fails only this one.

Sequence: producer and consumer both complete in tick 0 (same-tick
dependency satisfaction). producer is then cancelled. A second, extending
`run` invocation must show, at tick 1: producer skipped/cancelled, AND
consumer skipped/waiting-on-dependency -- NOT consumer executing again on
the strength of producer's still-"completed" last_status.

Requirement 10 adversarial case (retroactive dependency revocation).
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
        "producer | 5 | 1 | -        | true\n"
        "consumer | 1 | 1 | producer | true\n",
    )

    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if proc.returncode != 0:
        _lib.fail(f"first run: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log = _lib.read_log(work)
    tick0 = {}
    for line in log.splitlines():
        if not line:
            continue
        tick, job_id, status, exit_code, reason = line.split("\t")
        if int(tick) == 0:
            tick0[job_id] = status
    if tick0.get("producer") != "completed" or tick0.get("consumer") != "completed":
        _lib.fail(f"setup failed: expected both completed at tick 0, got {tick0}")

    proc = _lib.run_cli(work, ["cancel", "jobs.txt", "producer"])
    if proc.returncode != 0:
        _lib.fail(f"cancel: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    state = _lib.read_state(work)
    if state["jobs"]["producer"]["last_status"] != "completed":
        _lib.fail(
            "cancel must NOT rewrite last_status -- it stays 'completed' as a historical "
            f"record, got {state['jobs']['producer']['last_status']!r}"
        )

    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "2"])
    if proc.returncode != 0:
        _lib.fail(f"second run: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log = _lib.read_log(work)
    tick1 = {}
    for line in log.splitlines():
        if not line:
            continue
        tick, job_id, status, exit_code, reason = line.split("\t")
        if int(tick) == 1:
            tick1[job_id] = (status, reason)

    if tick1.get("producer") != ("skipped", "cancelled"):
        _lib.fail(f"expected producer skipped/cancelled at tick 1, got {tick1.get('producer')}")
    if tick1.get("consumer") != ("skipped", "waiting-on-dependency"):
        _lib.fail(
            "expected consumer skipped/waiting-on-dependency at tick 1 (producer's cancellation "
            f"must revoke its completed-ness), got {tick1.get('consumer')}"
        )

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
