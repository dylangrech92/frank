"""Adversarial: a job that fails EVERY time it runs permanently blocks its
dependent — the dependent must never transition to completed no matter how
many ticks are simulated. This is the negative counterpart to the core
'dependency_blocks_until_satisfied' probe, which uses a producer that
succeeds; here the producer command is `false` and never produces a
completed status for consumer to observe, across 5 ticks.

Requirement 4, final paragraph.
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
        "producer | 1 | 1 | -        | false\n"
        "consumer | 1 | 1 | producer | true\n",
    )
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "5"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0 (job failures never affect run's own exit code), got {proc.returncode}; stderr={proc.stderr!r}")

    log = _lib.read_log(work)
    by_tick = {}
    for line in log.splitlines():
        if not line:
            continue
        tick, job_id, status, exit_code, reason = line.split("\t")
        by_tick.setdefault(int(tick), {})[job_id] = (status, reason)

    for t in range(5):
        producer_status, _ = by_tick.get(t, {}).get("producer", (None, None))
        consumer_status, consumer_reason = by_tick.get(t, {}).get("consumer", (None, None))
        if producer_status != "failed":
            _lib.fail(f"tick {t}: expected producer failed, got {producer_status}")
        if consumer_status != "skipped" or consumer_reason != "waiting-on-dependency":
            _lib.fail(
                f"tick {t}: expected consumer permanently skipped/waiting-on-dependency, "
                f"got status={consumer_status} reason={consumer_reason}"
            )

    state = _lib.read_state(work)
    if "consumer" in state["jobs"]:
        _lib.fail(f"consumer must never have executed (no jobs entry), got {state['jobs'].get('consumer')}")
    if state["jobs"]["producer"]["run_count"] != 5:
        _lib.fail(f"expected producer run_count=5, got {state['jobs']['producer']}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
