"""Adversarial: re-running with the same --until after completion must
leave run.log and state.json BYTE-IDENTICAL to before the no-op re-run —
stricter than the core idempotent-rerun probe, which only checks run_count.
A buggy implementation could satisfy 'run_count unchanged' while still
re-appending identical-looking log lines (double bookkeeping that happens
to read back the same) or rewriting state.json with different key
ordering/whitespace. Raw bytes catch both.

Requirement 3 ("already complete" short-circuit), Requirement 6.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(
        work,
        "jobs.txt",
        "a | 1 | 1 | -   | true\n"
        "b | 1 | 1 | a   | true\n",
    )
    first = _lib.run_cli(work, ["run", "jobs.txt", "--until", "3"])
    if first.returncode != 0:
        _lib.fail(f"first run: expected exit 0, got {first.returncode}; stderr={first.stderr!r}")

    log_path = os.path.join(work, ".jobsched", "logs", "run.log")
    state_path = os.path.join(work, ".jobsched", "state.json")
    log_before = _read_bytes(log_path)
    state_before = _read_bytes(state_path)

    second = _lib.run_cli(work, ["run", "jobs.txt", "--until", "3"])
    if second.returncode != 0:
        _lib.fail(f"second run: expected exit 0, got {second.returncode}; stderr={second.stderr!r}")
    if "already complete" not in second.stdout:
        _lib.fail(f"expected 'already complete' message, got: {second.stdout!r}")

    log_after = _read_bytes(log_path)
    state_after = _read_bytes(state_path)

    if log_before != log_after:
        _lib.fail(f"run.log bytes changed on no-op re-run.\nbefore: {log_before!r}\nafter:  {log_after!r}")
    if state_before != state_after:
        _lib.fail(f"state.json bytes changed on no-op re-run.\nbefore: {state_before!r}\nafter:  {state_after!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
