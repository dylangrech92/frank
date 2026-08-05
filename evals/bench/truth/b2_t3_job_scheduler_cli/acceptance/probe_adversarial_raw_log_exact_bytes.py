"""Adversarial: the raw run.log bytes for one tick containing all three
statuses (completed, failed, skipped) in the same tick must match the
5-tab-separated-field, newline-terminated format EXACTLY — including that
`exit_code` is the literal character '-' (not empty string, not 'None',
not 'null') for skipped lines, and `reason` is the literal character '-'
(not empty) for completed/failed lines. Checks the raw bytes directly
rather than via str.split, so a stray extra space or a '\\r\\n' line ending
would be caught.

Requirement 6 (run log format), exact byte contract.
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
        "consumer   | 5 | 1 | producer | true\n"
        "producer   | 9 | 1 | -        | false\n"
        "standalone | 1 | 1 | -        | true\n",
    )
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log_path = os.path.join(work, ".jobsched", "logs", "run.log")
    with open(log_path, "rb") as f:
        raw = f.read()

    expected = (
        b"0\tproducer\tfailed\t1\t-\n"
        b"0\tconsumer\tskipped\t-\twaiting-on-dependency\n"
        b"0\tstandalone\tcompleted\t0\t-\n"
    )
    if raw != expected:
        _lib.fail(f"raw log bytes mismatch.\nexpected: {expected!r}\ngot:      {raw!r}")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
