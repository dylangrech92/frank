"""Core: a completed job's run.log line matches the documented 5-field
tab-separated format.

Requirement 7 happy path.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib

LINE_RE = re.compile(r"^0\ta\tcompleted\t0\t-$")


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    _lib.write_file(work, "jobs.txt", "a | 1 | 1 | - | true\n")
    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1"])
    if proc.returncode != 0:
        _lib.fail(f"expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    log = _lib.read_log(work)
    lines = [l for l in log.splitlines() if l]
    if len(lines) != 1:
        _lib.fail(f"expected exactly 1 log line, got {lines}")
    if not LINE_RE.match(lines[0]):
        _lib.fail(f"log line does not match expected format: {lines[0]!r}")
    if not log.endswith("\n"):
        _lib.fail("log file does not end with a newline")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
