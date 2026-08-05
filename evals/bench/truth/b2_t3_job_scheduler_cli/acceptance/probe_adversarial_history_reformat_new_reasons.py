"""Adversarial: `history`'s reformatting (Section 7) must pass the reason
field through byte-for-byte for the two NEW skip reasons, not just the
original `waiting-on-dependency`. An implementation that hardcodes a
status/reason mapping instead of echoing whatever was actually logged
(e.g. `reason if reason == "waiting-on-dependency" else "-"`) passes every
other history probe and fails only here.

Requirement 7/10 adversarial case (history must not re-derive reasons).
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
        "c | 3 | 1 | - | true\n"
        "a | 2 | 1 | - | true\n"
        "b | 1 | 1 | - | true\n",
    )

    proc = _lib.run_cli(work, ["cancel", "jobs.txt", "c"])
    if proc.returncode != 0:
        _lib.fail(f"cancel: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    proc = _lib.run_cli(work, ["run", "jobs.txt", "--until", "1", "--max-concurrent", "1"])
    if proc.returncode != 0:
        _lib.fail(f"run: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    proc = _lib.run_cli(work, ["history"])
    if proc.returncode != 0:
        _lib.fail(f"history: expected exit 0, got {proc.returncode}; stderr={proc.stderr!r}")

    out = proc.stdout
    if "tick=0 job=c status=skipped exit_code=- reason=cancelled" not in out:
        _lib.fail(f"expected c's line to show reason=cancelled verbatim, got: {out!r}")
    if "tick=0 job=b status=skipped exit_code=- reason=budget-exceeded" not in out:
        _lib.fail(f"expected b's line to show reason=budget-exceeded verbatim, got: {out!r}")

    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
