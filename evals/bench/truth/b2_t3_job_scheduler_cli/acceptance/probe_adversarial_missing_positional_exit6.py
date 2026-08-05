"""Adversarial: argparse's own default for a missing required positional
argument (or an unknown subcommand, or no subcommand at all) is exit code 2
— but Section 9 reserves 2 exclusively for schedule parse errors and
requires every usage problem, on every subcommand, to surface as exit 6
instead. This is the trap the spec calls out by name: an implementation
that hands argparse's ArgumentParser to add_subparsers() unmodified will
leak exit 2 here. Exercises three distinct usage-error shapes: missing
positional, no subcommand at all, and an unknown subcommand.

Requirement 9 (exit-code contract), explicitly-called-out collision trap.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)

    missing_positional = _lib.run_cli(work, ["validate"])
    if missing_positional.returncode != 6:
        _lib.fail(
            f"missing schedule_file positional: expected exit 6, got "
            f"{missing_positional.returncode}; stderr={missing_positional.stderr!r}"
        )

    no_subcommand = _lib.run_cli(work, [])
    if no_subcommand.returncode != 6:
        _lib.fail(f"no subcommand: expected exit 6, got {no_subcommand.returncode}; stderr={no_subcommand.stderr!r}")

    unknown_subcommand = _lib.run_cli(work, ["frobnicate"])
    if unknown_subcommand.returncode != 6:
        _lib.fail(
            f"unknown subcommand: expected exit 6, got "
            f"{unknown_subcommand.returncode}; stderr={unknown_subcommand.stderr!r}"
        )

    for proc in (missing_positional, no_subcommand, unknown_subcommand):
        if proc.returncode == 2:
            _lib.fail("argparse's raw default exit code 2 leaked through for a usage error")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
