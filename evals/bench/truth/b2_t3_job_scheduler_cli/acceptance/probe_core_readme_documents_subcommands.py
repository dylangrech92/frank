"""Core: README.md documents the CLI name and all five subcommands,
including the newer `cancel` subcommand and `run`'s `--max-concurrent` flag.

Requirement 11 happy path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    readme_path = os.path.join(work, "README.md")
    if not os.path.exists(readme_path):
        _lib.fail("README.md does not exist")
    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()

    if "jobsched.py" not in text:
        _lib.fail("README.md never mentions jobsched.py")
    for sub in ("validate", "run", "status", "history", "cancel"):
        if sub not in text:
            _lib.fail(f"README.md never mentions the '{sub}' subcommand")
    if "--max-concurrent" not in text:
        _lib.fail("README.md never mentions the --max-concurrent flag")
    if "exit code" not in text.lower():
        _lib.fail("README.md never mentions exit codes")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
