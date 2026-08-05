"""Core: CHANGELOG.md exists with a version/date heading and real content.

Requirement 11 happy path.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib

HEADING_RE = re.compile(r"^#{1,3}\s+\S", re.MULTILINE)


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    changelog_path = os.path.join(work, "CHANGELOG.md")
    if not os.path.exists(changelog_path):
        _lib.fail("CHANGELOG.md does not exist")
    with open(changelog_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not HEADING_RE.search(text):
        _lib.fail("CHANGELOG.md has no markdown heading")
    if len(text.strip()) < 80:
        _lib.fail(f"CHANGELOG.md content looks too thin ({len(text.strip())} chars)")
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
