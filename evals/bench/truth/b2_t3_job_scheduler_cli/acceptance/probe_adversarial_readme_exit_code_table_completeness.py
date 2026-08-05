"""Adversarial: README.md must reproduce ALL 8 rows of the Section 9
exit-code table (0, 1, 2, 3, 4, 5, 6, 70), each on its own table row
(pipe-delimited markdown row containing the code as a standalone token).
A common shortcut is documenting only the common codes (0/1/2) and leaving
the rarer ones (4, 5, 70) to prose or omitting them entirely — this probe
demands the full, exact 8-row table the spec requires, not a paraphrase.

Requirement 11 ("the complete exit-code table ... reproduced in full (all
8 rows)").
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _lib

REQUIRED_CODES = ["0", "1", "2", "3", "4", "5", "6", "70"]


def main(tree_path):
    work = _lib.copy_tree(tree_path)
    readme_path = os.path.join(work, "README.md")
    if not os.path.exists(readme_path):
        _lib.fail("README.md does not exist")
    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()

    # A markdown table row for a given code: a line starting with '|',
    # whose first cell (trimmed) is exactly that code. Processed strictly
    # line-by-line (never letting a match span a newline) so a separator
    # row like '|---|---|' can never swallow the next row's leading '|'.
    found_codes = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 2:
            continue
        first_cell = cells[1].strip()
        if first_cell:
            found_codes.add(first_cell)

    missing = [c for c in REQUIRED_CODES if c not in found_codes]
    if missing:
        _lib.fail(
            f"README exit-code table missing rows for codes {missing}; "
            f"found table-row codes = {sorted(found_codes)}"
        )
    _lib.ok()


if __name__ == "__main__":
    main(sys.argv[1])
