#!/usr/bin/env python3
"""Proves the first decoy is genuinely alive: ledger/handlers.py's
handle_csv / handle_json / handle_text have zero direct call sites
anywhere in the codebase (see the grep in the audit notes) -- they are
reached only through `render()`'s `getattr(sys.modules[__name__],
f"handle_{fmt}")` dispatch, driven by the CLI's `--format` flag. This
runs each of the three formats through that exact dispatch path and
checks the output is correct for each.
"""
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "ledger"
sys.path.insert(0, str(FIXTURE))

from ledger import handlers  # noqa: E402


def main() -> int:
    rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    fields = ["a", "b"]

    csv_out = handlers.render("csv", rows, fields)
    json_out = handlers.render("json", rows, fields)
    text_out = handlers.render("text", rows, fields)

    print("render('csv', ...) ->")
    print(csv_out)
    print("render('json', ...) ->")
    print(json_out)
    print("render('text', ...) ->")
    print(text_out)

    ok = (
        "1,2" in csv_out and "3,4" in csv_out
        and '"a": "1"' in json_out and '"b": "4"' in json_out
        and "1\t2" in text_out and "3\t4" in text_out
    )
    try:
        handlers.render("xml", rows, fields)
        unknown_raises = False
    except ValueError:
        unknown_raises = True

    if ok and unknown_raises:
        print("CONFIRMED: all three handle_* functions fire correctly through "
              "getattr dispatch, and an unrecognized format still fails loudly.")
        return 0
    print("NOT REPRODUCED: one of the dispatched formats produced unexpected output.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
