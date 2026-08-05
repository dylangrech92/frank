#!/usr/bin/env python3
"""Proves the two remaining decoys are genuinely alive / correct, not the
defects a naive skim might mistake them for:

1. ledger/errors.py:safe_call -- reads like a swallowed exception (a bare
   `try/except Exception`) but actually re-raises after logging. Proven
   here by catching the propagated exception at the call site and
   inspecting the log output.
2. ledger/validators.py's `@register_field_validator`-registered
   functions (_validate_account_id_field, _validate_currency_field,
   _validate_txn_id_field) -- have zero direct call sites (see the grep
   in the audit notes) but are genuinely invoked through
   `FIELD_VALIDATORS[column]` from `importer._validate_row` during a real
   import. Proven here by importing a row with a deliberately invalid
   account_id and showing it gets rejected.
"""
import io
import logging
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "ledger"
sys.path.insert(0, str(FIXTURE))

from ledger.errors import safe_call  # noqa: E402
from ledger.importer import import_directory  # noqa: E402


def boom():
    raise ValueError("simulated failure deep in the pipeline")


def check_reraise() -> bool:
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logging.getLogger("ledger.errors").addHandler(handler)
    logging.getLogger("ledger.errors").setLevel(logging.ERROR)

    raised = False
    try:
        safe_call(boom)
    except ValueError as exc:
        raised = True
        print(f"safe_call() propagated: {exc!r}")

    logged = "simulated failure deep in the pipeline" in log_stream.getvalue()
    print(f"exception re-raised to caller: {raised}")
    print(f"exception also logged before re-raising: {logged}")
    return raised and logged


def check_registry_alive() -> bool:
    batch_dir = FIXTURE / "data" / "batch_2026-01-15"
    batch = import_directory(batch_dir)
    print(f"clean import: {len(batch.accounts)} accounts, {len(batch.errors)} errors")

    import csv
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="ledger_decoy_"))
    try:
        work_dir = tmp / "batch"
        shutil.copytree(batch_dir, work_dir)
        rows = list(csv.DictReader((work_dir / "accounts.csv").open()))
        rows[0]["account_id"] = "bad id!"  # fails ACCOUNT_ID_RE
        with (work_dir / "accounts.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        dirty_batch = import_directory(work_dir)
        print(f"import with a deliberately invalid account_id: "
              f"{len(dirty_batch.accounts)} accounts accepted, "
              f"{len(dirty_batch.errors)} row(s) rejected: "
              f"{[e.reason for e in dirty_batch.errors]}")
        rejected = any("account_id" in e.reason for e in dirty_batch.errors)
        return rejected
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("--- decoy 1: safe_call is a catch-log-reraise wrapper, not a swallow ---")
    ok1 = check_reraise()
    print()
    print("--- decoy 2: FIELD_VALIDATORS registry entries are reachable via importer ---")
    ok2 = check_registry_alive()

    print()
    if ok1 and ok2:
        print("CONFIRMED: both decoys are alive/correct code, not defects.")
        return 0
    print("one or both decoy checks failed to reproduce the expected alive behavior.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
