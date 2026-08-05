#!/usr/bin/env python3
"""Demonstrates ledger/importer.py:_read_csv_rows swallowing a real I/O
error (permission denied) and returning an empty result set silently --
no exception, no error record, no log line, exit code 0.

Run from anywhere:
    python3 demo_t1_swallowed_io_error.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "ledger"
sys.path.insert(0, str(FIXTURE))

from ledger.importer import import_directory  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ledger_t1_"))
    try:
        batch_dir = tmp / "batch"
        shutil.copytree(FIXTURE / "data" / "batch_2026-01-15", batch_dir)

        txn_path = batch_dir / "transactions.csv"
        before_size = txn_path.stat().st_size
        print(f"before: transactions.csv exists, {before_size} bytes, readable")

        os.chmod(txn_path, 0o000)  # simulate a real permission-denied I/O error
        try:
            batch = import_directory(batch_dir)
        finally:
            os.chmod(txn_path, 0o644)  # restore so tempdir cleanup can remove it

        print(f"after chmod 000: accounts={len(batch.accounts)} "
              f"transactions={len(batch.transactions)} errors={len(batch.errors)}")

        if len(batch.transactions) == 0 and len(batch.errors) == 0:
            print("CONFIRMED: the unreadable file produced zero transactions AND "
                  "zero recorded errors -- the OSError from open() was caught and "
                  "discarded in _read_csv_rows with no trace left for the caller.")
            return 0
        else:
            print("NOT REPRODUCED: batch.errors was non-empty or transactions were "
                  "still imported; the swallowed-exception defect may have been fixed.")
            return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
