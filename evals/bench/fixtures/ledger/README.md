# ledger

A small command-line tool for importing daily transaction batches
exported from an upstream banking feed, validating them, and producing
reconciliation and summary reports.

## Layout

Each day's export lands in its own directory under `data/`, containing
two CSV files:

- `accounts.csv` — one row per account: id, display name, currency,
  timezone, opening balance, and the expected closing balance for the
  batch (normally taken from the bank's own end-of-day statement).
- `transactions.csv` — one row per transaction: id, account id, UTC
  timestamp, amount, category, and a free-text description.

## Usage

```
python3 main.py import data/batch_2026-01-15
python3 main.py list data/batch_2026-01-15 --page 1 --page-size 5
python3 main.py aggregate data/batch_2026-01-15 data/batch_2026-01-16
python3 main.py reconcile data/batch_2026-01-15
python3 main.py report data/batch_2026-01-15 --format text
python3 main.py statement data/batch_2026-01-15 ACC1001
python3 main.py export data/batch_2026-01-15 batch.json
python3 main.py diff data/batch_2026-01-15 data/batch_2026-01-16
```

Pass `--config path/to/config.json` to any command to override the
defaults in `ledger/config.py` (page size, base currency, cache TTL,
etc). See `config.json` in this directory for an example.

## Commands

- `import` — parse a batch directory and print how many accounts,
  transactions, and row-level errors were found.
- `list` — page through a batch's transactions.
- `aggregate` — print category totals for one or more batch
  directories (handy for a week-to-date summary across several days).
- `reconcile` — compare each account's opening balance plus its
  transactions against the batch's expected closing balance.
- `report` — the full report: account summary, currency exposure
  (every account's total converted into `base_currency`), top movers,
  recent activity, and per-account daily totals, in text, CSV, or
  JSON.
- `statement` — one account's transactions in chronological order with
  a running balance, starting from its opening balance.
- `export` — write an imported batch (accounts, transactions, and
  row-level errors) to a single JSON file.
- `diff` — compare two batches by account and transaction id, e.g. a
  same-day re-import against the original, or one day's batch against
  the next; reports what was added or removed and how the row-error
  count changed.

Transaction rows are validated against `known_categories` in the
active config; a row whose category isn't recognized is rejected the
same way a row with a malformed field is, and shows up in `import`'s
error list.

## Requirements

Python 3.11+, standard library only. No network access and no
third-party packages are required.
