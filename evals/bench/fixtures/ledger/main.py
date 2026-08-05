#!/usr/bin/env python3
"""ledger: a small batch CSV import / validate / aggregate / report tool.

Typical flow:

    python3 main.py import data/batch_2026-01-15
    python3 main.py list data/batch_2026-01-15 --page 1 --page-size 5
    python3 main.py aggregate data/batch_2026-01-15 data/batch_2026-01-16
    python3 main.py reconcile data/batch_2026-01-15
    python3 main.py report data/batch_2026-01-15 --format text
    python3 main.py statement data/batch_2026-01-15 ACC1001
    python3 main.py export data/batch_2026-01-15 batch.json
    python3 main.py diff data/batch_2026-01-15 data/batch_2026-01-16
"""
from __future__ import annotations

import argparse
import logging
import sys

from ledger import aggregator, diff, reports, serialization, statements
from ledger.config import Config
from ledger.errors import safe_call
from ledger.importer import import_directory
from ledger.paginator import get_page, total_pages
from ledger.validators import classify_amount

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ledger.cli")


def _load_config(args: argparse.Namespace) -> Config:
    return Config.load(args.config)


def cmd_import(args: argparse.Namespace) -> int:
    config = _load_config(args)
    batch = safe_call(import_directory, args.input_dir, config)
    print(f"imported {len(batch.accounts)} accounts, {batch.row_count} transactions, "
          f"{len(batch.errors)} errors from {args.input_dir}")
    for err in batch.errors:
        print(f"  skipped {err.source_file}:{err.line_number}: {err.reason}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    config = _load_config(args)
    page_size = args.page_size or config.page_size
    batch = safe_call(import_directory, args.input_dir, config)
    transactions = batch.transactions
    pages = total_pages(len(transactions), page_size)
    page = get_page(transactions, args.page, page_size)
    print(f"page {args.page} of {pages} ({len(transactions)} total transactions)")
    for txn in page:
        kind = classify_amount(txn.amount)
        print(f"  {txn.txn_id}  {txn.account_id}  {txn.amount:>10.2f}  {kind:<6}  {txn.category}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    for input_dir in args.input_dirs:
        batch = safe_call(import_directory, input_dir, config)
        totals = aggregator.summarize_by_category(batch.transactions)
        print(f"{input_dir}:")
        for category, total in sorted(totals.items()):
            print(f"  {category:<12} {total:>12.2f}")
        print()
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    config = _load_config(args)
    batch = safe_call(import_directory, args.input_dir, config)
    by_account: dict[str, list] = {}
    for txn in batch.transactions:
        by_account.setdefault(txn.account_id, []).append(txn)

    exit_code = 0
    for account in batch.accounts:
        txns = by_account.get(account.account_id, [])
        is_balanced, computed = aggregator.reconcile(txns, account)
        status = "OK" if is_balanced else "MISMATCH"
        if not is_balanced:
            exit_code = 1
        print(f"  {account.account_id}  {account.name:<20}  computed={computed:>12.2f}  "
              f"expected={account.expected_closing_balance:>12.2f}  {status}")
    return exit_code


def cmd_report(args: argparse.Namespace) -> int:
    config = _load_config(args)
    batch = safe_call(import_directory, args.input_dir, config)
    cache = reports.build_account_cache(batch, config.cache_ttl_seconds)
    account_rows = reports.account_summary_rows(batch, cache)

    print("== accounts ==")
    print(reports.render_report(
        args.format, account_rows,
        ["account_id", "account_name", "currency", "total"],
    ))

    print(f"\n== currency exposure ({config.base_currency}) ==")
    print(reports.render_report(
        args.format, reports.currency_exposure_rows(account_rows, config.base_currency),
        ["account_id", "account_name", "native_currency", "native_total",
         "target_currency", "converted_total"],
    ))

    print("\n== top movers ==")
    print(reports.render_report(
        args.format, reports.top_movers_rows(batch, limit=10),
        ["txn_id", "account_id", "amount", "category"],
    ))

    print("\n== recent activity ==")
    print(reports.render_report(
        args.format, reports.recent_activity_rows(batch, limit=10),
        ["txn_id", "account_id", "amount", "timestamp"],
    ))

    for account in batch.accounts:
        print(f"\n== daily totals: {account.name} ({account.account_id}) ==")
        print(reports.render_report(
            args.format, reports.daily_totals_rows(batch, account), ["date", "total"],
        ))
    return 0


def cmd_statement(args: argparse.Namespace) -> int:
    config = _load_config(args)
    batch = safe_call(import_directory, args.input_dir, config)
    account = next((a for a in batch.accounts if a.account_id == args.account_id), None)
    if account is None:
        print(f"no such account in this batch: {args.account_id}", file=sys.stderr)
        return 1
    txns = [t for t in batch.transactions if t.account_id == args.account_id]
    rows = statements.account_statement_rows(txns, account)
    print(f"== statement: {account.name} ({account.account_id}) ==")
    print(reports.render_report(
        args.format, rows, ["date", "txn_id", "description", "amount", "running_balance"],
    ))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    config = _load_config(args)
    batch = safe_call(import_directory, args.input_dir, config)
    serialization.write_batch_json(batch, args.output)
    print(f"exported {len(batch.accounts)} accounts, {batch.row_count} transactions "
          f"to {args.output}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    config = _load_config(args)
    before = safe_call(import_directory, args.before_dir, config)
    after = safe_call(import_directory, args.after_dir, config)
    result = diff.diff_batches(before, after)

    if not diff.has_changes(result):
        print(f"no changes between {args.before_dir} and {args.after_dir}")
        return 0

    print(f"== diff: {args.before_dir} -> {args.after_dir} ==")
    print(reports.render_report(args.format, diff.diff_rows(result), ["change", "id"]))
    print(f"\nerrors: {result.error_count_before} -> {result.error_count_after}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ledger", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to a JSON config file")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="import a batch and report row/error counts")
    p_import.add_argument("input_dir")
    p_import.set_defaults(func=cmd_import)

    p_list = sub.add_parser("list", help="list transactions, one page at a time")
    p_list.add_argument("input_dir")
    p_list.add_argument("--page", type=int, default=1)
    p_list.add_argument("--page-size", type=int, default=None)
    p_list.set_defaults(func=cmd_list)

    p_agg = sub.add_parser("aggregate", help="category totals across one or more batches")
    p_agg.add_argument("input_dirs", nargs="+")
    p_agg.set_defaults(func=cmd_aggregate)

    p_rec = sub.add_parser("reconcile", help="check each account's computed vs expected balance")
    p_rec.add_argument("input_dir")
    p_rec.set_defaults(func=cmd_reconcile)

    p_rep = sub.add_parser("report", help="full report: accounts, movers, activity, daily totals")
    p_rep.add_argument("input_dir")
    p_rep.add_argument("--format", default="text", choices=["text", "csv", "json"])
    p_rep.set_defaults(func=cmd_report)

    p_stmt = sub.add_parser("statement", help="one account's transactions with a running balance")
    p_stmt.add_argument("input_dir")
    p_stmt.add_argument("account_id")
    p_stmt.add_argument("--format", default="text", choices=["text", "csv", "json"])
    p_stmt.set_defaults(func=cmd_statement)

    p_exp = sub.add_parser("export", help="write an imported batch to a single JSON file")
    p_exp.add_argument("input_dir")
    p_exp.add_argument("output")
    p_exp.set_defaults(func=cmd_export)

    p_diff = sub.add_parser("diff", help="compare two batches by account/transaction id")
    p_diff.add_argument("before_dir")
    p_diff.add_argument("after_dir")
    p_diff.add_argument("--format", default="text", choices=["text", "csv", "json"])
    p_diff.set_defaults(func=cmd_diff)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
