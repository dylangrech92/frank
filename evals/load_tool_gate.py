"""Dispatch-level check of the deferred-tool-loading gate (no LLM required).

Replaces the old live `load_tool_gate` scenario: that one relied on the model
calling a gated tool *before* loading it so the rejection fired — read_file
made that reliable because models reach for it instinctively, but read_file is
now PINNED, and a protocol-following model loads first, so the rejection path
never appears on a live transcript. The gate is a dispatch-layer contract, so
it is asserted here deterministically instead:

    schemas()    — exposes exactly the PINNED set before any load_tool call.
    gated tool   — dispatch is rejected with code=not-loaded until loaded,
                   then executes (any post-load failure must be a different
                   code, e.g. missing-arguments — never the gate's).
    pinned tool  — dispatches immediately with no load_tool call, and
                   load_tool on it returns benign success, not an error.

Exits 0 on success, 1 on any assertion failure. Runs with the repo root on
``sys.path`` (evals/run.py inserts it before exec'ing this file), and touches
no repo files.
"""

from __future__ import annotations

import sys


def main() -> int:
    from tools.registry import PINNED, discover, dispatch, schemas

    discover()
    failures: list[str] = []

    exposed = {t["function"]["name"] for t in schemas()}
    if exposed != set(PINNED):
        failures.append(
            f"schemas() exposes {sorted(exposed)}, expected exactly PINNED {sorted(PINNED)}"
        )

    gated = dispatch("get_diagnostics", {})
    if gated.status != "error" or gated.code != "not-loaded":
        failures.append(
            f"unloaded tool must be rejected with code=not-loaded, got "
            f"status={gated.status!r} code={gated.code!r}"
        )

    loaded = dispatch("load_tool", {"name": "get_diagnostics"})
    if loaded.status != "success":
        failures.append(f"load_tool(get_diagnostics) failed: {loaded.body!r}")

    after = dispatch("get_diagnostics", {})
    if after.code == "not-loaded":
        failures.append("get_diagnostics is still gated after load_tool")

    pinned_call = dispatch("list_files", {})
    if pinned_call.code == "not-loaded":
        failures.append("pinned tool list_files was rejected by the gate")

    benign = dispatch("load_tool", {"name": "read_file"})
    if benign.status != "success":
        failures.append(f"load_tool on a pinned tool must be benign, got: {benign.body!r}")

    for f in failures:
        print(f"FAIL: {f}")
    if not failures:
        print("load_tool_gate inline checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
