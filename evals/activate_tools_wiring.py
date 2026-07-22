"""--activate-tools CLI wiring and MCP mode→tools mapping check (no LLM).

This script verifies two things:

1. The ``--activate-tools`` CLI flag works correctly:
   - Valid tool names are activated and appear in the schemas array
   - Invalid tool names cause a ValueError (in-process) and a nonzero exit
     with the tool name in stderr (subprocess gate)
   - The activation is idempotent (calling twice doesn't duplicate entries)

2. The MCP mode→tools mapping is consistent:
   - Every mode in MODE_TOOLS has a corresponding MODE_INSTRUCTIONS entry
   - Every tool name in MODE_TOOLS values exists in the registry
   - Every profiling tool mentioned in the performance_debug instruction
     text exists in the registry

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before exec'ing
this file); it also adds the repo root itself if not already present, so the
script can be invoked directly with ``.venv/bin/python evals/activate_tools_wiring.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


# Ensure the repo root is on sys.path so top-level imports (main, agent, mcp_server)
# resolve when this script is invoked directly. evals/run.py already sets PYTHONPATH
# for inline scenarios, but we also add it here for standalone invocation.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def check_1_activate_cli_tools() -> list[str]:
    """Check that _activate_cli_tools works correctly for valid names.

    Verifies:
    - Four profiling tools can be activated
    - They appear exactly once in the schemas array
    - Activating twice is idempotent (still exactly once each)
    """
    failures: list[str] = []

    try:
        from main import _activate_cli_tools
        import tools.registry as registry

        registry.discover()

        # First activation
        activated = _activate_cli_tools("profile_command,profile_hotspots,profile_memory,trace_execution")
        if len(activated) != 4:
            failures.append(f"Expected 4 activated tools, got {len(activated)}: {activated}")
            return failures

        schemas = registry.schemas()
        schema_names = [s["function"]["name"] for s in schemas]

        for tool_name in ["profile_command", "profile_hotspots", "profile_memory", "trace_execution"]:
            count = schema_names.count(tool_name)
            if count != 1:
                failures.append(f"Tool {tool_name!r} appears {count} times in schemas (expected 1)")

        # Second activation (idempotence check)
        _activate_cli_tools("profile_command,profile_hotspots,profile_memory,trace_execution")
        schemas = registry.schemas()
        schema_names = [s["function"]["name"] for s in schemas]

        for tool_name in ["profile_command", "profile_hotspots", "profile_memory", "trace_execution"]:
            count = schema_names.count(tool_name)
            if count != 1:
                failures.append(f"Tool {tool_name!r} appears {count} times after second activation (expected 1)")

    except Exception as exc:
        failures.append(f"Exception in check_1: {type(exc).__name__}: {exc}")

    return failures


def check_2_unknown_name_contract() -> list[str]:
    """Check that _activate_cli_tools raises ValueError for unknown tool names.

    Verifies:
    - Calling with a bogus tool name raises ValueError
    - The error message mentions the unknown tool name
    """
    failures: list[str] = []

    try:
        from main import _activate_cli_tools

        try:
            _activate_cli_tools("bogus_tool_name")
            failures.append("Expected ValueError for unknown tool name, but none was raised")
        except ValueError as exc:
            if "bogus_tool_name" not in str(exc):
                failures.append(f"ValueError message does not mention 'bogus_tool_name': {exc}")

    except Exception as exc:
        failures.append(f"Exception in check_2: {type(exc).__name__}: {exc}")

    return failures


def check_3_subprocess_gate() -> list[str]:
    """Check that the subprocess gate rejects unknown tool names.

    Verifies:
    - Running main.py with --activate-tools and a bogus tool name exits nonzero
    - stderr mentions the unknown tool name
    - The failure happens fast (before any LLM call)
    """
    failures: list[str] = []

    try:
        main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main.py")
        main_py = os.path.abspath(main_py)

        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, main_py, "-p", "x", "--activate-tools", "bogus_tool_name"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        elapsed = time.monotonic() - start

        if proc.returncode == 0:
            failures.append("Expected nonzero exit code, but got 0")
        else:
            if "bogus_tool_name" not in proc.stderr:
                failures.append(f"stderr does not mention 'bogus_tool_name': {proc.stderr[:500]}")

        if elapsed > 5:
            failures.append(f"Subprocess took {elapsed:.2f}s (expected <5s to fail before LLM call)")

    except subprocess.TimeoutExpired:
        failures.append("Subprocess timed out after 10s (should fail fast)")
    except Exception as exc:
        failures.append(f"Exception in check_3: {type(exc).__name__}: {exc}")

    return failures


def check_4_drift_guard() -> list[str]:
    """Check that MODE_INSTRUCTIONS and MODE_TOOLS are consistent.

    Verifies:
    - Every MODE_TOOLS key is in MODE_INSTRUCTIONS
    - Every tool name in MODE_TOOLS values exists in the registry
    - Every profiling tool mentioned in the performance_debug instruction
      text exists in the registry
    """
    failures: list[str] = []

    try:
        from mcp_server import MODE_INSTRUCTIONS, MODE_TOOLS
        import tools.registry as registry

        registry.discover()

        # Check that every MODE_TOOLS key is in MODE_INSTRUCTIONS
        for mode in MODE_TOOLS:
            if mode not in MODE_INSTRUCTIONS:
                failures.append(f"MODE_TOOLS has key {mode!r} but MODE_INSTRUCTIONS does not")

        # Check that every tool name in MODE_TOOLS values exists in the registry
        for mode, tools in MODE_TOOLS.items():
            for tool_name in tools:
                if tool_name not in registry._registry:
                    failures.append(f"Tool {tool_name!r} in MODE_TOOLS[{mode!r}] not found in registry")

        # Check that every profiling tool mentioned in the performance_debug
        # instruction text exists in the registry
        perf_debug_text = MODE_INSTRUCTIONS.get("performance_debug", "")
        profiling_tools = ["profile_command", "profile_hotspots", "profile_memory", "trace_execution"]

        for tool_name in profiling_tools:
            if tool_name not in perf_debug_text:
                failures.append(f"Tool {tool_name!r} not mentioned in performance_debug instruction text")
            elif tool_name not in registry._registry:
                failures.append(f"Tool {tool_name!r} mentioned in performance_debug text but not in registry")

    except Exception as exc:
        failures.append(f"Exception in check_4: {type(exc).__name__}: {exc}")

    return failures


def check_5_repeat_cap_exemption() -> list[str]:
    """Check that all four profiling tools are in the repeat-cap exemption set.

    Verifies:
    - The frozenset from agent (union of verification and profiling tools)
      contains all four profiling tool names
    """
    failures: list[str] = []

    try:
        import agent

        profiling_tools = {"profile_command", "profile_hotspots", "profile_memory", "trace_execution"}
        exemption_set = agent._REPEAT_CAP_EXEMPT

        for tool_name in profiling_tools:
            if tool_name not in exemption_set:
                failures.append(f"Tool {tool_name!r} not in _REPEAT_CAP_EXEMPT")

    except Exception as exc:
        failures.append(f"Exception in check_5: {type(exc).__name__}: {exc}")

    return failures


def main() -> int:
    all_failures: list[str] = []

    checks = [
        ("activate-cli-tools", check_1_activate_cli_tools),
        ("unknown-name-contract", check_2_unknown_name_contract),
        ("subprocess-gate", check_3_subprocess_gate),
        ("drift-guard", check_4_drift_guard),
        ("repeat-cap-exemption", check_5_repeat_cap_exemption),
    ]

    for check_name, check_fn in checks:
        try:
            failures = check_fn()
        except Exception as exc:
            failures = [f"{check_name} raised {type(exc).__name__}: {exc}"]

        for failure in failures:
            print(f"FAIL [{check_name}]: {failure}", file=sys.stderr)
            all_failures.append(failure)

    if all_failures:
        return 1

    print("PASS: --activate-tools CLI wiring and MCP mode→tools mapping are correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
