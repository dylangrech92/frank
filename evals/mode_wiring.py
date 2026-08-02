"""Mode-gating wiring and drift-guard check (no LLM required).

Replaces ``activate_tools_wiring.py`` (deleted): that file tested
``main._activate_cli_tools``, ``mcp_server.MODE_INSTRUCTIONS`` /
``MODE_TOOLS``, and the ``--activate-tools`` flag — all deleted in the move
from catalog-gated tool discovery to a mandatory ``--mode`` flag whose tool
set is static and complete per mode (see ``modes.py``).

This script verifies the mode-gating harness end to end, with no mocks:

  a. Launching without ``--mode`` fails fast, before any LLM call.
  b. Launching with an unknown ``--mode`` value fails fast.
  c. ``activate_mode(m)`` then ``schemas()`` is exactly mode *m*'s declared
     tool set for every mode — no extras, no duplicates, every name resolves
     in the real registry.
  d. ``dispatch()`` of a tool outside the active mode returns
     ``code="not-in-mode"`` AND never executes — proven by a real side
     effect (a file) that must NOT appear on disk.
  e. Drift guard between ``modes.py``'s prose and its declared tool tuples,
     plus ``mcp_server.py``'s five literal mode strings against ``MODES``.
     See ``check_drift_guard`` for why this is two separate, asymmetric
     assertions rather than one "every tool name in the prose" scan.
  f. ``tools/spawn_agents.py`` launches each child with ``--mode`` set to
     the PARENT's own active mode (the escalation-hole fix) — proven with a
     real subprocess, not assumed.
  g. Prints each mode's approximate schema token cost (informational only).
  h. Every live (non-inline) scenario in ``evals/scenarios.py`` builds a
     ``--mode <valid-mode>`` argv via ``evals/run.py``'s real
     ``build_live_cmd`` — the fix for the "run_live_scenario launched
     main.py with no --mode" gap. Checked for both the smoke-stub and the
     real-``main.py`` branch, by calling the real function, not a copy.

Exits 0 on success, prints ``FAIL: <reason>`` to stderr and exits 1 otherwise.
Runs with the repo root on ``sys.path`` (evals/run.py inserts it before
exec'ing this file); it also adds the repo root itself if not already
present, so the script can be invoked directly with
``.venv/bin/python evals/mode_wiring.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time


# Ensure the repo root is on sys.path so top-level imports (main, agent,
# modes, mcp_server, tools.registry) resolve when this script is invoked
# directly. evals/run.py already sets PYTHONPATH for inline scenarios, but we
# also add it here for standalone invocation.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _section(text: str, start_marker: str, end_marker: str | None) -> str:
    """Return the slice of *text* from *start_marker* up to (excluding) *end_marker*.

    Returns "" if *start_marker* is absent; returns the rest of the text if
    *end_marker* is absent or None.
    """
    start = text.find(start_marker)
    if start == -1:
        return ""
    if end_marker is None:
        return text[start:]
    end = text.find(end_marker, start)
    return text[start:] if end == -1 else text[start:end]


def _mentioned_tools(section_text: str, universe: set[str]) -> set[str]:
    """Return every name in *universe* that appears as a whole word in *section_text*.

    Word-boundary matching means "find" does not match inside "find_symbol"
    (underscore is a word character in Python's regex ``\\b``), so this
    correctly separates a bare tool name from a longer tool name that merely
    contains it as a prefix.
    """
    return {name for name in universe if re.search(rf"\b{re.escape(name)}\b", section_text)}


def check_missing_mode_fails_fast() -> list[str]:
    """a. Launching without --mode fails fast, before any LLM call.

    Verifies:
    - Nonzero exit
    - stderr names '--mode'
    - Fails in well under 5s (i.e. at argparse, not after attempting a
      network round-trip)
    """
    failures: list[str] = []
    try:
        main_py = os.path.join(_REPO_ROOT, "main.py")
        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, main_py, "-p", "x"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_REPO_ROOT,
        )
        elapsed = time.monotonic() - start

        if proc.returncode == 0:
            failures.append(f"missing --mode: expected nonzero exit, got 0 (stderr: {proc.stderr[:300]!r})")
        if "--mode" not in proc.stderr:
            failures.append(f"missing --mode: stderr does not name '--mode': {proc.stderr[:500]!r}")
        if elapsed > 5:
            failures.append(f"missing --mode: took {elapsed:.2f}s (expected <5s, i.e. fail before any LLM call)")
    except subprocess.TimeoutExpired:
        failures.append("missing --mode: subprocess timed out after 10s (should fail fast)")
    except Exception as exc:
        failures.append(f"Exception in check_missing_mode_fails_fast: {type(exc).__name__}: {exc}")

    return failures


def check_bogus_mode_fails_fast() -> list[str]:
    """b. Launching with an unknown --mode value fails fast.

    argparse's ``choices=modes.mode_names()`` should reject it before any
    config load or LLM call.
    """
    failures: list[str] = []
    try:
        main_py = os.path.join(_REPO_ROOT, "main.py")
        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, main_py, "-p", "x", "--mode", "bogus_mode_xyz"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_REPO_ROOT,
        )
        elapsed = time.monotonic() - start

        if proc.returncode == 0:
            failures.append(f"--mode bogus_mode_xyz: expected nonzero exit, got 0 (stderr: {proc.stderr[:300]!r})")
        if elapsed > 5:
            failures.append(f"--mode bogus_mode_xyz: took {elapsed:.2f}s (expected <5s)")
    except subprocess.TimeoutExpired:
        failures.append("--mode bogus_mode_xyz: subprocess timed out after 10s (should fail fast)")
    except Exception as exc:
        failures.append(f"Exception in check_bogus_mode_fails_fast: {type(exc).__name__}: {exc}")

    return failures


def check_per_mode_schemas_exact() -> list[str]:
    """c. activate_mode(m) -> schemas() is exactly mode m's declared tool set.

    For every mode in modes.MODES:
    - schemas() has no duplicate tool names
    - the set of schema names equals the set of mode.tools exactly (no
      extras, nothing missing)
    - every tool the mode declares resolves to a real registered Tool
    """
    failures: list[str] = []
    try:
        from modes import MODES
        import tools.registry as registry

        registry.discover()

        for mode_name, mode in MODES.items():
            registry.activate_mode(mode_name)
            schema_names = [s["function"]["name"] for s in registry.schemas()]

            if len(schema_names) != len(set(schema_names)):
                dupes = sorted({n for n in schema_names if schema_names.count(n) > 1})
                failures.append(f"mode {mode_name!r}: schemas() has duplicate name(s): {dupes}")

            declared = set(mode.tools)
            actual = set(schema_names)
            if declared != actual:
                missing = declared - actual
                extra = actual - declared
                failures.append(
                    f"mode {mode_name!r}: schemas() mismatch vs modes.MODES — "
                    f"missing {sorted(missing)}, extra {sorted(extra)}"
                )

            for tool_name in mode.tools:
                if registry.get_tool(tool_name) is None:
                    failures.append(f"mode {mode_name!r}: declared tool {tool_name!r} does not resolve in the registry")

    except Exception as exc:
        failures.append(f"Exception in check_per_mode_schemas_exact: {type(exc).__name__}: {exc}")

    return failures


def check_dispatch_out_of_mode_blocked() -> list[str]:
    """d. dispatch() of an out-of-mode tool returns not-in-mode and never executes.

    Activates 'research' (strictly read-only: no file-writing tool present)
    then dispatches 'write_file', which belongs to 'code'/'qa' but not
    'research'. Asserts both the error contract AND the real side effect
    (the file) does not exist on disk — the gate must block execution, not
    just mislabel a result after the fact.
    """
    import tempfile

    failures: list[str] = []
    try:
        import tools.registry as registry

        saved_mode = registry.current_mode()
        registry.activate_mode("research")

        tmp = tempfile.mkdtemp(prefix="mode-wiring-dispatch-")
        marker_name = "should_not_exist.txt"
        marker_path = os.path.join(tmp, marker_name)

        result = registry.dispatch("write_file", {"path": marker_name, "contents": "x"})

        if result.status != "error" or result.code != "not-in-mode":
            failures.append(
                f"out-of-mode dispatch: expected status='error' code='not-in-mode', "
                f"got status={result.status!r} code={getattr(result, 'code', None)!r}"
            )
        if os.path.exists(marker_path):
            failures.append("out-of-mode dispatch executed anyway — the file was created on disk")

        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    except Exception as exc:
        failures.append(f"Exception in check_dispatch_out_of_mode_blocked: {type(exc).__name__}: {exc}")

    return failures


def check_drift_guard() -> list[str]:
    """e. Prose/tool-list drift guard — two asymmetric assertions, not one scan.

    A naive "every tool name anywhere in a mode's prose must be in that
    mode's tool list" check produces a false positive: CODE_MODE's
    Constraints section deliberately names 'run_tests' as something NOT to
    use ("DO NOT run unit tests or the test suite (run_tests, pytest)."),
    and 'run_tests' is correctly absent from code mode's tool list — that
    mention is the point, not a bug.

    So this check is scoped in two directions that do NOT overlap:

    1. POSITIVE (recommended-in-prose implies declared): scan only the
       "Tools to use:" section — or, for a mode that structures its guidance
       as an ordered procedure instead of a bullet list, "Workflow:" —
       stopping before "Constraints:". Every tool named there must be in the
       mode's declared tool tuple. This section never negates a tool, so a
       straight subset check is safe here.

    2. INVERSE, the more valuable half (forbidden-in-prose implies absent):
       scan only the "Constraints:" section. Any tool named there as
       forbidden must NOT be in the mode's declared tool tuple. Concretely:
       'run_tests' must not be in code mode's tools. This is the actual
       regression guard — it is what stops the prose promising one thing
       while the harness silently allows another, which is exactly the
       contradiction this whole mode-gating change closes (previously the
       prose forbade running tests in code mode, but the flat catalog made
       run_tests callable anyway).

       Known limitation of the inverse half: several tools are named after
       ordinary English words ('report', 'format', 'find', 'git', 'lint',
       'snapshot', 'click', 'press'), and the scan cannot tell a tool
       reference from the plain verb. A Constraints bullet that uses one of
       those words as prose therefore asserts "this mode must not carry that
       tool" — true today for every mode, but by coincidence, not by
       intent. If this half ever fires on a mode that genuinely should carry
       the named tool, the fix is to reword the prose, not to relax the
       check: the check is the only thing holding prose and tool list
       together.

    Also checks, as the same "does the static harness config match
    modes.py truth" drift class:
    - every registered tool belongs to at least one mode (no orphans)
    - no mode declares an unknown tool name
    - mcp_server.py's five literal mode strings (passed positionally to
      _run_agent(...), read via AST since mcp_server.py deliberately does
      not import modes.py) are exactly {'research','code','qa',
      'performance_debug','verify'} and each is a valid modes.MODES key — this is
      the only thing standing between a renamed mode and a silently broken
      MCP tool.
    """
    import ast

    failures: list[str] = []
    try:
        from modes import MODES
        import tools.registry as registry

        registry.discover()
        all_tool_names = set(registry._registry)

        covered: set[str] = set()
        for mode_name, mode in MODES.items():
            # A mode names its recommended tools under one of two headers: a
            # "Tools to use:" bullet list, or verify's "Workflow:" numbered
            # procedure. Neither section ever negates a tool, so both feed the
            # positive half. Absence of BOTH is reported rather than skipped —
            # a section this scan cannot find is a check that silently passes,
            # which is the one failure mode a drift guard must not have.
            tools_section = _section(mode.instructions, "Tools to use:", "Constraints:")
            if not tools_section:
                tools_section = _section(mode.instructions, "Workflow:", "Constraints:")
            constraints_section = _section(mode.instructions, "Constraints:", "\nReturn ")
            if not tools_section:
                failures.append(
                    f"mode {mode_name!r}: no 'Tools to use:' or 'Workflow:' section — "
                    f"the drift guard's positive half has nothing to check"
                )
            if not constraints_section:
                failures.append(
                    f"mode {mode_name!r}: no 'Constraints:' section — the drift "
                    f"guard's inverse half has nothing to check"
                )

            recommended = _mentioned_tools(tools_section, all_tool_names)
            forbidden = _mentioned_tools(constraints_section, all_tool_names)

            not_declared = recommended - set(mode.tools)
            if not_declared:
                failures.append(
                    f"mode {mode_name!r}: 'Tools to use:' recommends {sorted(not_declared)} "
                    f"but they are absent from mode.tools"
                )

            wrongly_present = forbidden & set(mode.tools)
            if wrongly_present:
                failures.append(
                    f"mode {mode_name!r}: 'Constraints:' forbids {sorted(wrongly_present)} "
                    f"but they are present in mode.tools (prose/tool-list contradiction)"
                )

            unknown = [t for t in mode.tools if t not in all_tool_names]
            if unknown:
                failures.append(f"mode {mode_name!r} declares unknown tool(s): {unknown}")

            covered.update(mode.tools)

        orphans = all_tool_names - covered
        if orphans:
            failures.append(f"tool(s) registered but not reachable from any mode: {sorted(orphans)}")

        # mcp_server.py's literal mode strings, read via AST (that file does
        # not import modes.py on purpose, so this is a real drift check, not
        # a tautology against a shared import).
        mcp_server_path = os.path.join(_REPO_ROOT, "mcp_server.py")
        with open(mcp_server_path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=mcp_server_path)

        mcp_mode_calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if func_name == "_run_agent" and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        mcp_mode_calls.add(first.value)

        expected = {"research", "code", "qa", "performance_debug", "verify"}
        if mcp_mode_calls != expected:
            failures.append(
                f"mcp_server.py's _run_agent call sites name {sorted(mcp_mode_calls)}, "
                f"expected exactly {sorted(expected)}"
            )
        for m in mcp_mode_calls:
            if m not in MODES:
                failures.append(f"mcp_server.py calls _run_agent with mode {m!r}, not a key of modes.MODES")

    except Exception as exc:
        failures.append(f"Exception in check_drift_guard: {type(exc).__name__}: {exc}")

    return failures


def check_spawn_agents_propagates_parent_mode() -> list[str]:
    """f. spawn_agents launches each child with --mode set to the PARENT's own mode.

    This is the escalation-hole fix in tools/spawn_agents.py: a read-only
    'research' parent must not be able to spawn an edit-capable child. Proven
    with a real subprocess, not assumed — we cannot intercept subprocess.run's
    argv without mocking it (forbidden), so we activate 'research' (the most
    restrictive mode) in this process, call the real _spawn_one() helper
    exactly as SpawnAgents.run() calls it internally (same
    registry.current_mode() read, same argv construction), and read back the
    CHILD's own real 'mode: <name>' startup telemetry line
    (main.py prints this right after its own activate_mode() succeeds) from
    its actual, unmocked stderr.

    The child is given an empty task over stdin, so main.py's own
    "one-shot task is empty" fast-exit path fires immediately after printing
    the mode line — no network-reachable LLM endpoint is needed and the
    check stays fast and deterministic.
    """
    import json as json_mod
    import tempfile
    from pathlib import Path

    failures: list[str] = []
    saved_mode = None
    try:
        import tools.registry as registry
        from tools.spawn_agents import _spawn_one

        saved_mode = registry.current_mode()
        registry.activate_mode("research")
        mode = registry.current_mode()  # exactly what SpawnAgents.run() reads
        # activate_mode() just set this in-process, so it cannot be None here
        # — a real check (activate_mode silently failing would trip it), and
        # it narrows str | None to str for _spawn_one's required str param
        # rather than loosening that param to Optional on the production side.
        assert mode is not None, "registry.current_mode() is None right after activate_mode('research')"

        tmp = tempfile.mkdtemp(prefix="mode-wiring-spawn-")
        config_path = os.path.join(tmp, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            # Syntactically valid but never dialed: the child exits via the
            # empty-task path before any LLM call is attempted.
            json_mod.dump({"llm": {"base_url": "http://127.0.0.1:1/v1", "api_key": "x", "model": "x"}}, f)

        main_py = Path(_REPO_ROOT) / "main.py"
        outcome = _spawn_one(
            main_py,
            {"prompt": "", "cwd": tmp},
            dict(os.environ, CODING_AGENT_DEPTH="1"),
            15.0,
            mode,
        )

        stderr = outcome.get("stderr") or ""
        if f"mode: {mode}" not in stderr:
            failures.append(
                f"child stderr does not contain 'mode: {mode}' — the parent's active "
                f"mode was not propagated into the child's --mode argv "
                f"(exit_code={outcome.get('exit_code')!r} timed_out={outcome.get('timed_out')!r}). "
                f"stderr: {stderr[:1000]!r}"
            )
        for other_mode in ("code", "qa", "performance_debug", "verify"):
            if f"mode: {other_mode}" in stderr:
                failures.append(f"child ran under mode {other_mode!r}, not the parent's {mode!r}")

    except Exception as exc:
        failures.append(f"Exception in check_spawn_agents_propagates_parent_mode: {type(exc).__name__}: {exc}")
    finally:
        if saved_mode is not None:
            try:
                import tools.registry as registry
                registry.activate_mode(saved_mode)
            except Exception:
                pass

    return failures


def check_print_mode_token_costs() -> list[str]:
    """g. Print each mode's approximate schema token cost (informational only).

    len(json.dumps(schemas())) // 4 is the same rough chars-per-token
    estimate used elsewhere in this repo's token-budget code — good enough
    to spot a mode whose schema bloat silently doubled, not a billing figure.
    Always returns no failures; this check cannot fail on its own.
    """
    import json as json_mod

    try:
        from modes import MODES
        import tools.registry as registry

        registry.discover()
        for mode_name in MODES:
            registry.activate_mode(mode_name)
            schema_json = json_mod.dumps(registry.schemas())
            approx_tokens = len(schema_json) // 4
            print(f"mode {mode_name!r}: {len(MODES[mode_name].tools)} tools, ~{approx_tokens} schema tokens")
    except Exception as exc:
        print(f"check_print_mode_token_costs: {type(exc).__name__}: {exc}", file=sys.stderr)

    return []


def check_live_scenarios_declare_valid_mode() -> list[str]:
    """h. Every live scenario's constructed argv carries a valid --mode.

    Fix target: ``evals/run.py``'s ``run_live_scenario()`` used to build
    ``cmd`` inline with no ``--mode`` at all, so every live (real-LLM)
    scenario exited 2 before the agent ever started, and ``run.py`` itself
    reported that as a plain scenario failure — a silent, total loss of the
    live eval capability that read like a scoring bug.

    This calls the REAL ``evals/run.py:build_live_cmd`` (not a reimplemented
    copy) for every live scenario in the REAL ``evals/scenarios.py``, for
    both the smoke-stub branch and the real-``main.py`` branch, and asserts
    the resulting argv contains ``--mode`` immediately followed by a value in
    ``modes.mode_names()``. Import of ``evals.run`` also re-runs its own
    ``_validate_scenarios`` gate as a side effect, which is exactly the
    complementary "fail loudly if a live scenario has no mode key at all"
    half of this same fix.
    """
    import tempfile
    from pathlib import Path

    failures: list[str] = []
    try:
        import run as eval_run
        from scenarios import SCENARIOS
        from modes import mode_names

        valid_modes = set(mode_names())
        live_scenarios = [s for s in SCENARIOS if not s.get("inline")]

        if not live_scenarios:
            failures.append("no live (non-inline) scenarios found — cannot check argv construction")

        fake_cfg_path = Path(tempfile.gettempdir()) / "mode-wiring-fake-config.json"

        for scenario in live_scenarios:
            for smoke in (True, False):
                cmd = eval_run.build_live_cmd(scenario, fake_cfg_path, smoke)
                branch = "smoke" if smoke else "live"
                if "--mode" not in cmd:
                    failures.append(
                        f"scenario {scenario['name']!r} ({branch} branch): argv has no '--mode' flag: {cmd}"
                    )
                    continue
                idx = cmd.index("--mode")
                if idx + 1 >= len(cmd):
                    failures.append(f"scenario {scenario['name']!r} ({branch} branch): '--mode' has no value: {cmd}")
                    continue
                mode_value = cmd[idx + 1]
                if mode_value not in valid_modes:
                    failures.append(
                        f"scenario {scenario['name']!r} ({branch} branch): --mode value {mode_value!r} "
                        f"not in modes.mode_names() {sorted(valid_modes)}: {cmd}"
                    )

    except Exception as exc:
        failures.append(f"Exception in check_live_scenarios_declare_valid_mode: {type(exc).__name__}: {exc}")

    return failures


def main() -> int:
    all_failures: list[str] = []

    checks = [
        ("missing-mode-fails-fast", check_missing_mode_fails_fast),
        ("bogus-mode-fails-fast", check_bogus_mode_fails_fast),
        ("per-mode-schemas-exact", check_per_mode_schemas_exact),
        ("dispatch-out-of-mode-blocked", check_dispatch_out_of_mode_blocked),
        ("drift-guard", check_drift_guard),
        ("spawn-agents-propagates-parent-mode", check_spawn_agents_propagates_parent_mode),
        ("mode-token-costs", check_print_mode_token_costs),
        ("live-scenarios-declare-valid-mode", check_live_scenarios_declare_valid_mode),
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

    print(
        "PASS: mode-gating wiring is correct (argv gate, schemas, dispatch gate, "
        "drift guard, spawn_agents propagation, live-scenario argv)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
