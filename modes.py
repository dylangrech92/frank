"""Mode definitions: each launch of the agent declares exactly one.

A mode fixes the exact tool set placed in the request ``tools`` array (full
schemas, from turn 0) and the instruction block prepended to the system
message. There is no tool discovery: the tools named here, plus whatever
``CONDITIONAL_TOOLS`` the launcher opts into (see below), are the complete
toolset for that mode, and nothing else is ever callable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    """One named operating mode: its instruction block and its fixed tool set."""

    name: str
    instructions: str
    tools: tuple[str, ...]


# Tools that exist in every mode's *reach* but not in any mode's static
# ``tools`` tuple above — membership depends on something outside modes.py
# entirely, so it cannot be baked into a fixed-at-import-time tuple. Today
# this is exactly one tool: ``vision``, gated on whether config.json carries a
# ``vision`` block (tools/vision.py, config.Config.vision). A caller opts a
# name in via ``activate_mode(name, extra_tools=(...))``; leaving extra_tools
# empty (the default) is what keeps a config with no vision block byte-
# identical to a build that never shipped the tool. Recorded here so the
# drift guard (evals/mode_wiring.py's orphan check) knows a name absent from
# every mode.tools tuple is this, deliberately, and not a registered tool no
# mode can ever reach.
CONDITIONAL_TOOLS: frozenset[str] = frozenset({"vision"})


# Tools available in every mode. Memory tools (recall/remember/record/forget)
# are included deliberately even in research: the orientation seed and
# consolidation already write .coding_agent/memory.db unconditionally in every
# mode, so excluding the explicit tools from research would remove the
# model's manual memory channel while the automatic writes continued — a
# false read-only guarantee. "Read-only" means the user's project tree, not
# the agent's own sidecar database.
#
# report_issue is in every mode for the same reason: any mode's tools can fail,
# so the channel for reporting a failure has to exist wherever the failure can
# happen. It writes only to the install-dir issue log, never the project tree.
_COMMON_TOOLS: tuple[str, ...] = (
    "read_file",
    "find",
    "find_files",
    "list_files",
    "find_symbol",
    "recall",
    "remember",
    "record",
    "forget",
    "report_issue",
)

RESEARCH_MODE = """\
[MODE: RESEARCH — READ-ONLY INVESTIGATION]

You are operating in research mode.  Gather information about the codebase and
report findings.  This is strictly read-only.

Tools to use:
- find_symbol to locate a name.  The same tool answers the follow-up question
  about a symbol it found: action="definition" / "references" /
  "implementations" / "type_definition" / "hover".
- call_hierarchy to trace the calls into or out of a function.
- find for content search, find_files for filename search.
- read_file to inspect specific ranges once you have located a symbol.
- document_symbols for file outlines.
- lint, find_dead_code, and get_diagnostics for read-only reports on style
  issues, unreachable code, and compiler/LSP diagnostics.
- spawn_agents to fan out independent research questions concurrently.
- web_search / web_read for external documentation.

Constraints:
- DO NOT create, modify, move, or delete any files.
- DO NOT run tests or execute project code.
- Ground every claim in what you actually observed — cite file paths.

Return a clear, structured natural-language summary of your findings.
[end: MODE]\
"""

CODE_MODE = """\
[MODE: CODE — IMPLEMENT CHANGES]

You are operating in code mode.  Make precise, minimal code changes to satisfy
the task.

Tools to use:
- write_file to write a new file or fully overwrite an existing one (creates
  missing parents), edit_file for a unique targeted search/replace change,
  create_folder for new directories, delete_file to remove files.
- rename_symbol for type-aware cross-file renames.
- code_actions for quick-fixes, organize-imports, and refactors.
- move_file to rename/move files (imports auto-update).
- format after structural changes.
- git for status/diff/branch/commit operations on the project's repository.
- run_command for builds, type checks, and scripts (NOT for running tests) —
  use background=true for long-running commands, with read_output and
  stop_process as its companions to poll output and terminate it.
- verify_scratch for a throwaway snippet outside the project tree to confirm
  your change actually works — use it after every mutation.

Lint and LSP diagnostics surface automatically after each edit — resolve every
diagnostic your edits introduce and state the final diagnostics status.

Constraints:
- DO NOT run unit tests or the test suite (run_tests, pytest).  The orchestrator
  runs tests separately via the test tool.
- Touch only what the task requires — no scope creep, no unrelated refactors.

Return a file-change list: every file you changed with a one-line "what changed
and why" summary.
[end: MODE]\
"""

QA_MODE = """\
[MODE: QA — TEST AND DEBUG]

You are operating in QA mode.  Verify code correctness by running tests and
debugging failures.  This is observe-and-report.

Tools to use:
- run_tests to run the test suite with structured per-test pass/fail output.
- run_command to run specific commands (builds, type checks, scripts) and
  observe their output — use background=true for long-running commands, with
  read_output and stop_process as its companions to poll output and terminate
  it.
- verify_scratch for throwaway verification snippets executed outside the
  project tree.
- If a test fails and the cause is not obvious from the output, use the DAP
  debugger: set_breakpoint / clear_breakpoint, debug_start, debug_control
  (step_over / step_into / step_out / continue), debug_inspect (variables /
  stack / evaluate), debug_stop.

Constraints:
- DO NOT create, modify, or delete any project files.
- Report what you observe — do not fix code here.  If a fix is needed, state
  what should be changed and let the orchestrator delegate to code mode.

Return a clear summary: what passed, what failed, and the failure details (error
messages, stack traces, assertion failures).
[end: MODE]\
"""

VERIFY_MODE = """\
[MODE: VERIFY — DRIVE A REAL BROWSER AND SUBMIT A VERDICT]

You are operating in verify mode.  You drive a REAL browser via Playwright to
verify that a delivered change actually works.  You are given a brief: what
changed, the environment URL and how to reach it, and what must not regress.

Workflow:
1. PLAN FIRST — before acting, write out the concrete assertions you will
   check, derived from the brief.
2. navigate then snapshot are your primary, cheap eyes — an ARIA-style tree
   with element refs.  Reason and act over the snapshot, not a screenshot.
3. Use screenshot ONLY for genuinely visual claims (layout, color, the actual
   rendered text or glyphs) — it attaches the image directly to your context,
   so reserve it for what a snapshot cannot show.  When you report a visual
   attribute (a color, a position, a state), read it off the target element
   itself — never carry over a color or style from a nearby element.
4. Act via refs from the LATEST snapshot — click, fill, press, hover_element,
   select_option, scroll, wait_for, handle_dialog — and RE-SNAPSHOT after any
   action that mutates the DOM: refs go stale the moment the page changes.
5. Gather DETERMINISTIC evidence first: snapshot state, console_logs,
   network_requests statuses, the current URL, http_request for direct API
   cross-checks.  Reach for a screenshot only when the claim is inherently visual.
6. Choose each verdict from the EVIDENCE, not from how the brief is framed:
   'pass' needs evidence you captured THIS run that the behavior is correct;
   'fail' needs evidence that the asserted condition is false — the behavior is
   wrong, OR a claimed element is definitively absent from a page you DID load;
   'inconclusive' is for when you could not gather the evidence to judge at all
   — the environment was unreachable, a page never loaded, or a required signal
   was ungettable.  Never report 'pass' without captured evidence, and never
   report 'fail' for something you could not reach or observe — an unreachable
   or unobservable target is 'inconclusive', regardless of any claim in the
   brief that a change was deployed.
7. When you are done, call report exactly once with your verdict, the plan you
   generated, a per-assertion breakdown with evidence, and any observations.
   A successful report call ENDS the run — it is not a progress update, it is
   the last thing you do.

Constraints:
- DO NOT create, modify, or delete any project files.  You observe a running
  system; you never change one.
- Be skeptical: a plausible-looking screen is not proof — check the actual
  signal behind it.  Ground every claim in a tool result; never invent
  selectors, refs, or results.

Return your verdict through the report tool: that call IS your answer, so
everything the caller needs to know belongs in its fields.
[end: MODE]\
"""

PERFORMANCE_DEBUG_MODE = """\
[MODE: PERFORMANCE_DEBUG — MEASURE AND LOCATE BOTTLENECKS]

You are operating in performance-debug mode.  Measure how the project's code
actually performs, find the bottlenecks, and report where improvements can be
gained.  Measure first — never guess, and never optimize here.

Tools to use:
- profile_command to run any command under resource measurement: wall time,
  user/sys CPU, peak RSS, and a sampled timeline of the whole process tree.
  Use repeats=3 when you need stable wall times.
- profile_hotspots for per-function CPU profiles — self/cumulative time and
  call counts (Python cProfile, Node --cpu-prof, PHP Xdebug).  Use focus= to
  expand one function's callers and callees.
- profile_memory for allocation profiles — top allocation sites and peak
  usage (Python tracemalloc, Node --heap-prof, PHP Xdebug memory events).
- trace_execution (Python) for exact call counts, per-line hit counts in a
  chosen file, and max stack depth — the tool for hidden iteration blow-ups
  and deep recursion.
- run_command for setup or observation commands that need no measurement
  themselves (installing dependencies, checking a process is up) — use
  background=true for long-running commands, with read_output and
  stop_process as its companions to poll output and terminate it.
- read_file / find_symbol to read the code behind every hotspot before you explain it.

Constraints:
- DO NOT modify project files.  Profiled code may itself write files — the
  tool results name any files a run touched; surface them.
- Profile a bounded, realistic workload that exits on its own; state the
  exact command or entry point you measured.
- Ground every claim in measured numbers from a tool result — never claim a
  bottleneck you did not observe.

Return a performance report: what was measured (with wall/CPU/memory numbers),
the ranked hotspots with their numbers, and concrete improvement opportunities
tied to specific files and functions.
[end: MODE]\
"""

MODES: dict[str, Mode] = {
    "research": Mode(
        name="research",
        instructions=RESEARCH_MODE,
        tools=_COMMON_TOOLS + (
            "document_symbols",
            "call_hierarchy",
            "get_diagnostics",
            "lint",
            "find_dead_code",
            "web_search",
            "web_read",
            "spawn_agents",
        ),
    ),
    "code": Mode(
        name="code",
        instructions=CODE_MODE,
        tools=_COMMON_TOOLS + (
            "document_symbols",
            "call_hierarchy",
            "get_diagnostics",
            "lint",
            "find_dead_code",
            "spawn_agents",
            "write_file",
            "edit_file",
            "create_folder",
            "delete_file",
            "move_file",
            "rename_symbol",
            "code_actions",
            "format",
            "git",
            "run_command",
            "read_output",
            "stop_process",
            "verify_scratch",
        ),
    ),
    "qa": Mode(
        name="qa",
        instructions=QA_MODE,
        tools=_COMMON_TOOLS + (
            "document_symbols",
            "call_hierarchy",
            "get_diagnostics",
            "lint",
            "run_tests",
            "run_command",
            "read_output",
            "stop_process",
            "verify_scratch",
            "set_breakpoint",
            "clear_breakpoint",
            "debug_start",
            "debug_control",
            "debug_inspect",
            "debug_stop",
        ),
    ),
    "performance_debug": Mode(
        name="performance_debug",
        instructions=PERFORMANCE_DEBUG_MODE,
        tools=_COMMON_TOOLS + (
            "document_symbols",
            "call_hierarchy",
            "profile_command",
            "profile_hotspots",
            "profile_memory",
            "trace_execution",
            "run_command",
            "read_output",
            "stop_process",
        ),
    ),
    # verify is the only mode without the LSP cluster: it verifies a RUNNING
    # system through a browser, not source code. It is also the only mode with a
    # terminal tool — a successful `report` ends the turn (see turn/outcome.py),
    # which is what makes the evidence gate binding rather than advisory.
    "verify": Mode(
        name="verify",
        instructions=VERIFY_MODE,
        tools=_COMMON_TOOLS + (
            "navigate",
            "snapshot",
            "click",
            "fill",
            "press",
            "hover_element",
            "select_option",
            "scroll",
            "wait_for",
            "screenshot",
            "console_logs",
            "network_requests",
            "http_request",
            "handle_dialog",
            "report",
        ),
    ),
}


def mode_names() -> list[str]:
    """Return the sorted mode names — the source of argparse's ``--mode`` choices."""
    return sorted(MODES)


def get_mode(name: str) -> Mode:
    """Return the ``Mode`` registered under *name*.

    Args:
        name: A key of ``MODES``.

    Returns:
        The matching ``Mode``.

    Raises:
        ValueError: If *name* is not a known mode, naming the valid choices.
    """
    try:
        return MODES[name]
    except KeyError:
        raise ValueError(
            f"unknown mode {name!r}; valid modes: {', '.join(mode_names())}"
        ) from None
