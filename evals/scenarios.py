"""Declarative eval scenarios replaying proven live-agent behavior.

Each entry in ``SCENARIOS`` is a plain dict:

    name:        unique scenario id (also used for --only substring matching
                 and for the output/result filenames).
    description: one-line human summary, printed in --list output.
    turns:       list of single-line prompt strings piped to the agent's
                 stdin, one per line, in order (EOF after the last one).
    config:      'default' to run unmodified against the repo's config.json,
                 or a dict deep-merged on top of it (e.g. to shrink
                 llm.context_limit so compaction/oversize paths trigger).
    setup:       dict of relative-path -> file content to materialize inside
                 a fresh temp project dir before the run. The special key
                 'bigfile_bytes' generates a ~N-byte python-ish text file
                 named bigfile.py instead of a literal-content entry.
    checks:      list of assertion dicts evaluated over the captured run.
                 Every check has a 'kind' plus kind-specific keys:
                   - 'regex-present': {'stream': 'stdout'|'stderr', 'pattern': str}
                       Fails unless `pattern` matches somewhere on `stream`.
                   - 'regex-absent':  {'stream': ..., 'pattern': str}
                       Fails if `pattern` matches anywhere on `stream`.
                   - 'ordered':       {'stream': ..., 'patterns': [str, ...]}
                       Fails unless each pattern in order is found starting
                       search after the previous match's end.
                   - 'regex-note':    {'stream': ..., 'pattern': str}
                       Informational only — never fails; reported in the
                       results table as NOTE-PRESENT / NOTE-ABSENT.
                   - 'file-lines-min': {'path': str, 'min': int}
                       Fails unless the file (relative to the temp project
                       dir) exists and has at least `min` non-empty-file
                       lines (counts newlines; a file must have >= min
                       lines to pass).
    inline:      (mutually exclusive with turns/config/setup) path to a
                 python script (relative to the evals/ dir) that is run
                 directly with the current interpreter instead of spawning
                 main.py; it must exit 0 to pass and non-zero to fail. Used
                 for dispatch-level checks that need no live LLM.
"""

from __future__ import annotations

MATHLIB_PY = '''"""Small math helper module used as eval fixture content."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def mean(values):
    """Return the arithmetic mean of values.

    Raises:
        ValueError: if values is empty.
    """
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)


def is_even(n):
    """Return True if n is even."""
    return n % 2 == 0
'''

SCENARIOS: list[dict] = [
    {
        "name": "load_tool_gate",
        "description": (
            "dispatch-level gate contract: a catalog tool is rejected with "
            "code=not-loaded until load_tool activates it; PINNED tools are "
            "callable immediately and load_tool on one is benign (inline — "
            "a protocol-following model loads before calling, so the "
            "rejection never appears on a live transcript)."
        ),
        "inline": "load_tool_gate.py",
    },
    {
        "name": "json_multiline",
        "description": (
            "Multi-line file content in a tool call survives JSON parsing "
            "without a malformed-json error."
        ),
        "turns": [
            'Create a file named haiku.txt containing a three-line haiku '
            'about compilers, each line on its own line, followed by a '
            'blank line and the attribution "- qwen". Then read it back '
            "to confirm the contents.",
        ],
        "config": "default",
        "setup": {},
        "checks": [
            {"stream": "stderr", "kind": "regex-absent", "pattern": r"malformed-json"},
            {"stream": "stdout", "kind": "regex-present", "pattern": r"qwen"},
            {"kind": "file-lines-min", "path": "haiku.txt", "min": 4},
        ],
    },
    {
        "name": "oversize_central",
        "description": (
            "run_command output too large for the remaining budget is "
            "discarded centrally with a result-too-large error."
        ),
        "turns": [
            "Use the run_command tool to run exactly this shell command: "
            "cat bigfile.py — then tell me what happened.",
        ],
        "config": {
            "llm": {"context_limit": 9000},
            "compaction": {"reserve_ratio": 0.1, "reserve_min_tokens": 500},
        },
        "setup": {"bigfile_bytes": 130_000},
        "checks": [
            {"stream": "stderr", "kind": "regex-present", "pattern": r"code=result-too-large"},
            {
                "stream": "stderr",
                "kind": "regex-present",
                "pattern": r"result is too large to fit in context",
            },
        ],
    },
    {
        "name": "oversize_readfile_paging",
        "description": (
            "read_file on an oversize file is rejected with "
            "code=file-too-large and the model pages via start_line."
        ),
        "turns": [
            "Read bigfile.py in full and tell me what the module does "
            "overall in 2-3 sentences.",
        ],
        "config": {
            "llm": {"context_limit": 9000},
            "compaction": {"reserve_ratio": 0.1, "reserve_min_tokens": 500},
        },
        "setup": {"bigfile_bytes": 130_000},
        "checks": [
            {"stream": "stderr", "kind": "regex-present", "pattern": r"code=file-too-large"},
            {"stream": "stderr", "kind": "regex-present", "pattern": r"start_line"},
            {"stream": "stdout", "kind": "regex-present", "pattern": r"\S"},
        ],
    },
    {
        "name": "compaction_e2e",
        "description": (
            "End-to-end check (stub LLM, no network): the deterministic "
            "replacement for the old live compaction scenario. Under a tiny "
            "context_limit the compaction ladder fires on every run — it splices "
            "a digest back under the cap, re-injects the original request as the "
            "task anchor, prunes tool scaffolding past the fold, and completes "
            "the turn; and when the summarizer cannot shrink, the overflow "
            "force_fold engages instead of hanging. Model-recall quality is "
            "intentionally out of scope (a model property, not a harness one)."
        ),
        "inline": "compaction_e2e.py",
    },
    {
        "name": "scope_decline",
        "description": (
            "The agent declines an out-of-scope (non-software) request "
            "without calling any tools."
        ),
        "turns": [
            "Plan me a three-course dinner menu for six guests this "
            "weekend, with a shopping list.",
        ],
        "config": "default",
        "setup": {},
        "checks": [
            {"stream": "stderr", "kind": "regex-absent", "pattern": r"Tool call:"},
            {
                "stream": "stdout",
                "kind": "regex-present",
                "pattern": r"(?i)software engineering|codebase|project",
            },
        ],
    },
    {
        "name": "search_focus_trace",
        "description": (
            "web_search is used to check for newer-Python simplifications "
            "of mathlib.py; the [focus] nudge is reported informationally "
            "only (fires only when the model over-searches)."
        ),
        "turns": [
            "This project targets Python 3.12. Search the web and tell me "
            "whether anything in newer Python releases would let us "
            "simplify mathlib.py. Keep it strictly relevant to this "
            "codebase.",
        ],
        "config": "default",
        "setup": {"mathlib.py": MATHLIB_PY},
        "checks": [
            {"stream": "stdout", "kind": "regex-present", "pattern": r"\S"},
            {"stream": "stderr", "kind": "regex-present", "pattern": r"Tool call: web_search"},
            {"stream": "stderr", "kind": "regex-note", "pattern": r"\[focus\]"},
        ],
    },
    {
        "name": "inline_loop_guard",
        "description": (
            "Dispatch-level check (no LLM): the loop-guard steer suffix "
            "is absent on the first identical failing dispatch and "
            "present on the second."
        ),
        "inline": "inline_loop_guard.py",
    },
    {
        "name": "repeat_call_guard",
        "description": (
            "Dispatch-level check (no LLM): the success-path loop-guard "
            "steer is absent on the first identical successful call and "
            "present on the second; errors are not double-steered; the "
            "hard-cap constant and run_command/run_tests exemption are set."
        ),
        "inline": "repeat_call_guard.py",
    },
    {
        "name": "repeat_dedup_render",
        "description": (
            "End-to-end check (stub LLM, no network): a repeated identical "
            "successful read-only call is deduped — its full body replaced by a "
            "short stub from repeat #2 on — while safety conditions force the "
            "full body instead: a fingerprint mismatch when a re-read reflects a "
            "just-applied edit, and a compaction since the last full render (both "
            "would otherwise strand the model). Verification tools (run_command) "
            "are deduped to a [no-change] stub only when nothing has been modified "
            "since the identical run; a mutation between runs (or changed output) "
            "forces the full body and re-stamps, and read-only dedup ignores the "
            "mutation stamp."
        ),
        "inline": "repeat_dedup_render.py",
    },
    {
        "name": "verify_scratch_contract",
        "description": (
            "Dispatch-level check (no LLM): verify_scratch honors its full "
            "contract with real subprocesses — pass/fail exit codes, "
            "invalid-interpreter rejected with zero side effects, cwd is the "
            "real project root, and the temp file never leaks."
        ),
        "inline": "verify_scratch_contract.py",
    },
    {
        "name": "compaction_tail_prune",
        "description": (
            "Dispatch-level check (no LLM): after compaction folds past the "
            "last user message, the assembled tail carries no tool results or "
            "tool_calls; a new post-compaction user turn still keeps its "
            "in-flight tool scaffolding verbatim."
        ),
        "inline": "compaction_tail_prune.py",
    },
    {
        "name": "steer_channel",
        "description": (
            "Dispatch-level check (no LLM): append_steer persists a user-role "
            "message with the [harness] prefix and steer flag (round-tripped "
            "through resume); the provider payload strips the steer key; the "
            "compaction summarizer labels the steer 'harness' not 'user'; a "
            "plain append_user message is unaffected."
        ),
        "inline": "steer_channel.py",
    },
    {
        "name": "edit_lines",
        "description": (
            "Dispatch-level check (no LLM): edit_lines replaces a line range, "
            "inserts on an empty range (end=start-1) and at the top, rejects "
            "out-of-bounds ranges with code=bad-range, honors the not-read-yet "
            "and file-changed-on-disk gates, preserves trailing-newline "
            "behavior, and echoes a cat -n numbered preview; read_file emits "
            "cat -n output with TRUE line numbers under paging."
        ),
        "inline": "edit_lines.py",
    },
    {
        "name": "loop_guard_escalation",
        "description": (
            "End-to-end check (stub LLM, no network): a model that re-issues an "
            "identical blocked call is force-finalized by the harness after "
            "_BLOCKED_STREAK_CAP consecutive blocks (with a fold-surviving steer "
            "emitted first); the give-up envelope is synthesized truthfully from "
            "turn_report — naming applied files + verification runs, never "
            "claiming work is 'unavailable', and stamping verified with the same "
            "formula as a normal finalize; a real dispatch between blocks resets "
            "the streak so escalation never fires prematurely; and a blocked-round "
            "steer survives _prune_messages while the tool scaffolding before it drops."
        ),
        "inline": "loop_guard_escalation.py",
    },
    {
        "name": "repro_steer",
        "description": (
            "End-to-end check (stub LLM, no network): the first relevant file "
            "mutation of a turn that has run nothing to observe the problem gets "
            "a fold-surviving reproduce-before-edit steer (user role, "
            "STEER_PREFIX) plus a 'repro-steer: fired' telemetry line, fired once "
            "per turn; a prior FAILING run_command suppresses it (exit-status-"
            "agnostic, since verification_runs records every run); and a second "
            "edit in the same turn does not append a second steer."
        ),
        "inline": "repro_steer.py",
    },
    {
        "name": "steer_scaffolding",
        "description": (
            "End-to-end + dispatch-level check (stub LLM, no network): a "
            "mid-turn harness steer never moves the in-flight-turn boundary in "
            "the sent view. Through the real turn loop, an unverified edit's "
            "reproduce-before-edit steer leaves the first round's assistant "
            "tool_calls row and its tool result intact (the steer sits AFTER "
            "them); driving _prune_messages directly, an in-turn steer keeps "
            "every row while a completed turn still collapses to [user, final "
            "answer]; and a post-compaction tail with no real user folds its "
            "tool scaffolding while the steer row rides through."
        ),
        "inline": "steer_scaffolding.py",
    },
    {
        "name": "compaction_force_fold",
        "description": (
            "Dispatch-level check (no LLM): the overflow-ladder force_fold "
            "advances the watermark past all but the most recent messages with "
            "no summarizer call — seeds a marker when no summary exists, "
            "preserves an existing one, walks the boundary off an orphaning "
            "tool row, and refuses when nothing remains to fold."
        ),
        "inline": "compaction_force_fold.py",
    },
    {
        "name": "record_graph",
        "description": (
            "Dispatch-level check (no LLM): the unified `record` graph tool writes "
            "each kind and echoes the new node id; a pivot supersedes a decision by "
            "TITLE (edge + superseded_at stamp); an ambiguous title lists candidates "
            "and refuses the whole write; an unknown title refuses transactionally; "
            "and supersedes by raw id still works."
        ),
        "inline": "record_graph.py",
    },
    {
        "name": "find_dead_code_contract",
        "description": (
            "Dispatch-level check (no LLM): find_dead_code flags a genuinely "
            "unused Python function via vulture (path:line + % confidence, "
            "count >= 1, adapter='vulture'); a clean file reports count == 0; a "
            "nonexistent path errors code='not-found'; and a directory with only "
            "a non-Python file errors code='no-adapter'. Skips if vulture is "
            "absent from PATH."
        ),
        "inline": "find_dead_code_contract.py",
    },
    {
        "name": "run_command_mutations",
        "description": (
            "Dispatch-level check (no LLM): run_command's foreground path "
            "snapshot-diffs the project tree and publishes 'created'/'changed'/"
            "'deleted' mutation events for files a shell command touches; writes "
            "into __pycache__/ and dot-dirs emit nothing; a pure read stays "
            "silent; and the timeout path still fires the event for a file "
            "written before the kill. Every event carries an absolute path."
        ),
        "inline": "run_command_mutations.py",
    },
    {
        "name": "run_command_grounding",
        "description": (
            "Dispatch-level check (no LLM): run_command appends a grounding "
            "line to a NONZERO-exit render naming the exit code and the concrete "
            "absolute working directory (so a hallucinated cd self-corrects), "
            "while a zero-exit render and the timeout render carry no such line "
            "(no token tax on success); and the tool description states commands "
            "already run at the project root so a cd is never needed."
        ),
        "inline": "run_command_grounding.py",
    },
    {
        "name": "consolidation_ops",
        "description": (
            "Dispatch-level check (no LLM): consolidation's op-apply seam "
            "(apply_ops) writes a DECISION node; a PIVOT supersedes a decision by "
            "EXACT TITLE (edge + superseded_at stamp) and by raw id; an unknown "
            "pivot title writes nothing and is counted as a noop; and an ADD atom "
            "op still writes a fact row after the refactor."
        ),
        "inline": "consolidation_ops.py",
    },
    {
        "name": "llm_wall_clock",
        "description": (
            "Dispatch-level check (no LLM, no network): the streaming path's "
            "whole-call wall-clock ceiling. Driving _read_sse_response directly, "
            "a source that trickles SSE lines forever under a tiny ceiling raises "
            "the distinct ResponseCeilingError within ceiling + 1s and its message "
            "reports the discarded work (elapsed seconds + accumulated content); a "
            "normal stream (usage chunk + [DONE]) still parses unchanged (text "
            "joined, tool-call fragments merged, tokens captured); and driving "
            "chat() end-to-end, a breach raises after exactly one request issue — "
            "it never retries."
        ),
        "inline": "llm_wall_clock.py",
    },
    {
        "name": "verify_tool_wiring",
        "description": (
            "End-to-end check (stub LLM, no network): when the reproduce-before-"
            "edit steer fires, the harness activates the verification tool it "
            "names, so a scripted model can call run_command DIRECTLY the next "
            "round (no load_tool) and it dispatches — recorded in "
            "verification_runs status=success, never a not-loaded error; a "
            "read-only turn never fires the steer and leaves run_command "
            "inactive; and activating the verification tools twice never "
            "duplicates a schema entry."
        ),
        "inline": "verify_tool_wiring.py",
    },
    {
        "name": "envelope_net_changes",
        "description": (
            "End-to-end check (stub LLM, no network): the result envelope's "
            "files_changed entries are annotated with reverted only on a KNOWN "
            "net no-op. A file edited then written back byte-for-byte is flagged "
            "reverted while verified stays True; a create-then-delete is flagged; "
            "a file left changed carries no reverted key; and a file whose first "
            "mutation is a run_command shell side effect (no capturable "
            "pre-image) is never flagged even when a later call restores its "
            "original content — unknown is never guessed."
        ),
        "inline": "envelope_net_changes.py",
    },
]


def find(substr: str) -> list[dict]:
    """Return scenarios whose name contains *substr* (case-sensitive substring match)."""
    return [s for s in SCENARIOS if substr in s["name"]]
