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

NOTES_MD = """# Notes

Scratch notes fixture for the compaction-coherence eval scenario. Not load
bearing content — just needs to exist so the model has a second file to
read and discuss during the verbose-analysis turns.

- Project convention: prefer small, well-documented helper functions.
- Remember to keep functions pure where possible.
"""


SCENARIOS: list[dict] = [
    {
        "name": "load_tool_gate",
        "description": (
            "read_file is rejected with code=not-loaded until load_tool "
            "brings it into the active tool set."
        ),
        "turns": [
            "Read mathlib.py and list every function it defines with a "
            "one-line description of each.",
        ],
        "config": "default",
        "setup": {"mathlib.py": MATHLIB_PY},
        "checks": [
            {"stream": "stderr", "kind": "regex-present", "pattern": r"code=not-loaded"},
            {
                "stream": "stderr",
                "kind": "ordered",
                "patterns": [
                    r"code=not-loaded",
                    r"Tool call: load_tool",
                    r"Tool call: read_file",
                ],
            },
        ],
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
        "name": "compaction_coherence",
        "description": (
            "Mid-session compaction fires under a tight context limit and "
            "the model still recalls facts from before the compaction."
        ),
        "turns": [
            "Give me a verbose analysis (600+ words) of mathlib.py: "
            "explain every function's purpose, edge cases, and how you "
            "would test it.",
            "Give me a verbose analysis (600+ words) of notes.md: explain "
            "every point it makes and how it should shape how we write "
            "code in this project.",
            "Without reading any files again, from memory alone: what was "
            "the ORIGINAL first request I gave you this session, which "
            "files have you read, and what does the mean() function do "
            "when given an empty list?",
        ],
        "config": {
            "llm": {"context_limit": 4500},
            "compaction": {"reserve_ratio": 0.1, "reserve_min_tokens": 500},
        },
        "setup": {"mathlib.py": MATHLIB_PY, "notes.md": NOTES_MD},
        "checks": [
            {"stream": "stderr", "kind": "regex-present", "pattern": r"post-compaction #"},
            {"stream": "stdout", "kind": "regex-present", "pattern": r"ValueError"},
        ],
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
]


def find(substr: str) -> list[dict]:
    """Return scenarios whose name contains *substr* (case-sensitive substring match)."""
    return [s for s in SCENARIOS if substr in s["name"]]
