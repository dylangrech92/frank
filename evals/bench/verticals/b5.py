"""B5 — Silent bugs & dead-code audit (`--mode research`, 1 task).

Fixture `ledger/`: a batch CSV import / validate / aggregate / report
application (~2k LOC, stdlib-only). The task spans all three difficulty
tiers inside a single findings-list answer rather than one task per tier
(see truth/b5_multi_ledger_audit/ for the full ground-truth card, reference
answer, and per-defect validation evidence).

Scoring is entirely inside the `findings_list` grader, which implements the
DESIGN.md B5 formula directly: recall per tier (T1 20, T2 30, T3 30) +
precision 20, weights summing to 100. `tree_guard` and `envelope_guard` are
carried at weight 0 with `gate: true` — research mode's non-negotiable gate
(byte-identical tree, `verified: null`, `files_changed: []`) per DESIGN.md's
envelope-honesty section, not a scored contributor.
"""

TASKS = [
    {
        "id": "b5_multi_ledger_audit",
        "vertical": "B5",
        "tier": "multi",
        "mode": "research",
        "fixture": "ledger",
        "prompt": (
            "You are performing a code audit of this repository (a batch "
            "CSV import / validate / aggregate / report application called "
            "\"ledger\"). Read the code -- do not run it destructively or "
            "modify anything -- and produce a NUMBERED findings list of "
            "genuine defects: silent bugs (code that runs without crashing "
            "but produces wrong, stale, or lost output) and dead code "
            "(functions, branches, or config that can never execute or have "
            "no effect).\n\n"
            "For each finding, use this exact format:\n\n"
            "N. <file path> -- <one-line description of the concrete defect "
            "and why it is wrong>\n\n"
            "Requirements:\n"
            "- Only report defects you can point to a specific file and "
            "mechanism for. A vague or generic claim (\"this could be "
            "improved\", \"consider adding tests\") is not a finding.\n"
            "- Do not report code you have not actually traced through -- "
            "reachable code that just looks unusual is not a finding, and "
            "neither is code that looks suspicious at a glance but turns "
            "out, on closer reading, to behave correctly.\n"
            "- False positives are counted against you: every finding that "
            "does not correspond to a real defect lowers your score, so a "
            "long list of maybes will score worse than a short, accurate "
            "list.\n"
            "- This is a research-only task: do not edit any files."
        ),
        "timeout_s": 2400,
        "graders": [
            {
                "kind": "findings_list",
                "weight": 100,
                "tier_weights": {"T1": 20, "T2": 30, "T3": 30},
                "precision_weight": 20,
            },
            {"kind": "tree_guard", "weight": 0, "gate": True, "mode": "byte_identical"},
            {"kind": "envelope_guard", "weight": 0, "gate": True, "expect_verified": None},
        ],
    },
]
