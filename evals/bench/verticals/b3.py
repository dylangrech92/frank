"""B3 -- Discovery (`--mode research`, 6 single-question tasks).

Fixture `relay/`: an event-pipeline application (stdlib-only) built to
defeat shallow, single-hop search. It carries five deliberate
indirection mechanisms, each load-bearing on a real execution path: a
string-keyed stage-dispatch table, decorator-based handler registration
on an in-process event bus, a legacy compatibility shim re-exporting a
current symbol under an old name, a callback passed as a value and
invoked from a different module than the one that scheduled it, and
exactly one dynamic import (the enrichment plugin loader).

Every task below carries a single named trap: a specific, plausible,
demonstrably-reachable wrong answer that a grep-level or one-hop
investigation lands on. See `truth/<task_id>/card.json` for the exact
fact/trap regexes and `truth/<task_id>/validation/` for the evidence
that grounds each one in the fixture.

Scoring (per DESIGN.md's B3 section): the entire 0-100 task score comes
from a single `answer_facts` grader whose own 100 points split as
required-fact recall 70 + coherence 10 (folded into `recall_weight=90`
so a perfect, trap-free, coherent answer reaches 100; a triggered trap
subtracts its `penalty`, which is exactly the missing 20 -- see the
grader's own docstring for why `recall_weight` absorbs both the base 70
and the conditional 20). `tree_guard` (byte-identical) and
`envelope_guard` (`verified: null`, `files_changed: []`) are carried at
weight 0 with `gate: true`: a research run that mutates the fixture, or
whose envelope doesn't honestly report having made no changes, forces
the whole task to 0 regardless of how good the prose answer is -- this
is DESIGN.md's non-negotiable envelope-honesty gate, expressed as a
veto rather than a partial deduction.
"""

_GATE_GRADERS = [
    {"kind": "tree_guard", "weight": 0, "gate": True, "mode": "byte_identical"},
    {"kind": "envelope_guard", "weight": 0, "gate": True, "expect_verified": None},
]


def _answer_facts_graders():
    return [
        {"kind": "answer_facts", "weight": 100, "recall_weight": 90, "coherence_weight": 10},
        *_GATE_GRADERS,
    ]


TASKS = [
    {
        "id": "b3_t1_deliver_stage_dispatch",
        "vertical": "B3",
        "tier": "T1",
        "mode": "research",
        "fixture": "relay",
        "prompt": (
            "The pipeline's `deliver` stage is implemented by a function "
            "called `run_deliver`. Where exactly is `run_deliver` defined, "
            "where is it invoked from, and what is the precise mechanism "
            "that connects the `deliver` stage to that function during a "
            "normal pipeline run? Don't just say 'the pipeline calls it' "
            "-- point to the actual definition site and call site, and "
            "explain how the call is made."
        ),
        "timeout_s": 1800,
        "graders": _answer_facts_graders(),
    },
    {
        "id": "b3_t2_normalize_caller_set",
        "vertical": "B3",
        "tier": "T2",
        "mode": "research",
        "fixture": "relay",
        "prompt": (
            "Find every place in this codebase that calls "
            "`relay.handlers.transform.normalize_event` -- the function "
            "that trims and coerces an order's fields. I want the "
            "complete caller set, including anything registered "
            "indirectly (not called by that literal name in the source) "
            "and anything reached through a re-exported or aliased "
            "import. For each caller, say which module it lives in and "
            "how it reaches `normalize_event` (direct import, "
            "registration, or alias)."
        ),
        "timeout_s": 1800,
        "graders": _answer_facts_graders(),
    },
    {
        "id": "b3_t2_order_created_trace",
        "vertical": "B3",
        "tier": "T2",
        "mode": "research",
        "fixture": "relay",
        "prompt": (
            "Trace a single order event through this pipeline end to "
            "end, starting from `python3 main.py demo` reading a raw "
            "record out of `sample_events.jsonl`, all the way to wherever "
            "a successfully-processed order is ultimately persisted under "
            "the default configuration. Name every stage and handler the "
            "event passes through, in order, and identify the exact file "
            "and function where it is finally written to disk."
        ),
        "timeout_s": 1800,
        "graders": _answer_facts_graders(),
    },
    {
        "id": "b3_t2_enrich_error_handling",
        "vertical": "B3",
        "tier": "T2",
        "mode": "research",
        "fixture": "relay",
        "prompt": (
            "Look at the pipeline's `enrich` stage. Which functions it "
            "calls can raise an exception, what exception type does each "
            "one raise, and once that exception reaches the orchestrator, "
            "what actually happens to the event in each case? I want the "
            "full error-handling behavior for every failure mode "
            "reachable from that stage, not just that 'errors are "
            "handled'."
        ),
        "timeout_s": 1800,
        "graders": _answer_facts_graders(),
    },
    {
        "id": "b3_t3_retry_config_flag",
        "vertical": "B3",
        "tier": "T3",
        "mode": "research",
        "fixture": "relay",
        "prompt": (
            "Which single configuration flag governs whether a failed "
            "event is retried at all, as opposed to merely tuning how "
            "retries behave once they're already allowed? Identify the "
            "flag and the exact place it's checked, and explain "
            "concretely what would change about the pipeline's behavior "
            "if that check were deleted from the code -- not what the "
            "flag's docstring says it does, what actually changes at "
            "runtime."
        ),
        "timeout_s": 1800,
        "graders": _answer_facts_graders(),
    },
    {
        "id": "b3_t3_deadletter_write_paths",
        "vertical": "B3",
        "tier": "T3",
        "mode": "research",
        "fixture": "relay",
        "prompt": (
            "Enumerate every distinct code path in this repository that "
            "can cause a record to be written to the dead-letter file. "
            "For each one, name the file and function, and describe the "
            "condition that triggers it. I want an exhaustive list -- "
            "missing one, or including something that merely logs "
            "about a dead-lettered event without writing the file "
            "itself, will be scored against you."
        ),
        "timeout_s": 1800,
        "graders": _answer_facts_graders(),
    },
]
