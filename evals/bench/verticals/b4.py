"""B4 — Performance diagnosis (`--mode performance_debug`, 2 tasks).

Fixture `grind/`: a small batch pipeline (generate -> parse -> dedup ->
enrich -> aggregate) that turns synthetic access-log events into a
per-region summary report, driven by a deterministic entry point
(`python3 workload.py --scale N`). Per DESIGN.md's B4 section, the fixture
plants a dominant CPU hotspot (T1), a stdlib-attributed second cost whose
actionable cause sits at a specific call site (T2 CPU), an unbounded
memoization cache (T2 memory), an O(n^2) accumulation that is invisible at
the default scale and dominant at 10x (T3), and a cold decoy that looks
like the worst offender on paper but never executes.

Two tasks split the CPU-shaped diagnosis from the memory-shaped one, per
DESIGN.md's explicit "one CPU-focused, one memory-focused" split — a single
prompt asking for both would blur which profiler (`profile_hotspots` /
`trace_execution` vs `profile_memory`) the answer actually had to use.
Ground truth, measured evidence, and reference answers live in
truth/b4_multi_grind_hotspots/ and truth/b4_t2_unbounded_cache/.

Neither prompt names a culprit file or function — both name only the
workload entry point and the operational complaint, matching how the other
verticals' prompts are frozen from a user-style report rather than a
pointer at the fix.

`tree_guard` (byte_identical) and `envelope_guard` (expect_verified=None)
both carry `weight: 0, gate: True`: performance_debug has no file-editing
tools, so a truthful run leaves the tree untouched and reports
`verified: null` / `files_changed: []` — DESIGN.md's envelope-honesty
contract. Weight 0 means neither contributes to the scored total on its
own; `gate: True` is what actually enforces them — per
`graders/__init__.py`, a gated grader scoring below its threshold forces
the whole task's score to 0, so a run that mutates the fixture (or lies
about `verified`) is zeroed regardless of how good the perf_report answer
is.

`perf_report`'s ranking band (`graders/perf_report.py`) only scores facts
that carry an integer `rank` in card.json, and needs >=2 ranked+recalled
facts to be non-vacuous. Whether a task's facts get ranked is a per-task
call grounded in whether relative ordering is a real, single-context
measured property:

- `b4_multi_grind_hotspots`' card ranks exactly the three scale-1 hotspots
  measured in one cProfile pass and reported in that same order
  (`canonicalize_path` #1 at 14.5%, `raw_decode` #2 at 8.4%,
  `mark_duplicates` #3 at 7.6% — see
  `truth/b4_multi_grind_hotspots/validation/cprofile_scale1.txt`).
  `t2_redundant_parse_cause_site` (a cause-site explanation of *why* T2
  costs what it does, not a competing hotspot) and
  `t3_scale_dependent_ranking_shift` (a claim about the *different*,
  scale-10 ordering) are deliberately left unranked: folding either into
  the scale-1 rank set would either be meaningless (the cause-site fact
  isn't "more or less expensive" than the others) or self-contradicting
  (the shift fact's whole point is that `mark_duplicates` overtakes
  `canonicalize_path` at 10x — the opposite of its scale-1 rank).
- `b4_t2_unbounded_cache`'s card ranks nothing; its `perf_report` grader
  spec instead passes `ranking_weight: 0` (redistributed into
  `recall_weight`/`evidence_weight`). Its three facts are facets of one
  causal chain (cache exists -> grows unbounded -> because the key has
  near-unique cardinality), not competing hotspots — there is no measured
  "which one matters more" to rank.
"""

TASKS = [
    {
        "id": "b4_multi_grind_hotspots",
        "vertical": "B4",
        "tier": "multi",
        "mode": "performance_debug",
        "fixture": "grind",
        "prompt": (
            "You maintain \"grind\", a nightly batch job that turns raw "
            "event logs into a per-region summary report. The entry point "
            "is `python3 workload.py --scale N` (N controls input volume; "
            "N=1 is today's nightly volume).\n\n"
            "Two things are going on:\n"
            "1. The nightly run has gotten noticeably slower over the last "
            "few releases and on-call wants to know what is actually "
            "costing the time right now, with file and function names, not "
            "guesses.\n"
            "2. Ingestion volume is expected to grow roughly 10x within the "
            "quarter, and we need to know now what will need to be fixed "
            "*before* that happens -- something that is cheap today can "
            "become the dominant cost at higher volume, and we'd rather "
            "find that with a profiler than in production.\n\n"
            "Profile `workload.py` and produce a measured report: name every "
            "real hotspot (file + function), attach the actual numbers you "
            "measured for each claim (time, percentage of total, call "
            "counts -- whatever the profiler gives you), and call out "
            "anything whose ranking changes between today's volume and the "
            "10x case. If something in the code looks alarming but your "
            "profiling shows it isn't actually costing anything, say so -- "
            "we'd rather not waste a sprint optimizing something that "
            "never runs."
        ),
        "timeout_s": 1800,
        "graders": [
            # Ranking band graded via card.json's three scale-1 rank fields
            # (default weights: recall 50 / evidence 30 / ranking 10 /
            # decoy 10) — see module docstring for why those three and not
            # the other two facts.
            {"kind": "perf_report", "weight": 100},
            {"kind": "tree_guard", "weight": 0, "gate": True, "mode": "byte_identical"},
            {"kind": "envelope_guard", "weight": 0, "gate": True, "expect_verified": None},
        ],
    },
    {
        "id": "b4_t2_unbounded_cache",
        "vertical": "B4",
        "tier": "T2",
        "mode": "performance_debug",
        "fixture": "grind",
        "prompt": (
            "\"grind\" (a nightly batch job, entry point `python3 "
            "workload.py --scale N`) runs inside a container with a fixed "
            "memory limit. At today's volume (N=1) it fits comfortably, but "
            "we're planning to raise the input volume and the on-call "
            "engineer who last ran it at a higher scale said it \"felt like "
            "it was using a lot more memory than the data size would "
            "suggest.\"\n\n"
            "Profile the memory behavior of `workload.py` across at least "
            "two scales and produce a measured report: what is actually "
            "driving memory growth, is it bounded or does it keep growing "
            "with input size, and where in the code does it happen (file + "
            "function). Back every claim with real numbers from the "
            "profiler (bytes/KiB/MB, entry counts, growth ratios across "
            "scales) -- not a guess from reading the code."
        ),
        "timeout_s": 1800,
        "graders": [
            # No card.json rank fields — its three facts are facets of one
            # causal chain, not competing hotspots to order (see module
            # docstring). ranking_weight: 0 with the freed 10 points
            # redistributed into recall/evidence (recall still the primary
            # "did you name the mechanism" signal; evidence gets the larger
            # share since citing the real hit-rate/growth numbers is what
            # separates genuine profiling from reciting the story).
            {
                "kind": "perf_report",
                "weight": 100,
                "recall_weight": 55,
                "evidence_weight": 35,
                "ranking_weight": 0,
                "decoy_weight": 10,
            },
            {"kind": "tree_guard", "weight": 0, "gate": True, "mode": "byte_identical"},
            {"kind": "envelope_guard", "weight": 0, "gate": True, "expect_verified": None},
        ],
    },
]
