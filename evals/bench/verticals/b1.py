"""B1 -- Bug fixing.

Three tasks against the ``queueworks`` fixture (a small job-queue library):
a crash whose traceback surfaces away from its causal line, a silent
action-at-a-distance wrong-output bug, and an interacting pair where the
obvious single-line fix alone still fails hidden acceptance.
"""

TASKS = [
    {
        "id": "b1_t1_restart_crash",
        "vertical": "B1",
        "tier": "T1",
        "mode": "code",
        "fixture": "queueworks",
        "prompt": (
            "We run queueworks jobs in short-lived worker processes -- "
            "process whatever's pending, let old completed jobs get "
            "purged, then exit and let a fresh process pick up the queue "
            "later. Somewhere in that restart-and-resume cycle we've "
            "started hitting a crash. Repro on a clean checkout:\n\n"
            "    cd queueworks && python3 -m queueworks.demo restart\n\n"
            "It runs fine for a bit and then blows up with a TypeError "
            "instead of printing the final job statuses. This didn't used "
            "to happen -- it only started once we added the "
            "purge-old-completed-jobs step before the restart. Can you "
            "figure out what's actually going wrong and fix it so the "
            "restart-and-resume flow finishes cleanly?"
        ),
        "timeout_s": 1800,
        "graders": [
            {"kind": "answer_facts", "weight": 20},
            {
                "kind": "acceptance",
                "weight": 50,
                # core = the literal reported crash is gone; edge is the same
                # no-crash check at a bigger seq/purge gap (an overfit-to-the-
                # repro trap, not a distinct correctness dimension); adversarial
                # holds two probes -- one checking priority ORDER survives the
                # restart (catching a shallow fix that swallows the collision
                # and silently drops/reorders a job), one checking the resumed
                # counter accounts for every job status still in the store and
                # doesn't crash on a fully-purged restart (catching a fix that
                # narrows to pending-only jobs, or drops the empty-store
                # default) -- so it carries the most weight.
                "band_weights": {"core": 35.0, "edge": 25.0, "adversarial": 40.0},
            },
            {
                "kind": "tree_guard",
                "weight": 20,
                "mode": "confined_diff",
                "allowed_paths": ["queueworks/store.py"],
            },
            {"kind": "envelope_guard", "weight": 10, "expect_verified": True},
        ],
    },
    {
        "id": "b1_t2_report_order_corruption",
        "vertical": "B1",
        "tier": "T2",
        "mode": "code",
        "fixture": "queueworks",
        "prompt": (
            "Our ops dashboard shows two panels for a batch of jobs: a "
            "'slowest jobs' table and a chronological completion report. "
            "Someone noticed the chronological report comes out in the "
            "wrong order whenever the slowest-jobs panel is generated "
            "first -- jobs that finished first show up in the middle or "
            "at the end of the report instead. No exception, nothing in "
            "the logs, it just silently renders wrong. Repro:\n\n"
            "    cd queueworks && python3 -m queueworks.demo dashboard\n\n"
            "Compare the 'true completion order' the script prints "
            "against the chronological report's order right below it -- "
            "they don't match. Can you track down why generating the "
            "slowest-jobs stats is affecting the chronological report, "
            "and fix it?"
        ),
        "timeout_s": 1800,
        "graders": [
            {"kind": "answer_facts", "weight": 20},
            {
                "kind": "acceptance",
                "weight": 50,
                # core = the visible report is back in order; edge repeats the
                # same stats-then-report sequence twice (an overfit-to-one-call
                # trap on the same dimension as core); adversarial reads
                # store.get_completion_log() directly, bypassing reporting.py,
                # so it's the only probe that verifies the underlying SHARED
                # state was actually fixed rather than just the rendered view
                # -- that's the actual crux of a silent action-at-a-distance
                # bug, so it gets the largest share.
                "band_weights": {"core": 30.0, "edge": 20.0, "adversarial": 50.0},
            },
            {
                "kind": "tree_guard",
                "weight": 20,
                "mode": "confined_diff",
                "allowed_paths": ["queueworks/stats.py", "queueworks/store.py"],
            },
            {"kind": "envelope_guard", "weight": 10, "expect_verified": True},
        ],
    },
    {
        "id": "b1_t3_retry_cap_interaction",
        "vertical": "B1",
        "tier": "T3",
        "mode": "code",
        "fixture": "queueworks",
        "prompt": (
            "A job type that always fails is supposed to retry up to our "
            "configured max_retries=3 (four attempts total, per the "
            "README's retry semantics) with an exponential backoff capped "
            "at max_delay. QA says a permanently-failing job is only "
            "making 3 attempts before giving up instead of 4. Repro:\n\n"
            "    cd queueworks && python3 -m queueworks.demo retries\n\n"
            "Can you dig into why it's giving up early and get the retry "
            "behavior fully matching what the README describes?"
        ),
        "timeout_s": 1800,
        "graders": [
            {"kind": "answer_facts", "weight": 20},
            {
                "kind": "acceptance",
                "weight": 50,
                # core isolates the retry-COUNT bug (retry.py) and edge
                # isolates the delay-CAP bug (worker.py) -- two independent
                # dimensions, weighted equally since neither is a scaled
                # repeat of the other. adversarial holds two probes: one
                # checks attempt count AND the cap together using the repro's
                # own policy constants (operationalizing this task's
                # "interacting pair" property -- confirmed empirically to be
                # the only band that fails against EITHER single-file naive
                # fix), the other repeats that same combined check under a
                # RetryPolicy the repro never shows, catching a fix fit to the
                # specific numbers on screen (a hard-coded "3"/"1.0") rather
                # than one that actually reads self.max_retries /
                # self.retry_policy.max_delay -- so this band carries the
                # most weight.
                "band_weights": {"core": 25.0, "edge": 25.0, "adversarial": 50.0},
            },
            {
                "kind": "tree_guard",
                "weight": 20,
                "mode": "confined_diff",
                "allowed_paths": ["queueworks/retry.py", "queueworks/worker.py"],
            },
            {"kind": "envelope_guard", "weight": 10, "expect_verified": True},
        ],
    },
]
