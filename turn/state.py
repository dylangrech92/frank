from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _TurnState:
    """Mutable per-turn state threaded through one ``handle_user_message`` call.

    Groups the locals the turn loop shares across its LLM calls, dispatch
    rounds, and gate cascade so the helpers can read and mutate them by
    reference. One instance lives for exactly one turn and is discarded.
    """

    # S4 per-turn report (files_changed, verification_runs, usage, gate flags);
    # also exposed on ``session.turn_report`` so --json mode can read it back
    # even if the turn raises before returning. ``answer`` holds the UNPREFIXED
    # final text — the single source of truth for both the return value (which
    # may still get the "[UNVERIFIED CHANGES] " prefix layered on for prose
    # callers) and the envelope's answer field.
    turn_report: dict
    cap: int
    window: int
    comp_cfg: dict
    # Token estimate of the assembled context the current chat request was built
    # from (after any pre-flight compaction). Reused by the oversize-result
    # guard and the usage calibration for that same request.
    est: int = 0
    # Compactions performed this turn, capped across the whole turn.
    compactions: int = 0
    # Chars streamed through on_delta for the CURRENT chat call only (reset
    # before each call), so the return path knows whether the answer already
    # reached the sink or must be delivered whole (non-streaming/gated fallback).
    streamed: int = 0
    # The turn's original user message, kept for the bug-report lexicon check the
    # no-failure-observed steer gates on. Set once at turn start.
    user_message: str = ""
    # H1/S3 verify gate: True once a file was created/changed/renamed with no
    # subsequent PASSING run_tests/run_command/verify_scratch; cleared the moment
    # such a call passes (a run that merely completed but exited nonzero / left a
    # failing test does NOT clear it). The nudge fires at most once per turn.
    needs_verification: bool = False
    verification_nudge_fired: bool = False
    # Reproduce-before-edit steer: True once the first relevant file mutation
    # of the turn landed with zero verification runs recorded so far. Fires the
    # fold-surviving steer at most once per turn (see _dispatch_sequential_call /
    # _dispatch_round).
    repro_steer_fired: bool = False
    # No-failure-observed steer: True once the turn's first relevant file mutation
    # landed while >= 1 verification run had been recorded AND every one passed,
    # on a task whose original request reads as a bug report. Mutually exclusive
    # with repro_steer_fired (that owns the zero-runs case). Fires the fold-
    # surviving steer at most once per turn (see _dispatch_sequential_call /
    # _dispatch_round).
    no_failure_steer_fired: bool = False
    # Empty-answer bounce: the model returned no text and no tool calls (a
    # local-model bare-stop failure mode). Fires at most once per turn; a second
    # empty turn falls through to a transparent placeholder at the return site.
    empty_answer_nudge_fired: bool = False
    # H5 graph-memory nudge input: whether a record(kind='decision'/'spec') call ran
    # this turn (paired with mutated_paths); the fired-once-per-session flag lives in
    # _GRAPH_MEMORY_NUDGE_FIRED.
    record_decision_or_spec_called: bool = False
    # Reactive web-search focus nudge: consecutive successful web_search calls
    # since the last web_read.
    searches_without_read: int = 0
    # files_changed dedupe set: first-tool-wins, in order of first mutation.
    files_changed_seen: set[str] = field(default_factory=set)
    # Turn-start pre-images for net-change detection: normalized absolute path
    # (the SAME resolved key a files_changed entry carries, so the two join
    # exactly) -> sha256 hex of the file's bytes BEFORE this turn first touched
    # it, or None meaning "did not exist at turn start". Captured at the dispatch
    # seam before a mutating tool writes; first capture per path wins (= the
    # turn-start state). A path ABSENT from this map is UNKNOWN — e.g. a file
    # whose first mutation came from a run_command shell side effect, which
    # carries no path argument to snapshot — and is never guessed at annotation
    # time. Read once at the turn-exit choke point to flag reverted entries.
    preimages: dict[str, str | None] = field(default_factory=dict)
    # Rendered error envelopes seen this turn, for the repeated-identical-failure
    # loop-guard (keyed by (name, rendered)).
    seen_errors: dict[tuple[str, str], int] = field(default_factory=dict)
    # Identical (name, arg-signature) pairs dispatched this turn, for the
    # repeated-successful-call loop-guard steer plus the hard dispatch cap.
    seen_calls: dict[tuple[str, str], int] = field(default_factory=dict)
    # Full-render dedup stamps for repeated identical successful dedup-eligible
    # calls, keyed by (name, arg-signature) ->
    # (render fingerprint, compactions-at-render, mutations-at-render).
    # Lets _repeat_call_check replace a provably-unchanged repeat with a short stub
    # instead of re-emitting the body: the fingerprint forces a fresh body when a
    # re-read reflects a just-applied edit, the compaction count forces one when a
    # fold may have dropped the earlier result out of context, and the mutation
    # count forces one for verification tools when a file changed since the run
    # (read-only dedup carries but does not gate on it). Distinct from seen_calls,
    # whose int tally the pre-dispatch hard cap depends on.
    seen_renders: dict[tuple[str, str], tuple[str, int, int]] = field(default_factory=dict)
    # Identical non-empty assistant texts emitted on TOOL-BEARING rounds this
    # turn, keyed by the stripped text, for the narration-runaway guard
    # (_text_runaway_count). Terminal no-tool-call rounds are never tallied here
    # — a final answer echoing earlier narration is an ending, not a loop.
    seen_texts: dict[str, int] = field(default_factory=dict)
    # Every distinct path mutated this turn (H1/H5 tracking).
    mutated_paths: set[str] = field(default_factory=set)
    # Monotonic count of file-mutation EVENTS this turn (not distinct paths — a
    # second edit to an already-mutated path bumps it), spanning both direct
    # edit-tool mutations and run_command's snapshot-diff shell mutations (both
    # publish on the _TURN_MUTATIONS bus). Recorded at every dedup-eligible full
    # render; a verification repeat is only stubbed when this is unchanged since
    # the identical earlier run, so a byte-identical re-run with nothing modified
    # is deduped while a genuine edit→retest re-renders the full body.
    mutation_events: int = 0
    # Loop-guard escalation: consecutive tool calls refused by the repeat-call
    # hard cap, counted regardless of key (alternating between two blocked calls
    # is just as stuck). Incremented on every hard-cap block, reset to 0 the
    # moment any call actually dispatches (sequential or a parallel-safe batch) —
    # a real dispatch proves progress. When it reaches ``_BLOCKED_STREAK_CAP`` the
    # turn is force-finalized instead of looping the same blocked call forever
    # (see handle_user_message / _blocked_loop_giveup).
    blocked_streak: int = 0
