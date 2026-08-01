"""How a turn ends, on every path.

``_finalize_answer`` is the normal exit: the no-tool-call gate cascade that
bounces once on an empty answer, once on an unverified file mutation, then marks
the answer and delivers it. The give-up exits — ``_over_cap_giveup``,
``_blocked_loop_giveup``, ``_text_loop_giveup`` — synthesize an answer when the
turn cannot continue at all. ``_report_finalize`` is verify mode's terminal-tool
exit: an accepted ``report`` call ends the turn on the spot. Every one of them
stamps the turn through ``_stamp_turn_outcome``, so all exits report the same
truth the same way.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Callable

import ui
from llm import ChatResponse
from session import Session
from turn.state import _TurnState
from turn.steering import _verification_nudge_text
from turn.verification import _available_verification_tools

# How many files / verification runs a synthesized give-up answer lists
# before collapsing the tail into a "+N more" count, so a huge turn cannot bloat
# the envelope.
_GIVEUP_FILES_CAP = 10
_GIVEUP_RUNS_CAP = 5


def _turn_verified(state: "_TurnState") -> bool | None:
    """Single source of truth for ``turn_report["verified"]``.

    Tri-state, so a consumer can tell an unverified edit from a turn that left no
    net change to verify:

    * ``None`` — no net change is known to exist, so there is nothing to verify:
      either the turn mutated no file (e.g. a read-only research turn), or it
      mutated files but every ``files_changed`` entry is flagged ``reverted`` (an
      edit-then-revert or a create-then-delete that left the tree byte-identical
      to its start). Neither pass nor fail applies. An entry whose revert cannot
      be known — the ``reverted`` flag is absent, meaning a genuine change OR an
      unknown pre-image (a ``run_command``-first mutation) — is NOT a known no-op,
      so it keeps the ``True``/``False`` verdict below rather than cascading here.
    * ``True`` — the turn mutated a file AND the verification gate is clear (a
      ``run_tests``/``run_command``/``verify_scratch`` call PASSED after the last
      mutation, clearing ``needs_verification`` — a run that merely completed but
      exited nonzero or left a failing test does not count).
    * ``False`` — the turn mutated a file but the gate is still open (no passing
      run after the last mutation).

    Reads the ``reverted`` annotations, so ``_annotate_reverted_files`` must have
    run first — ``_stamp_turn_outcome`` enforces that order at every exit path.
    Shared verbatim by ``_finalize_answer`` and both early-exit give-up paths so
    the flag means the same thing however the turn ended — a give-up envelope no
    longer under-reports verified work, and a net no-op turn no longer
    masquerades as an unverified edit.
    """
    if not state.mutated_paths:
        return None
    changed = state.turn_report["files_changed"]
    if changed and all(entry.get("reverted") is True for entry in changed):
        return None  # every mutation verifiably reverted — a known net no-op
    return not state.needs_verification


def _annotate_reverted_files(state: "_TurnState") -> None:
    """Flag ``files_changed`` entries the turn left byte-identical to their start.

    ``files_changed`` is an activity log — a file edited and then reverted to its
    original bytes still appears — so a consumer reading it alone would conclude
    the tree changed when the net state did not. This walks the log once and adds
    ``entry["reverted"] = True`` to an entry ONLY when the harness knows the
    file's turn-start state (a pre-image was captured, see ``_capture_preimage``)
    AND the file ended the turn identical to it:

    * a recorded hash that matches the current on-disk bytes (edited then written
      back), or
    * a recorded ``None`` (did not exist at turn start) and the path no longer
      exists (created then deleted).

    A path with no recorded pre-image is UNKNOWN and never annotated — the flag
    is absent, never guessed. The key is left off entirely otherwise (absent =
    changed-or-unknown), keeping the envelope shape minimal and append-only.
    Called at the single turn-exit choke point beside the ``verified`` stamp so
    every exit path (normal finalize and both give-up paths) reports net state.
    """
    from pathlib import Path as _Path

    for entry in state.turn_report["files_changed"]:
        key = entry.get("path")
        if key not in state.preimages:
            continue  # unknown turn-start state — never guess
        pre = state.preimages[key]
        target = _Path(key)
        if pre is None:
            if not target.exists():
                entry["reverted"] = True
            continue
        try:
            if (
                target.exists()
                and hashlib.sha256(target.read_bytes()).hexdigest() == pre
            ):
                entry["reverted"] = True
        except OSError:
            continue  # unreadable now — cannot confirm a revert, leave absent


def _stamp_turn_outcome(state: "_TurnState") -> None:
    """Annotate net-change state, then stamp ``verified`` — in that order.

    Every turn-exit path (the normal finalize and both give-up paths) must
    annotate ``files_changed`` reverts (``_annotate_reverted_files``) BEFORE
    computing ``verified`` (``_turn_verified``), because a turn whose every
    mutation was verifiably reverted is a known net no-op that reports
    ``verified`` as ``None`` — a verdict that reads the ``reverted`` flags this
    annotation writes. Folding the two calls into one function shared by every
    exit path makes that ordering contract structural, so it cannot silently
    break at a single call site.
    """
    _annotate_reverted_files(state)
    state.turn_report["verified"] = _turn_verified(state)


def _synthesize_giveup_answer(reason: str, retry_hint: str, state: "_TurnState") -> str:
    """Build a truthful give-up answer from the facts already in ``turn_report``.

    The two early-exit give-up paths used to emit a static string claiming
    "a report of what was accomplished is unavailable", which is false whenever
    the turn had already applied edits and verified them on disk. ``turn_report``
    already holds those facts (``files_changed`` + ``verification_runs``, each run
    with whether it passed), so this synthesizes an honest envelope from them with
    no extra LLM call. *reason* is
    the leading sentence naming why the harness ended the turn; *retry_hint* is
    the path-appropriate advice used only when the turn did no work at all.
    """
    files = state.turn_report["files_changed"]
    runs = state.turn_report["verification_runs"]

    if not files and not runs:
        # No mutations and no runs — nothing to report beyond the plain give-up,
        # staying close to each path's original terminal message.
        return f"{reason} No files were changed and nothing was run this turn — {retry_hint}"

    parts = [reason, "Work already applied this turn."]

    if files:
        seen: list[str] = []
        for entry in files:
            path = entry.get("path")
            if path and path not in seen:
                seen.append(path)
        shown = seen[:_GIVEUP_FILES_CAP]
        seg = "Files changed: " + ", ".join(shown)
        extra = len(seen) - len(shown)
        if extra > 0:
            seg += f" (+{extra} more)"
        parts.append(seg + ".")

    if runs:
        shown_runs = runs[:_GIVEUP_RUNS_CAP]
        rendered = [
            f"{r.get('tool')} {str(r.get('detail', ''))!r} "
            f"({'passed' if r.get('passed') else 'failed'})"
            for r in shown_runs
        ]
        seg = "Verification runs: " + "; ".join(rendered)
        extra = len(runs) - len(shown_runs)
        if extra > 0:
            seg += f" (+{extra} more)"
        parts.append(seg + ".")

    parts.append(
        "The turn was cut short, so parts of the request may be incomplete — "
        "review the applied changes before retrying."
    )
    return " ".join(parts)


def _finalize_giveup(
    session: Session,
    state: "_TurnState",
    reason: str,
    retry_hint: str,
    on_delta: Callable[[str], None] | None,
) -> str:
    """Shared tail for the two early-exit give-up paths.

    Both ``_over_cap_giveup`` and ``_blocked_loop_giveup`` end a turn without
    reaching ``_finalize_answer``. They fold their common tail here: synthesize a
    truthful answer from ``turn_report`` (no extra LLM call), record it as the
    turn's final answer across transcript + ``turn_report`` + ``on_delta``, and
    annotate net-change state then stamp ``turn_report["verified"]`` with the one
    shared formula (``_stamp_turn_outcome``).
    """
    msg = _synthesize_giveup_answer(reason, retry_hint, state)
    session.append_assistant(msg)
    state.turn_report["answer"] = msg
    _stamp_turn_outcome(state)
    if on_delta is not None:
        on_delta(msg + "\n")
    return msg


def _over_cap_giveup(
    session: Session, state: "_TurnState", on_delta: Callable[[str], None] | None
) -> str:
    """Terminal give-up when compaction cannot get the context under cap.

    Records a truthful synthesized answer as the turn's final answer
    (transcript, ``turn_report``, and on_delta payload), stamps ``verified``, and
    returns it verbatim.
    """
    reason = (
        "This turn was ended by the harness: the context went over the model's "
        "token budget and compaction could not reduce it further."
    )
    return _finalize_giveup(
        session, state, reason, "start a new session or shorten the request.", on_delta
    )


def _blocked_loop_giveup(
    session: Session, state: "_TurnState", on_delta: Callable[[str], None] | None
) -> str:
    """Terminal give-up when the same tool call is blocked in a tight loop.

    The escalation ladder above the repeat-call hard cap: once the harness has
    refused ``_BLOCKED_STREAK_CAP`` identical calls in a row with no dispatch in
    between (the model ignored both the block error and the fold-surviving steer),
    further chat rounds only burn a full LLM round-trip per blocked call. This
    ends the turn instead — mirroring ``_over_cap_giveup``: records a truthful
    synthesized answer as the turn's final answer (transcript,
    ``turn_report``, and on_delta payload), stamps ``verified``, and returns it
    verbatim.
    """
    print(
        ui.telemetry(
            f"loop-guard: escalated — turn force-finalized after "
            f"{state.blocked_streak} consecutive blocked calls"
        ),
        file=sys.stderr,
    )
    reason = (
        f"This turn was ended by the harness: the same tool call was repeated "
        f"and blocked {state.blocked_streak} times in a row with no progress."
    )
    return _finalize_giveup(
        session, state, reason, "rephrase or split the request and try again.", on_delta
    )


def _text_loop_giveup(
    session: Session,
    state: "_TurnState",
    emissions: int,
    on_delta: Callable[[str], None] | None,
) -> str:
    """Terminal give-up when the model keeps re-emitting identical narration.

    The third runaway bound, beside the repeat-call hard cap and its escalation
    ladder: those two count *calls*, so a model that re-narrates the same plan
    verbatim while varying its calls just enough to stay under the identical-call
    tally slips both. ``_text_runaway_count`` catches that on tool-bearing rounds
    only; this ends the turn the same way as the blocked-loop path — a truthful
    synthesized answer recorded across transcript, ``turn_report`` and on_delta,
    with ``verified`` stamped — so a narration loop is reported honestly instead
    of burning rounds until an external timeout.

    The looping assistant message is deliberately NOT appended before this runs:
    dropping it keeps the transcript free of a ``tool_calls`` row with no
    matching tool results, which a resumed session could not send.
    """
    print(
        ui.telemetry(
            f"loop-guard: escalated — turn force-finalized after "
            f"{emissions} identical assistant messages"
        ),
        file=sys.stderr,
    )
    reason = (
        f"This turn was ended by the harness: the model sent the same message "
        f"{emissions} times while still calling tools, making no progress."
    )
    return _finalize_giveup(
        session, state, reason, "rephrase or split the request and try again.", on_delta
    )


def _render_report_answer(payload: dict) -> str:
    """Render an accepted ``report`` payload as the turn's prose answer.

    The structured payload is what a machine caller reads (out of the envelope's
    ``verdict`` / ``plan`` / ``assertions`` / ``observations`` fields); this is
    the same content for a human reading stdout or the transcript. Nothing is
    truncated and nothing is inferred — every line comes from a field the
    evidence gate in ``tools/report.py`` already validated.
    """
    verdict = payload.get("verdict", "inconclusive")
    plan = payload.get("plan") or []
    assertions = payload.get("assertions") or []
    observations = (payload.get("observations") or "").strip()

    lines = [f"VERDICT: {verdict}"]

    if plan:
        lines.append("")
        lines.append("Plan:")
        lines.extend(f"  {i}. {step}" for i, step in enumerate(plan, 1))

    if assertions:
        lines.append("")
        lines.append("Assertions:")
        for i, item in enumerate(assertions, 1):
            lines.append(f"  {i}. [{item.get('verdict')}] {item.get('assertion')}")
            evidence = item.get("evidence") or {}
            detail = (evidence.get("detail") or "").strip()
            if detail:
                lines.append(f"     evidence ({evidence.get('kind')}): {detail}")
            artifact = evidence.get("artifact")
            if artifact:
                lines.append(f"     artifact: {artifact}")

    if observations:
        lines.append("")
        lines.append("Observations:")
        lines.append(f"  {observations}")

    return "\n".join(lines)


def _report_finalize(
    session: Session, state: "_TurnState", on_delta: Callable[[str], None] | None
) -> str:
    """Terminal-tool exit: an accepted ``report`` call ends the verify turn.

    Only reached once ``tools/report.py`` has ACCEPTED the payload — the
    evidence gate (no unevidenced pass/fail, no incoherent top-level verdict)
    has already run, so this never has to re-judge the content. A rejected
    report never sets ``state.terminal_report``, so the model keeps its turn and
    can fix the verdict.

    Mirrors the give-up exits' contract: records the rendered report as the
    turn's final answer across the transcript, ``turn_report["answer"]``, and
    ``on_delta``, publishes the structured payload under
    ``turn_report["report"]`` for the one-shot envelope, and stamps
    ``verified`` through the one shared formula.

    ``verified`` stays tri-state and is NOT set from the verdict: it answers
    "did this turn verify the files it changed", and a verify run changes no
    files, so it correctly reports ``None``. The verification outcome the caller
    wants is ``verdict``, which is a separate field precisely so the two cannot
    be confused.
    """
    payload = state.terminal_report or {}
    msg = _render_report_answer(payload)
    session.append_assistant(msg)
    state.turn_report["answer"] = msg
    state.turn_report["report"] = {
        "verdict": payload.get("verdict"),
        "plan": payload.get("plan") or [],
        "assertions": payload.get("assertions") or [],
        "observations": payload.get("observations") or "",
    }
    _stamp_turn_outcome(state)
    if on_delta is not None:
        on_delta(msg + "\n")
    return msg


def _finalize_answer(
    session: Session,
    state: "_TurnState",
    response: ChatResponse,
    on_delta: Callable[[str], None] | None,
) -> str | None:
    """Run the no-tool-call gate cascade and return the turn's answer.

    In order: bounce once on an empty answer (empty-answer nudge), bounce once
    on an unverified file mutation (H1 verification nudge), then — on the second
    pass through — mark the S3 verify gate, surface an empty-answer placeholder,
    and deliver the answer through on_delta.

    Returns the final answer string (terminal — the caller returns it), or
    ``None`` when a nudge bounced (the caller loops again). ``append_assistant``
    for this response has already run in ``handle_user_message`` before this
    call, so the transcript holds the model's original, unprefixed text. The
    end-of-turn consolidation hook is NOT fired here — ``handle_user_message``
    owns it as the single choke point across every turn-exit path.
    """
    text = response.text or ""

    # Empty-answer retry: the model ended the turn with no text and no tool
    # calls — a common local-model failure mode (a bare stop token after
    # consuming tool results) that REPL mode would silently swallow (it discards
    # the return value and only on_delta delivers output, so an empty return
    # leaves the user at a blank prompt with the tool calls having visibly run).
    # Inject a harness steer and loop once more rather than re-rolling the
    # identical request (which risks a deterministic re-collapse). Bounded to a
    # single retry per turn; a second empty turn is surfaced via the placeholder
    # below.
    if not text.strip() and not state.empty_answer_nudge_fired:
        state.empty_answer_nudge_fired = True
        print(
            ui.telemetry("empty-answer-nudge: fired (empty assistant turn)"),
            file=sys.stderr,
        )
        session.append_steer(
            "You produced no answer this turn. Respond now with a concise "
            "summary of what you did or found, grounded in the tool results "
            "above. Do not call more tools unless a result is genuinely missing."
        )
        return None

    # H1 — post-mutation verification nudge: the model is about to end the turn
    # having mutated files without running anything to verify the change. Inject
    # a harness steer and do one more loop iteration instead of returning. Fires
    # at most once per turn, and only names verification tools the active mode
    # actually carries (_available_verification_tools). If none are available —
    # unreachable in practice, since every mode with a mutating tool also carries
    # run_command — there is nothing truthful to steer with, so this falls
    # straight through to the S3 gate below instead of bouncing on a toolless
    # nudge.
    if state.needs_verification and not state.verification_nudge_fired:
        state.verification_nudge_fired = True
        available = _available_verification_tools()
        if available:
            print(
                ui.telemetry("verification-nudge: fired (unverified file mutation)"),
                file=sys.stderr,
            )
            session.append_steer(_verification_nudge_text(available))
            return None
        print(
            ui.telemetry(
                "verification-nudge: skipped (no verification tool in active mode)"
            ),
            file=sys.stderr,
        )

    # S3 — hard verify gate: this is the SECOND final answer of the turn (the
    # bounce above already fired and needs_verification is still set — a
    # run_tests/run_command/verify_scratch call never succeeded in between).
    # Accept it, but mark it: prefix the *returned* text with a harness-side
    # "[UNVERIFIED CHANGES] " so the caller sees the state, unless the model
    # already declared the change unverified in its own words (case-insensitive
    # match). The transcript already recorded the model's original, unprefixed
    # text — only the return value / on_delta payload gets the marker.
    final_text = text
    # S4 — record the UNPREFIXED answer before any marker is layered on; this is
    # the single source of truth the --json envelope reads back via
    # session.turn_report.
    state.turn_report["answer"] = final_text
    if state.needs_verification and state.verification_nudge_fired:
        state.turn_report["declared_unverified"] = "unverified" in final_text.lower()
        if not state.turn_report["declared_unverified"]:
            final_text = "[UNVERIFIED CHANGES] " + final_text
            print(
                ui.telemetry(
                    "verification-gate: unresolved after bounce — "
                    "marking [UNVERIFIED CHANGES]"
                ),
                file=sys.stderr,
            )
    # Net-change annotation + one shared formula for verified, both stamped in
    # order at this single turn-exit choke point via _stamp_turn_outcome (shared
    # with the give-up paths) so every exit reports the same truth the same way.
    _stamp_turn_outcome(state)

    # Empty-answer placeholder: if the model returned no text at all (the retry
    # above already fired once and still came back empty), surface a transparent
    # placeholder. Layered onto the return value / on_delta payload ONLY — the
    # transcript and turn_report["answer"] already hold the real empty string as
    # the source of truth. Uses the original response text (not the
    # possibly-prefixed final_text) so it overrides a vacuous
    # "[UNVERIFIED CHANGES] " prefix too.
    if not text.strip():
        final_text = "(no response from model)"
        print(
            ui.telemetry("empty-answer: no text after retry — surfacing placeholder"),
            file=sys.stderr,
        )

    if on_delta is not None and final_text:
        if state.streamed == 0:
            on_delta(final_text)  # non-streaming / gated fallback: deliver whole
        on_delta("\n")
    return final_text
