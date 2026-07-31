from __future__ import annotations

import hashlib
import sys
from typing import Callable

import ui
from session import Session
from turn.state import _TurnState

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
