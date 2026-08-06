"""One chat call, made only once the context provably fits under the cap.

``_run_llm_with_compaction`` is the single place a turn talks to the model. It
owns the whole pre-flight ladder — the summarizer loop with its thrash guard,
then the free ``force_fold`` last resort — and the ``OverCapError`` retry for
when the provider rejects the request despite the estimate. ``_record_llm_usage``
closes that loop: it folds the response's real token counts back into the
calibration the next pre-flight estimate is built from. The two live together
because the ladder's accuracy depends on the recording running after every call.
"""

from __future__ import annotations

import sys
from typing import Callable

import compaction
import ui
from llm import ChatResponse, LLMClient, OverCapError, StreamStalledError
from session import Session
from turn.outcome import _over_cap_giveup
from turn.state import _TurnState

# How many times one logical LLM call may be re-issued after its SSE stream
# died mid-drain (StreamStalledError). Kept small: a transient backend hiccup
# recovers on the first retry, a wedged backend won't recover on any — and
# each stalled attempt can already burn the full per-read-gap timeout.
_MAX_STREAM_STALL_RETRIES = 2


def _record_llm_usage(
    session: Session,
    state: "_TurnState",
    response: ChatResponse,
    est: int,
    context: list[dict],
) -> None:
    """Fold a completed chat response's token usage into session + turn stats.

    S2 real ``prompt_tokens`` usage remembered together with where this context
    ended so the next trigger check can compose real + delta, S4 per-turn usage
    accumulation on ``turn_report``, and the cumulative stats.json row. Usage
    fields stay ``None`` all turn on providers that never report usage.
    """
    if response.prompt_tokens is not None:
        # S2 — remember the real prompt_tokens (plus where this context ended)
        # so the next pre-flight trigger check can compose real + delta instead
        # of re-estimating the whole transcript from scratch.
        session.last_prompt_tokens = response.prompt_tokens
        session.last_prompt_context_len = len(context)
        print(
            ui.telemetry(
                f"usage: actual prompt_tokens={response.prompt_tokens} "
                f"vs estimated ~{est} tokens (delta {response.prompt_tokens - est:+d})"
            ),
            file=sys.stderr,
        )

    # S4 — sum usage across every LLM call this turn (None until the first
    # real figure arrives, then a running total).
    if response.prompt_tokens is not None:
        state.turn_report["usage"]["prompt_tokens"] = (
            (state.turn_report["usage"]["prompt_tokens"] or 0) + response.prompt_tokens
        )
    if response.completion_tokens is not None:
        state.turn_report["usage"]["completion_tokens"] = (
            (state.turn_report["usage"]["completion_tokens"] or 0)
            + response.completion_tokens
        )

    # Usage stats: fold this LLM call into the session's cumulative stats.json
    # row (run time, tool-call count, token totals).
    session.record_llm_call(
        response.prompt_tokens,
        response.completion_tokens,
        len(response.tool_calls),
    )


def _remeasure(
    session: Session,
    state: "_TurnState",
    tool_schemas: list[dict],
    step: str | None = None,
) -> tuple[list[dict], int, int]:
    """Re-assemble the context and re-estimate its token signal.

    The single measurement atom used at the top of each loop pass and after
    every compaction/force-fold step. Returns the assembled ``context``, its
    raw ``estimate_tokens`` figure, and the ``trigger_estimate`` to compare
    against ``state.cap``. ``step`` tags the telemetry line with the step
    counter (``"post-compaction"`` / ``"force-fold"``) or is ``None`` for the
    plain pre-flight measurement.
    """
    context = session.assemble_context()
    est = compaction.estimate_tokens(context, tool_schemas)
    trigger = compaction.trigger_estimate(session, context, tool_schemas)
    tag = "" if step is None else f" [{step} #{state.compactions}]"
    print(
        ui.telemetry(
            f"context: {len(context)} messages, ~{est} tokens (cap {state.cap}){tag}"
        ),
        file=sys.stderr,
    )
    return context, est, trigger


def _run_llm_with_compaction(
    session: Session,
    client: LLMClient,
    state: "_TurnState",
    tool_schemas: list[dict],
    delta_cb: Callable[[str], None] | None,
    verbose: bool,
    on_delta: Callable[[str], None] | None,
) -> ChatResponse | str:
    """Compact the assembled context under cap, then make one chat call.

    Runs the full pre-flight compaction ladder (summarizer loop with a thrash
    guard, then the free force_fold last resort), makes one ``client.chat``
    call per attempt, and folds the successful response's usage into the
    session/turn stats. When the provider rejects the request with ``OverCapError`` despite
    the estimate, it compacts once and retries the whole cycle. When the SSE
    stream dies without a complete response (``StreamStalledError``), the call
    is re-issued up to ``_MAX_STREAM_STALL_RETRIES`` times — the context is
    unchanged by the failed call, and this is the one caller that may retry
    past forwarded deltas: fragments already streamed belong to the abandoned
    response, the retried call re-delivers the whole answer, and the telemetry
    line between them makes the discard visible.

    Returns the ``ChatResponse`` on success, or the over-cap give-up string
    (already delivered to the transcript/on_delta) when compaction is exhausted
    — the caller returns that string as the turn's answer.
    """
    # Thrash guard: bail out of the summarizer loop after this many consecutive
    # compactions that failed to reduce the context (further calls won't
    # converge) and fall through to the force_fold last resort.
    max_non_shrink = 2
    stream_stalls = 0

    while True:
        # Pre-flight: keep the assembled request at or below the shared cap,
        # compacting older turns before the call is ever made.
        context, est, trigger = _remeasure(session, state, tool_schemas)
        # S2 — usage-driven trigger: real prompt_tokens from the last response
        # plus an estimate of what was appended since, when available;
        # otherwise the plain whole-context estimate. See
        # compaction.trigger_estimate for the composition rationale.
        if session.last_prompt_tokens is not None:
            print(
                ui.telemetry(
                    f"trigger: ~{trigger} tokens (real {session.last_prompt_tokens} "
                    f"+ delta)"
                ),
                file=sys.stderr,
            )
        non_shrink_streak = 0
        while trigger > state.cap:
            if not compaction.compact(
                session, client, state.window, state.comp_cfg
            ):
                # Summarization exhausted (budget or nothing foldable) — break
                # to the force_fold last resort rather than dead-ending.
                break
            state.compactions += 1
            # Compaction reshaped the assembled context (summary spliced in),
            # so the prior real-usage baseline's index no longer lines up.
            session.last_prompt_tokens = None
            prev_trigger = trigger
            context, est, trigger = _remeasure(session, state, tool_schemas, "post-compaction")
            # Thrash guard: stop paying for summarizer calls that aren't
            # reducing the context. Two consecutive no-reduction compactions
            # mean further calls won't converge — fall through to force_fold.
            if trigger >= prev_trigger:
                non_shrink_streak += 1
                if non_shrink_streak >= max_non_shrink:
                    break
            else:
                non_shrink_streak = 0

        # Overflow ladder: summarization could not get under cap. As a last
        # resort, drop all but the most recent messages with no LLM call (older
        # context is lost rather than failing the turn), then re-check.
        if trigger > state.cap and compaction.force_fold(session, state.comp_cfg):
            state.compactions += 1
            session.last_prompt_tokens = None
            context, est, trigger = _remeasure(session, state, tool_schemas, "force-fold")
        if trigger > state.cap:
            return _over_cap_giveup(session, state, on_delta)

        if verbose:
            msg_lines: list[str] = []
            for i, m in enumerate(context):
                role_val = str(m.get("role", ""))
                content_val = str(m.get("content", ""))
                msg_lines.append(
                    f"assembled context\n[{i}] {role_val}: {content_val}"
                )
            print("\n".join(msg_lines), file=sys.stderr)

        # S3 — hard verify gate: once the H1 bounce has fired and the mutation
        # is still unresolved, this call's answer may need a harness-added
        # "[UNVERIFIED CHANGES] " prefix decided *after* the response is fully
        # in hand. Streaming the raw text through on_delta as it arrives would
        # violate the "sink receives the final text exactly once" contract (the
        # prefix must lead, and it can't be inserted retroactively into an
        # already-streamed prefix-less stream). So this one call is buffered
        # (delta_cb withheld) and delivered whole — with or without the marker —
        # once the gate decision is made in _finalize_answer. Any iteration
        # where the gate isn't in this pending state streams exactly as before.
        try:
            state.streamed = 0
            gate_pending = state.verification_nudge_fired and state.needs_verification
            chat_delta_cb = None if gate_pending else delta_cb
            state.turn_report["usage"]["llm_calls"] += 1
            response: ChatResponse = client.chat(context, tool_schemas, chat_delta_cb)
        except OverCapError:
            # Provider rejected on length despite the estimate — compact and
            # retry; if summarization can't help, fall back to force_fold.
            if compaction.compact(
                session, client, state.window, state.comp_cfg
            ):
                state.compactions += 1
            elif compaction.force_fold(session, state.comp_cfg):
                state.compactions += 1
            else:
                return _over_cap_giveup(session, state, on_delta)
            # Same reasoning as the pre-flight compaction path above: the
            # baseline this real measurement was keyed to no longer applies.
            session.last_prompt_tokens = None
            continue
        except StreamStalledError as exc:
            # The SSE stream died without a complete response (llm.py already
            # burned its own pre-first-token attempt). The failed call changed
            # nothing in the session, so re-issuing it is safe; any fragments
            # already streamed belong to the abandoned response and the retried
            # call re-delivers the whole answer. state.streamed resets at the
            # top of the try, so the answer-already-streamed accounting starts
            # fresh with the retry.
            stream_stalls += 1
            if stream_stalls > _MAX_STREAM_STALL_RETRIES:
                raise
            print(
                ui.telemetry(
                    f"llm stream stalled ({exc}); re-issuing call "
                    f"(retry {stream_stalls}/{_MAX_STREAM_STALL_RETRIES})"
                ),
                file=sys.stderr,
            )
            continue

        state.est = est
        _record_llm_usage(session, state, response, est, context)
        return response
