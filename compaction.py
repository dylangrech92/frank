"""Token estimation and cap computation for reactive context compaction.

Pure helpers with no side effects: an estimator that mirrors what the provider
will actually count (assembled messages *and* tool schemas), and the single
shared cap definition used by both the pre-flight overflow check and the
compaction retry loop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from llm import OverCapError
from session import Session  # noqa: E402


# What one attached screenshot is counted as costing. A vision part is billed by
# the provider as a fixed tile budget, NOT by the length of its base64 payload:
# a 200KB PNG is ~270k base64 characters, which chars/4 would score as ~67k
# tokens against a real cost nearer 1-2k. Left unhandled, a single screenshot
# would appear to blow the window and send the compaction ladder into a fold on
# a context that was never over cap. This is a deliberate over-estimate of the
# real tile cost — over-counting a screenshot is safe, under-counting is not.
_IMAGE_PART_TOKENS = 2000


def _flatten_content(content: Any) -> str:
    """Render a message's ``content`` as countable/summarizable text.

    A plain string passes through. A content-parts list (vision messages — see
    ``Session.append_screenshot``) is flattened part by part, with an
    ``image_url`` part replaced by a fixed-size marker rather than its base64
    data URI: the payload is neither worth summarizing nor countable by
    character length (see ``_IMAGE_PART_TOKENS``).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else json.dumps(content)

    out: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            out.append(json.dumps(part))
        elif part.get("type") == "image_url":
            # Padded to _IMAGE_PART_TOKENS' worth of chars/4 so the estimator
            # bills the image without depending on the base64 length.
            out.append("[screenshot attached]" + ("x" * (_IMAGE_PART_TOKENS * 4)))
        elif part.get("type") == "text":
            out.append(str(part.get("text", "")))
        else:
            out.append(json.dumps(part))
    return "\n".join(out)


def _serialize_for_estimate(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
) -> str:
    """Flatten messages and tool schemas into one text blob for counting.

    Everything that occupies real context is included: each message's role and
    content, any native ``tool_calls`` (name + arguments), tool-result linkage
    fields, and the full JSON of every tool schema. Non-string content is
    JSON-encoded so structured payloads are counted, not skipped.
    """
    parts: List[str] = []

    for m in messages:
        parts.append(str(m.get("role", "")))
        parts.append(_flatten_content(m.get("content", "")))
        name = m.get("name")
        if name:
            parts.append(str(name))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            parts.append(str(fn.get("name", "")))
            parts.append(str(fn.get("arguments", "")))

    if tools:
        for t in tools:
            parts.append(json.dumps(t))

    return "\n".join(parts)


def estimate_tokens(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
) -> int:
    """Estimate the token count of an assembled request.

    Uses ``tiktoken`` (``cl100k_base``) when it can be imported; otherwise falls
    back to the spec's chars/4 heuristic. Tool schemas are always counted — they
    occupy real context on every call.

    Args:
        messages: The assembled message list that would be sent to the provider.
        tools: Optional tool/function schema list included in the request.

    Returns:
        An integer token estimate (at least 1 for any non-empty input).
    """
    text = _serialize_for_estimate(messages, tools)
    if not text:
        return 0

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def trigger_estimate(
    session: "Session",
    context: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
) -> int:
    """Return the compaction-trigger signal for the assembled *context*.

    ``prompt_tokens`` from the most recent provider response only measures the
    prompt of *that* request — the context has grown since (that response's
    own text plus whatever tool results followed it). So when a real
    measurement is available, the signal is composed as::

        trigger = last_real_prompt_tokens + calibrated_estimate(messages appended since)

    rather than trusting the real number alone (stale) or re-estimating the
    whole transcript (throws away the real measurement). ``session``'s
    ``last_prompt_context_len`` marks where that measurement's context ended;
    everything in *context* beyond it is new and only estimated. The estimate
    of that delta (and the whole-context fallback estimate below) is scaled by
    ``session.token_estimate_ratio``, the EMA-calibrated real/estimated ratio,
    so both paths drift toward the provider's real tokenizer over time.

    Falls back to the plain (calibrated) estimate of the whole context when no
    real measurement exists yet, or when the recorded baseline no longer fits
    *context* (e.g. right after a compaction reshaped it — callers reset
    ``last_prompt_tokens`` to None in that case).

    Args:
        session: The active Session carrying the usage/calibration state.
        context: The just-assembled message list for the upcoming request.
        tools: Tool schemas for the upcoming request (counted in the fallback
            estimate; omitted from the delta estimate since the real
            measurement already accounted for them).

    Returns:
        An integer token estimate to compare against the compaction cap.
    """
    if (
        session.last_prompt_tokens is not None
        and session.last_prompt_context_len <= len(context)
    ):
        new_messages = context[session.last_prompt_context_len:]
        delta_est = estimate_tokens(new_messages) if new_messages else 0
        return session.last_prompt_tokens + int(delta_est * session.token_estimate_ratio)

    return int(estimate_tokens(context, tools) * session.token_estimate_ratio)


def update_calibration(
    session: "Session", real_tokens: int, estimated_tokens: int, alpha: float = 0.3
) -> None:
    """Update the session's real-vs-estimate EMA ratio with a new observation.

    Skipped when *estimated_tokens* is non-positive (nothing to divide by;
    the ratio stays at its previous value until a usable observation arrives).

    Args:
        session: The active Session whose ``token_estimate_ratio`` is updated.
        real_tokens: The real ``prompt_tokens`` reported for a request.
        estimated_tokens: The raw (uncalibrated) estimate computed for the
            same request's assembled context.
        alpha: EMA smoothing factor; higher weighs the new observation more.
    """
    if estimated_tokens <= 0:
        return
    observed_ratio = real_tokens / estimated_tokens
    session.token_estimate_ratio = (
        alpha * observed_ratio + (1 - alpha) * session.token_estimate_ratio
    )


def compute_cap(window: int, compaction_cfg: Dict[str, Any] | None = None) -> int:
    """Return the usable token cap below the model's context window.

    The single shared definition::

        cap = window - max(reserve_ratio * window, reserve_min_tokens)

    ``reserve_ratio`` (default 0.10) and ``reserve_min_tokens`` (default 8000)
    are read from the ``compaction`` config block when present.

    Args:
        window: The model's full context window (``llm.context_limit``).
        compaction_cfg: The ``compaction`` config block, or None for defaults.

    Returns:
        The integer token cap; assembled requests should stay at or below it.
    """
    cfg = compaction_cfg or {}
    reserve_ratio = cfg.get("reserve_ratio", 0.10)
    reserve_min = cfg.get("reserve_min_tokens", 8000)
    reserve = max(reserve_ratio * window, reserve_min)
    return int(window - reserve)


# ---------------------------------------------------------------------------
# Reactive compaction: summarize older turns into a fixed six-section digest.
# ---------------------------------------------------------------------------


COMPACTION_SYSTEM_PROMPT = """You are compacting a long software-engineering conversation so it fits the model's context window. Produce a faithful, information-dense summary of everything so far, organized into EXACTLY these six sections, each under its exact markdown heading and in this order:

## Task
The user's overall goal(s) and any explicit requirements or constraints.

## State
What has been accomplished so far and the current state of the work.

## Files-touched
Every file created, edited, or examined, each with a one-line note on what changed or why it mattered. Preserve exact paths.

## Open
Unresolved problems, TODOs, failing tests, or questions still outstanding.

## Decisions
Key technical decisions taken and the reasoning behind them.

## Last
What was happening most recently, in enough detail that work can resume seamlessly.

Rules: keep exact file paths, function/class names, error messages, and concrete values. Drop pleasantries and repetition. Never invent anything that is not present in the conversation. Output only the six sections."""


# How many trailing messages compaction preserves verbatim. Everything older —
# including the older tool-call trail of the *current* in-flight turn — is
# eligible to be folded into the summary. Keeping a recent window lets the model
# continue coherently from what it most recently saw; the summary carries the
# rest. Overridable via the ``compaction`` config block's ``keep_recent_messages``.
DEFAULT_KEEP_RECENT = 8


def _render_messages_for_summary(messages):
    """Flatten messages into readable text for the summarizer's input.

    Harness steer messages (``steer`` flag set — see ``Session.append_steer``)
    are relabeled ``harness`` instead of ``user`` so the summarizer never
    attributes harness guidance to the human. This keeps the "## Task — the
    user's overall goal(s)" section grounded in real user input; a steer's
    ``STEER_PREFIX``-marked content stays visible under the ``harness`` label as
    context the summarizer can weigh but not misread as a stated goal.
    """
    lines = []
    for m in messages:
        role = "harness" if m.get("steer") else m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            # Vision message: keep the label, drop the base64 — the summarizer
            # cannot read an image and must not be fed a data URI as prose.
            content = " ".join(
                str(part.get("text", ""))
                if part.get("type") == "text"
                else "[screenshot attached]"
                for part in content
                if isinstance(part, dict)
            ).strip()
        elif not isinstance(content, str):
            content = json.dumps(content)
        calls = m.get("tool_calls") or []
        if calls:
            names = ", ".join(
                (c.get("function", {}) or {}).get("name", "") for c in calls if isinstance(c, dict)
            )
            content = (content + f"  [tool_calls: {names}]").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _fold_boundary(
    messages: List[Dict[str, Any]],
    summary_covers: int,
    keep_recent: int,
    budget: int,
) -> int | None:
    """Choose the fold boundary: summarize ``[summary_covers:cut]``, keep ``[cut:]``.

    Starts from "keep the last *keep_recent* messages", then folds the oldest
    unfolded messages forward only while the region's rendered token estimate
    stays within *budget* (so the summary call itself never overflows). The kept
    tail must not begin on a ``tool`` message — a tool result whose originating
    ``assistant`` tool-call was folded into the summary would be an orphan the
    provider rejects — so the boundary is walked back off any leading tool row.

    Crucially this treats the current in-flight turn's older tool-call trail as
    foldable, which is what lets a single long turn (e.g. one-shot mode) be
    compacted at all — the old "summarize only completed prior turns" boundary
    was a no-op for a session with just one user message.

    Args:
        messages: The full ``session._messages`` list.
        summary_covers: Watermark — messages before this are already summarized.
        keep_recent: Minimum number of trailing messages to preserve verbatim.
        budget: Token ceiling for the region handed to the summarizer.

    Returns:
        The exclusive fold boundary ``cut`` (``> summary_covers``), or ``None``
        when no valid boundary beyond the watermark exists (cannot make progress).
    """
    max_cut = len(messages) - keep_recent
    if max_cut <= summary_covers:
        return None

    total = 0
    cut = summary_covers
    for i in range(summary_covers, max_cut):
        tokens = estimate_tokens([messages[i]])
        # Always fold at least one message (a lone over-budget row cannot be
        # split); otherwise stop once adding the next row would breach budget.
        if total + tokens > budget and cut > summary_covers:
            break
        total += tokens
        cut = i + 1

    while cut > summary_covers and messages[cut].get("role") == "tool":
        cut -= 1
    if cut <= summary_covers:
        return None
    return cut


def compact(session: "Session", client, window: int, compaction_cfg=None) -> bool:
    """Fold older messages into a running summary and advance the watermark.

    Summarizes the region ``[_summary_covers:cut]`` — where *cut* preserves a
    recent tail and keeps the summarizer's own request within budget (see
    ``_fold_boundary``) — folding any prior summary forward for continuity, then
    installs the result via ``session.set_summary``. Unlike the earlier version,
    the fold boundary may advance *into* the current in-flight turn, so a single
    long turn can be reduced instead of dead-ending in a give-up. The on-disk
    transcript is never touched; only the assembled view shrinks.

    Args:
        session: The active Session.
        client: An LLMClient used to produce the summary.
        window: The model's context window — bounds the summarizer's own request.
        compaction_cfg: The ``compaction`` config block (``keep_recent_messages``,
            plus the shared ``reserve_ratio`` / ``reserve_min_tokens`` cap knobs).

    Returns:
        True if a new summary was produced and installed; False if no foldable
        region remains or the summary call made no progress (caller treats False
        as "cannot compact further").
    """
    cfg = compaction_cfg or {}
    keep_recent = int(cfg.get("keep_recent_messages", DEFAULT_KEEP_RECENT))
    msgs = session._messages

    # Reserve room for the summarizer's fixed overhead (its system prompt and any
    # prior summary folded in) so the region we hand it cannot breach the cap.
    cap = compute_cap(window, cfg)
    prior = session._summary or ""
    fixed = estimate_tokens(
        [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prior},
        ]
    )
    budget = max(cap - fixed, 1)

    cut = _fold_boundary(msgs, session._summary_covers, keep_recent, budget)
    if cut is None:
        return False

    material = _render_messages_for_summary(msgs[session._summary_covers:cut])
    if prior:
        material = (
            "Previous summary of even earlier turns:\n"
            + prior
            + "\n\nConversation since that summary:\n"
            + material
        )

    request = [
        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
        {"role": "user", "content": material},
    ]
    try:
        response = client.chat(request, None)
    except OverCapError:
        # A single row larger than the whole budget can still be handed over
        # (it cannot be split). If the provider rejects even that, compaction
        # genuinely cannot reduce further — report no progress instead of
        # crashing the turn with an unhandled overflow.
        return False
    # The compaction summary is a real (token-billed) LLM call — count it in the
    # session's usage stats too, with zero tool calls.
    session.record_llm_call(response.prompt_tokens, response.completion_tokens, 0)
    summary_text = (response.text or "").strip()

    # A blank summary (e.g. a reasoning model that emitted only hidden reasoning
    # and no content) cannot reduce the context. Do not install it; report that
    # compaction made no progress so the caller gives up honestly instead of
    # silently looping on an empty, ignored summary.
    if not summary_text:
        return False

    session.set_summary(summary_text, cut)
    return True


# Cliff last-resort: when ``compact`` cannot get the context under cap,
# ``force_fold`` advances the watermark past all but the most recent
# ``DEFAULT_FORCE_KEEP_RECENT`` messages with NO summarizer call — accepting
# that dropping old context outright is the lesser evil than failing the turn.
# Smaller than ``DEFAULT_KEEP_RECENT`` so it frees space the normal fold (which
# keeps 8) could not, and the only path that rescues a session with a few very
# large messages where ``compact`` returns False immediately (nothing foldable
# at keep_recent=8). Overridable via the ``compaction`` block's
# ``force_keep_recent_messages``.
DEFAULT_FORCE_KEEP_RECENT = 2

# Summary seeded only when force_fold runs with no prior summary, so
# ``assemble_context``'s tail-slice (keyed on a truthy ``_summary``) engages.
_FORCE_FOLD_MARKER = (
    "Earlier turns were dropped to fit the context window — compaction could "
    "not summarize them down further. Re-read any files you still need."
)


def force_fold(session: "Session", compaction_cfg=None) -> bool:
    """Last-resort free eviction: drop all but the most recent messages, no LLM.

    Advances ``session``'s summary watermark past everything older than the last
    ``force_keep_recent_messages`` (default 2) messages, with no summarizer
    call. Used when ``compact`` cannot get the assembled context under cap: the
    older context is lost outright (no summary covers it) rather than failing
    the whole turn — the lesser evil at the cliff. This is the only path that
    rescues a session with a few very large messages, where ``compact`` returns
    False immediately (nothing foldable at ``keep_recent``=8). Reuses
    ``_fold_boundary`` so the kept tail never begins on an orphaned ``tool`` row.

    Args:
        session: The active Session.
        compaction_cfg: The ``compaction`` config block
            (``force_keep_recent_messages``).

    Returns:
        True if the watermark advanced; False if nothing remained to fold
        (already at or below the force-keep tail), so the caller gives up.
    """
    cfg = compaction_cfg or {}
    keep_recent = int(cfg.get("force_keep_recent_messages", DEFAULT_FORCE_KEEP_RECENT))
    msgs = session._messages
    # No summarizer call, so no input budget — fold as far as _fold_boundary
    # allows (it still walks the boundary back off an orphaning tool row).
    cut = _fold_boundary(msgs, session._summary_covers, keep_recent, float("inf"))
    if cut is None:
        return False
    session.set_summary(session._summary or _FORCE_FOLD_MARKER, cut)
    return True
