"""Shared machinery for the optional ``vision`` provider: build its client, ask it about one image.

Two callers share this module: ``agent._attach_screenshot`` (the automatic
pre-send description of every captured screenshot) and ``tools.vision`` (the
on-demand tool the model can call about any image file). Both send exactly
ONE image in a single no-tools, no-history chat call — the same out-of-band
shape the compaction summarizer uses (this is not a turn, it is one question
with one answer) — and both bill the call to the session the same way.
"""

from __future__ import annotations

import sys

import ui
from llm import LLMClient
from session import Session


# What the vision provider is asked for by default: a description a text-only
# coding agent can verify a page against. Facts only — an invented button or
# an imagined error is worse than no description at all, because the main
# model cannot tell the two apart and will report the invention as evidence.
DESCRIBE_PROMPT = (
    "You are describing an image for a coding agent that cannot see images. "
    "Report only what is actually visible, as fact. Never speculate about "
    "intent, about what is off-screen, or about what the image is supposed "
    "to look like. Cover, in this order: every piece of visible text "
    "(verbatim wherever it is legible); the layout structure (regions, "
    "navigation, forms, tables, buttons and the order they appear in); "
    "colours and styling anomalies (overlapping or clipped elements, "
    "unreadable contrast, broken alignment, unstyled content); any error "
    "message, warning, dialog, modal or overlay, quoted exactly; and "
    "anything still loading, empty, or rendered broken (spinners, "
    "skeletons, missing-image placeholders). If the image is blank or shows "
    "only an error, say exactly that and nothing more."
)


def vision_client() -> LLMClient | None:
    """Build a client for the optional ``vision`` provider, or None if unconfigured.

    Rebuilt from this process's config rather than threaded down from main.py —
    the same construction ``memory/explorer._default_client`` already does, off
    the same ``CODING_AGENT_CONFIG`` env var main.py exports. That keeps every
    entry point (REPL, one-shot, MCP-spawned child, subagent) on one code path
    instead of only whichever ones remembered to pass the client along, at the
    cost of one small JSON read per call — a rounding error next to the
    provider call it precedes.

    A config that cannot be read or parsed means no vision provider is
    configured, which is the pre-existing behaviour: images go to the main
    provider untouched.
    """
    import os

    from config import load as config_load

    try:
        cfg = config_load(os.environ.get("CODING_AGENT_CONFIG", "config.json"))
    except (OSError, ValueError):
        return None
    if cfg.vision is None:
        return None
    return LLMClient(cfg.vision)


def describe_image(
    session: Session,
    client: LLMClient,
    data_uri: str,
    label: str,
    question: str | None = None,
) -> str | None:
    """Ask the vision provider about one image; return its text or None.

    A single-shot call: no tools, no history, no streaming. The image goes to
    THIS provider and nowhere else. With no *question* the provider is asked
    for the general ``DESCRIBE_PROMPT`` description; with one, the same system
    prompt still applies (facts only, no speculation) but the user turn asks
    the question directly instead of requesting a general description — the
    provider answers what was asked, not a generic caption.

    Returns None on any failure, including an empty description: a reasoning
    model that emits only hidden reasoning and no content is a known trap, and
    an empty string spliced into the transcript would read as "the image
    showed nothing" rather than "the description failed".
    """
    ask = f"{label}. {question}" if question else f"{label}. Describe this image."
    request = [
        {"role": "system", "content": DESCRIBE_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ask},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]
    try:
        response = client.chat(request, None)
    except Exception as exc:
        print(
            ui.telemetry(f"vision: describe call failed ({type(exc).__name__}: {exc})"),
            file=sys.stderr,
        )
        return None

    # A billed call, whether or not it produced anything usable — count it the
    # same way compaction counts its summarizer call, with zero tool calls.
    session.record_llm_call(response.prompt_tokens, response.completion_tokens, 0)

    description = (response.text or "").strip()
    if not description:
        print(
            ui.telemetry("vision: describe call returned an empty description"),
            file=sys.stderr,
        )
        return None
    return description
