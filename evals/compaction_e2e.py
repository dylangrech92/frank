"""Deterministic end-to-end check of the harness's compaction guarantees.

Replaces the old live ``compaction_coherence`` scenario, whose trigger depended
on how verbosely a real local model happened to answer (so it passed only
occasionally). This drives the *real* production turn loop
(``agent.handle_user_message``) with a scripted stub LLM under a tiny
``context_limit``, so the compaction ladder fires on every run and the
harness-side guarantees can be asserted exactly. Model-recall *quality* — whether
a live model still remembers folded facts — is intentionally out of scope here;
that is a property of the model, not the harness, and cannot be gated
deterministically.

Two scenarios, each in its own throwaway temp project dir (chdir'd so read_file
resolves against cwd; the cwd is restored afterward):

    Normal fold — the stub inflates the transcript with long assistant turns
        until compaction fires, the summarizer returns a short canned digest,
        and the stub then emits a plain final answer. Asserts (failure-list
        style):
          1. trigger + splice — the summarizer was called at least once, the
             watermark advanced, and the reassembled context now fits the cap.
          2. coherence anchor — the original user request survives verbatim in
             the assembled context (re-injected as the task anchor once folded
             behind the watermark), so the goal is never lost.
          3. tail hygiene — no ``tool`` rows and no ``tool_calls`` from before
             the fold watermark survive in the assembled tail.
          4. turn completes — the scripted final answer is returned (not the
             over-cap give-up, not an exception) and ``turn_report['answer']``
             matches it.

    Force-fold overflow — the summarizer is USELESS (it echoes its own input,
        so ``compact`` never shrinks the context). Asserts the turn still
        terminates with a string (the over-cap give-up, never a raise or hang),
        that the summarizer loop ran and thrashed (>= 1 summarizer call), and
        that the free ``force_fold`` last resort engaged (its telemetry appears
        on stderr).

Exits 0 on success, 1 on any assertion failure or unexpected exception. Runs
with the repo root on ``sys.path`` (evals/run.py inserts it before running this
file) and touches no repo files.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

from evals._stub import _StubClient, disable_memory_hooks

# A distinctive original request — checked verbatim to prove the task anchor is
# re-injected after the user message is folded behind the watermark.
ORIGINAL_TASK = (
    "ZEBRA-ANCHOR-7731: give me a full architectural analysis of the fixture "
    "modules and how their helpers compose."
)

# ~700 tokens of filler per assistant turn — several turns push the assembled
# transcript past the (tiny) cap so the compaction ladder must engage.
_FILLER = (
    "This paragraph is deterministic filler that inflates the running transcript "
    "so the reactive compaction ladder is forced to engage on every run. "
) * 55

# A short, well-formed six-section digest the normal summarizer returns; short
# enough that installing it drops the assembled context back under the cap.
_CANNED_SUMMARY = (
    "## Task\nZEBRA-ANCHOR-7731 architectural analysis of the fixture modules.\n"
    "## State\nSeveral fixture modules were read and discussed.\n"
    "## Files-touched\nf0.py..f9.py — small fixture helpers, read only.\n"
    "## Open\nNothing outstanding.\n"
    "## Decisions\nKeep the analysis grounded in the fixtures.\n"
    "## Last\nAbout to deliver the consolidated analysis.\n"
)

_FINAL_ANSWER = "Consolidated analysis complete: the fixture helpers compose cleanly."

# A digest larger than any cap this scenario uses, on its own — the "useless
# summarizer" installs it so no amount of folding can bring the assembled
# context under the cap (the digest alone already exceeds it), forcing the
# thrash guard and then the free force_fold last resort. Sized well above
# check_force_fold_overflow's cap (measured ~8x margin at the constants
# below), not just relative to _FILLER, since tool-schema overhead is now a
# large fixed part of every estimate and only the message-side of the budget
# is what compaction can ever shrink.
_OVERSIZED_SUMMARY = _FILLER * 16


def _make_fixtures(project_dir: Path, n: int) -> list[str]:
    """Write *n* tiny distinct fixture modules and return their relative names."""
    names = []
    for i in range(n):
        name = f"f{i}.py"
        (project_dir / name).write_text(
            f'"""Fixture helper module {i}."""\n\n\ndef helper_{i}(x):\n    return x + {i}\n',
            encoding="utf-8",
        )
        names.append(name)
    return names


def _inflating_factory(session, fixtures, stop_when_summarized: bool):
    """Return a single adaptive turn factory reused for every scripted call.

    Each call emits a long filler turn plus a *distinct* ``read_file`` call.
    Both the arguments and the filler's leading sentence vary per round, so
    neither the repeat-call guard nor the narration-runaway guard (which force-
    finalizes a turn re-emitting byte-identical prose on tool-bearing rounds)
    ends the turn before compaction can be observed — the loop stays alive and
    the transcript keeps growing. When *stop_when_summarized* is set, the factory
    switches to a plain final answer as soon as a summary exists — i.e. once
    compaction has fired — so the normal scenario terminates cleanly right after
    the fold.
    """
    from llm import ChatResponse, ToolCall

    state = {"n": 0}

    def factory():
        if stop_when_summarized and session._summary:
            return ChatResponse(text=_FINAL_ANSWER, tool_calls=[])
        i = state["n"]
        state["n"] += 1
        tc = ToolCall(
            id=f"call-{i}",
            name="read_file",
            arguments={"path": fixtures[i % len(fixtures)]},
        )
        return ChatResponse(text=f"Analysis pass {i}. {_FILLER}", tool_calls=[tc])

    return factory


def _short_summarizer(messages):
    """Summarizer stub that installs a short digest (compaction actually shrinks)."""
    from llm import ChatResponse

    return ChatResponse(text=_CANNED_SUMMARY)


def _useless_summarizer(messages):
    """Summarizer stub that returns an oversized digest, so ``compact`` never shrinks.

    A real summarizer condenses; this one installs a digest larger than the test
    window, so every fold leaves the assembled context over cap — the thrash
    guard trips and the ladder falls through to the free force_fold last resort.
    """
    from llm import ChatResponse

    return ChatResponse(text=_OVERSIZED_SUMMARY)


def check_normal_fold() -> list[str]:
    """Compaction fires, splices a digest under cap, keeps the anchor, and ends."""
    disable_memory_hooks()
    import agent
    import compaction
    import tools.registry as registry
    from session import Session

    # agent.handle_user_message dispatches read_file through the real
    # registry, which requires an active mode (registry.schemas() raises
    # RuntimeError otherwise). 'research' is the narrowest mode that still
    # contains read_file (a _COMMON_TOOLS member present in every mode) —
    # this scenario is read-only by construction.
    saved_mode = registry.current_mode()
    registry.activate_mode("research")

    failures: list[str] = []
    # Tool schemas are counted in every context estimate (compaction.py:
    # "Tool schemas are always counted — they occupy real context on every
    # call"), and mode-gating now loads a mode's *entire* declared tool set
    # for the whole process rather than a small catalog-selected subset —
    # 'research' alone is already several thousand tokens of schema. window
    # must clear that fixed floor with real room to spare for several filler
    # turns, or the very first cap check fails before the scenario's
    # intended flow (several turns, then a fold) ever runs.
    window = 12000
    comp_cfg = {
        "reserve_ratio": 0.1,
        "reserve_min_tokens": 500,
        "keep_recent_messages": 2,
    }
    cap = compaction.compute_cap(window, comp_cfg)

    original_cwd = os.getcwd()
    tmp = Path(tempfile.mkdtemp(prefix="compaction-e2e-normal-"))
    os.chdir(tmp)
    try:
        fixtures = _make_fixtures(tmp, 10)
        session = Session(str(tmp), "test-model", "You are a test agent.")
        factory = _inflating_factory(session, fixtures, stop_when_summarized=True)
        client = _StubClient(
            [factory],
            max_calls=40,
            context_limit=window,
            summarizer=_short_summarizer,
        )

        answer = agent.handle_user_message(
            ORIGINAL_TASK,
            session,
            client,  # pyright: ignore[reportArgumentType]
            compaction_cfg=comp_cfg,
            on_delta=None,
        )

        # 1. trigger + splice
        if client.summarizer_calls < 1:
            failures.append("compaction never fired (no summarizer call)")
        if session._summary_covers <= 0:
            failures.append(
                f"watermark did not advance (covers={session._summary_covers})"
            )
        ctx = session.assemble_context()
        est = compaction.estimate_tokens(ctx, agent.schemas())
        if est > cap:
            failures.append(
                f"reassembled context ~{est} tokens still exceeds the cap {cap}"
            )

        # 2. coherence anchor — the original request survives verbatim.
        joined = "\n".join(str(m.get("content", "")) for m in ctx)
        if ORIGINAL_TASK not in joined:
            failures.append(
                "the original user request was not re-injected verbatim as the "
                "task anchor after being folded behind the watermark"
            )

        # 3. tail hygiene — no tool scaffolding survives past the fold.
        tail = ctx[2:]
        if any(m.get("role") == "tool" for m in tail):
            failures.append("a tool-role row survived past the compaction watermark")
        if any("tool_calls" in m for m in tail):
            failures.append("a tool_calls key survived past the compaction watermark")

        # 4. turn completes with the scripted final answer.
        if answer != _FINAL_ANSWER:
            failures.append(
                f"expected the scripted final answer {_FINAL_ANSWER!r}, got {answer!r}"
            )
        if session.turn_report.get("answer") != _FINAL_ANSWER:
            failures.append(
                f"turn_report['answer']={session.turn_report.get('answer')!r} "
                f"does not equal the scripted final answer"
            )
    finally:
        os.chdir(original_cwd)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    return failures


def check_force_fold_overflow() -> list[str]:
    """A useless summarizer thrashes, force_fold engages, and the turn still ends."""
    disable_memory_hooks()
    import agent
    import tools.registry as registry
    from session import Session

    # See check_normal_fold: agent.handle_user_message needs an active mode
    # to dispatch read_file. 'research' matches this scenario's read-only shape.
    saved_mode = registry.current_mode()
    registry.activate_mode("research")

    failures: list[str] = []
    # A larger window than the normal scenario so several full turns accumulate
    # (transcript length past keep_recent) before the cap is crossed — otherwise
    # compaction would trigger with too few messages to fold and the summarizer
    # loop would never run. Also needs the same fixed-schema-floor headroom as
    # check_normal_fold (see its comment) on top of that.
    window = 18000
    # keep_recent (6) deliberately larger than force_keep_recent (2): the normal
    # fold leaves a tail force_fold can still shrink, so the free last resort has
    # room to engage after the summarizer loop thrashes.
    comp_cfg = {
        "reserve_ratio": 0.1,
        "reserve_min_tokens": 500,
        "keep_recent_messages": 6,
        "force_keep_recent_messages": 2,
    }

    original_cwd = os.getcwd()
    tmp = Path(tempfile.mkdtemp(prefix="compaction-e2e-forcefold-"))
    os.chdir(tmp)
    try:
        fixtures = _make_fixtures(tmp, 10)
        session = Session(str(tmp), "test-model", "You are a test agent.")
        # Settle to a final answer once compaction has fired, so the turn ends
        # cleanly right after force_fold rather than re-inflating indefinitely.
        factory = _inflating_factory(session, fixtures, stop_when_summarized=True)
        client = _StubClient(
            [factory],
            max_calls=40,
            context_limit=window,
            summarizer=_useless_summarizer,
        )

        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            answer = agent.handle_user_message(
                ORIGINAL_TASK,
                session,
                client,  # pyright: ignore[reportArgumentType]
                compaction_cfg=comp_cfg,
                on_delta=None,
            )
        stderr_text = captured.getvalue()

        # The turn must terminate with a string (give-up or a clean final answer
        # after force_fold rescued it) — never a raise or a hang.
        if not isinstance(answer, str) or not answer.strip():
            failures.append(f"turn did not return a non-empty string, got {answer!r}")
        if client.summarizer_calls < 1:
            failures.append(
                "the summarizer loop never ran, so the thrash guard was not exercised"
            )
        if "force-fold" not in stderr_text:
            failures.append(
                "force_fold did not engage (no 'force-fold' telemetry on stderr)"
            )
    finally:
        os.chdir(original_cwd)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    return failures


def main() -> int:
    all_failures: list[str] = []
    for label, fn in (
        ("normal-fold", check_normal_fold),
        ("force-fold", check_force_fold_overflow),
    ):
        try:
            failures = fn()
        except Exception as exc:  # a raised exception is itself a failure
            failures = [f"{label} raised {type(exc).__name__}: {exc}"]
        for f in failures:
            print(f"FAIL [{label}]: {f}", file=sys.stderr)
        all_failures.extend(failures)

    if all_failures:
        return 1
    print(
        "PASS: compaction fires deterministically, splices a digest under cap, "
        "keeps the task anchor and prunes the tail; the overflow ladder "
        "force-folds instead of hanging when the summarizer cannot shrink"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
