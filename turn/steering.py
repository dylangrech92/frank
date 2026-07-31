from __future__ import annotations


# Escalation ladder above the hard cap. Once the cap starts refusing an
# identical call, a determined model can re-issue it every round — each a full
# LLM round-trip that dispatches nothing. Worse, when compaction drops the block
# error and the tool result from context, the model loses even the feedback that
# it is stuck (only a fold-surviving steer on the user row remains). After this
# many consecutive blocked calls with no dispatch in between, the turn is
# force-finalized (see _blocked_loop_giveup / handle_user_message).
_BLOCKED_STREAK_CAP = 3

# Fold-surviving steer for a blocked round, worded plainly (small local models
# treat bracketed tags as noise) and spelling out that it is an automated
# harness message, NOT the user's — append_steer already prepends STEER_PREFIX,
# so this is the body only. It rides the user role, which _prune_messages keeps
# across a compaction boundary, so the feedback survives even when the block
# error and tool result behind it are folded away.
_BLOCKED_ROUND_STEER = (
    "The last tool call was blocked because it has already run {n} times this "
    "turn with identical arguments. If you were re-issuing it because the earlier "
    "result is no longer visible, that result was removed from context to save "
    "space — do not repeat the call. Use what you already know, take a different "
    "action, or give your final answer now."
)

# Fold-surviving reproduce-before-edit steer, same wire-safety and plain-
# language rules as _BLOCKED_ROUND_STEER (append_steer prepends STEER_PREFIX, so
# this is the body only — no duplicate "not from the user" preamble). Fires on
# the first relevant file mutation of a turn that has run nothing to observe the
# problem, steering observed-output-first debugging over assumption-driven edits.
_REPRO_BEFORE_EDIT_STEER = (
    "The edit you just made was applied successfully and is already in the files. "
    "Do not re-check whether the original request still applies — it does, and "
    "your edit is part of it. Before editing anything else, run the relevant "
    "command with run_command and read its actual output: for a reported bug, "
    "crash, or wrong output that means reproducing the failure; otherwise it "
    "means running the code to confirm your change. Base any further edits on "
    "that observed output, not on assumption. The run_command tool is loaded "
    "into your toolset now — call it directly."
)

# Fold-surviving no-failure-observed steer, same wire-safety and plain-language
# rules as the steers above (append_steer prepends STEER_PREFIX, so this is the
# body only). Fires on the turn's first relevant file mutation when the task
# reports a failure but every run so far this turn that exercised the project has
# passed — i.e. the model is starting to "fix" a failure it has never actually
# seen fail. Mutually exclusive with the reproduce-before-edit steer above (that
# one owns the zero-runs case; this one owns the runs-all-passed case, including
# a turn where the only failing runs were shell-level environment noise that
# never reached project code — see _run_failure_is_environment_noise). The first
# two sentences and the directive from "Do not fix code you have not seen fail"
# onward are empirically tuned against reward-hacking and must stay verbatim.
_NO_FAILURE_OBSERVED_STEER = (
    "The edit you just made was applied successfully and is already in the files. "
    "Do not re-check whether the original request still applies — it does. The "
    "task reports a failure, but every command and test run this turn that "
    "exercised this project has passed: the reported failure has never been "
    "observed on this project as it stands. "
    "Do not fix code you have not seen fail, and do not modify files or data to "
    "force a failure — a failure you manufacture is not the reported failure. Run "
    "the reported failing command on the project exactly as it is; if it passes, "
    "revert your edit and state in your final answer that the reported problem "
    "could not be reproduced."
)


def _is_bug_report(message: str) -> bool:
    """True when the user's request reads as a bug report (case-insensitive).

    A plain substring match against ``_BUG_REPORT_LEXICON``. Keeps the
    no-failure-observed steer out of pure feature work — it only makes sense to
    warn "you are fixing a failure you never saw fail" when the request actually
    claims a failure.
    """
    low = message.lower()
    return any(term in low for term in _BUG_REPORT_LEXICON)


# Per-tool clause for the H1 post-mutation verification nudge below, keyed by
# name so the nudge names exactly the verification tools the active mode
# carries (see _available_verification_tools) instead of hard-coding all
# three — code mode has no run_tests, performance_debug has neither run_tests
# nor verify_scratch.
_VERIFICATION_TOOL_CLAUSE = {
    "run_command": "run_command against a separate script",
    "run_tests": "run_tests",
    "verify_scratch": "verify_scratch (a throwaway snippet, no file pollution)",
}


def _join_with_or(items: list[str]) -> str:
    """Join *items* with commas and a trailing "or", oxford-comma style."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def _verification_nudge_text(available: list[str]) -> str:
    """Build the H1 post-mutation verification nudge from the mode's actual tools.

    ``available`` is ``_available_verification_tools()`` — a sorted, non-empty
    subset of ``_VERIFICATION_TOOLS`` — so this only ever names tools the model
    can actually call in the active mode. Wording is preserved verbatim from
    the original hard-coded steer; only which tools are named, and the
    singular/plural framing of the closing sentence, vary with the list.
    """
    tools_text = _join_with_or([_VERIFICATION_TOOL_CLAUSE[name] for name in available])
    toolset_sentence = (
        "This tool is loaded into your toolset now — call it directly."
        if len(available) == 1
        else "These tools are loaded into your toolset now — call one directly."
    )
    return (
        "You modified files this turn but ran nothing to verify the "
        f"change. Verify it now with {tools_text} — never by adding "
        "repro/test code to a production file or repurposing its "
        "`if __name__ == \"__main__\"` block. "
        f"{toolset_sentence} Or state explicitly "
        "in your answer that the change is unverified. Either way, end "
        "your answer with a one-line verification breakdown: what "
        "you checked (tests, commands, diagnostics) and what it "
        "showed."
    )


# Word/phrase lexicon (case-insensitive substring) that marks a user request as a
# bug report rather than a pure feature task. Gates the no-failure-observed steer:
# warning "you are fixing a failure you never saw fail" only makes sense when the
# request actually claims a failure.
_BUG_REPORT_LEXICON = (
    "crash",
    "error",
    "exception",
    "traceback",
    "bug",
    "broken",
    "fails",
    "failing",
    "failure",
    "regression",
    "wrong output",
    "incorrect",
)
