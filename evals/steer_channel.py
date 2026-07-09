"""Dispatch-level check of the harness steer channel (no LLM required).

Harness turn guidance (the empty-answer bounce, the H1 verification nudge) is
injected via ``Session.append_steer`` rather than as a plain user message, so
the model, the compaction summarizer, and any transcript reader can all tell
harness guidance from real human input. This script drives that channel
directly and asserts its four load-bearing properties:

    (a) append_steer persists a ``user``-role message whose content carries the
        ``session.STEER_PREFIX`` prefix and which is flagged ``steer`` — and it survives a
        transcript round-trip (Session.resume).
    (b) the provider payload built from an assembled context (llm's single wire
        boundary) carries NO ``steer`` key, and the steer's content still leads
        with ``session.STEER_PREFIX``.
    (c) compaction's summarizer-input serialization labels the steer ``harness``,
        never ``user``, so summaries never quote it as the user's own words.
    (d) a plain append_user message is untouched: no ``steer`` key, no prefix,
        rendered as ``user`` for the summarizer.

Exits 0 on success, 1 on any assertion failure. Runs with the repo root on
``sys.path`` (evals/run.py inserts it before exec'ing this file).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    from compaction import _render_messages_for_summary
    from llm import to_wire_messages
    from session import STEER_PREFIX, Session

    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="steer-eval-") as tmp:
        root = Path(tmp)
        session = Session(root, "test-model", "You are a test agent.")
        session.append_user("Fix the parser bug in parser.py.")
        session.append_steer(
            "You modified files this turn but ran nothing to verify the change."
        )
        session_id = session.session_id
        session.close()

        # (a) persisted shape + round-trip through Session.resume.
        resumed = Session.resume(root, "test-model", "You are a test agent.", session_id)
        try:
            msgs = resumed._messages
            user_msg = next((m for m in msgs if m.get("content", "").startswith("Fix the")), None)
            steer_msg = next((m for m in msgs if m.get("steer")), None)

            if steer_msg is None:
                failures.append("no message carries the steer flag after round-trip")
            else:
                if steer_msg.get("role") != "user":
                    failures.append(f"steer role is {steer_msg.get('role')!r}, expected 'user'")
                if not str(steer_msg.get("content", "")).startswith(STEER_PREFIX):
                    failures.append(
                        f"steer content lacks the STEER_PREFIX marker: "
                        f"{steer_msg.get('content')!r}"
                    )
                if steer_msg.get("steer") is not True:
                    failures.append("steer flag did not survive the transcript round-trip")

            # (d) plain user message is unaffected.
            if user_msg is None:
                failures.append("plain user message missing after round-trip")
            else:
                if "steer" in user_msg:
                    failures.append("plain user message wrongly carries a steer flag")
                if user_msg.get("content", "").startswith(STEER_PREFIX):
                    failures.append("plain user message wrongly carries the STEER_PREFIX marker")

            # (b) wire boundary strips the steer key; content still leads with prefix.
            wire = to_wire_messages(resumed.assemble_context())
            if any("steer" in m for m in wire):
                failures.append("wire payload still carries a steer key")
            wire_steer = next(
                (m for m in wire if str(m.get("content", "")).startswith(STEER_PREFIX)), None
            )
            if wire_steer is None:
                failures.append("wire payload lost the STEER_PREFIX-marked steer content")

            # (c) summarizer serialization labels the steer 'harness', user 'user'.
            rendered = _render_messages_for_summary(resumed._messages)
            if "harness: " + STEER_PREFIX + "You modified files" not in rendered:
                failures.append(
                    "summarizer serialization did not label the steer as 'harness'"
                )
            if "user: Fix the parser bug" not in rendered:
                failures.append(
                    "summarizer serialization did not label the plain message as 'user'"
                )
        finally:
            resumed.close()

    for f in failures:
        print(f"FAIL: {f}")
    if not failures:
        print("steer_channel inline checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
