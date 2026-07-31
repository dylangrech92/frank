"""Shared stub-LLM scaffolding for end-to-end, network-free eval scripts.

Several evals drive the *real* production turn loop
(``agent.handle_user_message``) but must stay deterministic and offline. They do
that by swapping the live ``LLMClient`` for a scripted stub: a fixed sequence of
canned ``ChatResponse`` factories, one per expected chat call, with a hard
self-cap so a regression that fails to terminate fails fast here instead of
hanging. This module owns the reusable pieces so each eval only writes the parts
that are specific to the guarantee it checks.

Because ``compaction.compact`` invokes the *same* client object for its
summarizer call, ``_StubClient`` also recognizes that call (its request opens
with the compaction system prompt) and, when a *summarizer* callback is
supplied, answers it separately — counted apart from the scripted turn calls and
never advancing the turn script. Evals that keep a large ``context_limit`` (so
compaction never triggers) can ignore the hook entirely; behavior is unchanged
when no summarizer is passed.
"""

from __future__ import annotations

from typing import Callable, Sequence

# The compaction summarizer's request always opens with a system message whose
# content is ``compaction.COMPACTION_SYSTEM_PROMPT``; a prefix of that constant
# is the discriminator that tells a summarizer call apart from a normal turn
# call, imported so a rewording of the prompt cannot silently decouple the two.
from compaction import COMPACTION_SYSTEM_PROMPT as _COMPACTION_PROMPT

_SUMMARIZER_PROMPT_PREFIX = _COMPACTION_PROMPT[:32]

# Monotonic across every factory in the process, so a script assembled from
# several factories cannot mint the same call id twice. Ids must be unique for
# the whole transcript, not just within one factory: the wire protocol pairs a
# tool result to its call id, so a collision silently makes one assistant row
# look answered by another round's result.
_CALL_IDS = {"n": 0}


def _next_call_id(prefix: str) -> str:
    _CALL_IDS["n"] += 1
    return f"{prefix}-{_CALL_IDS['n']}"


class _StubConfig:
    """Minimal stand-in for LLMConfig — only ``context_limit`` is read.

    A large limit keeps compaction dormant so a run exercises whatever ladder it
    targets in isolation; a tiny limit forces the compaction ladder instead.
    """

    def __init__(self, context_limit: int = 200_000) -> None:
        self.context_limit = context_limit


def _is_summarizer_request(messages) -> bool:
    """True when *messages* is a compaction-summarizer request, not a turn call."""
    if not messages:
        return False
    content = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
    return isinstance(content, str) and content.startswith(_SUMMARIZER_PROMPT_PREFIX)


class _StubClient:
    """LLMClient-compatible stub that scripts a fixed sequence of chat responses.

    ``script`` is a list of callables, one per expected turn call, each returning
    the ``ChatResponse`` for that round; the last entry is reused once the script
    is exhausted (so a single trailing final-answer response covers any surplus,
    and a single-element adaptive factory can serve every round). A hard self-cap
    raises after ``max_calls`` turn chats so a regression that fails to terminate
    fails fast here instead of hanging.

    When *summarizer* is supplied, any chat call whose request is a compaction
    summarizer request (see ``_is_summarizer_request``) is routed to it and
    tallied in ``summarizer_calls`` — it neither advances ``script`` nor counts
    against ``max_calls``. When *summarizer* is None (the default), such a call
    would only ever arrive if compaction fired, which the large default
    ``context_limit`` prevents.
    """

    def __init__(
        self,
        script: Sequence[Callable[[], object]],
        max_calls: int = 30,
        context_limit: int = 200_000,
        summarizer: Callable[[list], object] | None = None,
    ) -> None:
        self.config = _StubConfig(context_limit)
        self._script = script
        self.calls = 0
        self._max_calls = max_calls
        self._summarizer = summarizer
        self.summarizer_calls = 0

    def chat(self, messages, tools=None, on_delta=None):
        if self._summarizer is not None and _is_summarizer_request(messages):
            self.summarizer_calls += 1
            return self._summarizer(messages)
        if self.calls >= self._max_calls:
            raise RuntimeError(
                f"stub chat exceeded its {self._max_calls}-call self-cap — the "
                f"turn never terminated (the ladder under test is likely broken)"
            )
        idx = min(self.calls, len(self._script) - 1)
        self.calls += 1
        return self._script[idx]()


def _same_call_response(name: str, arguments: dict, text: str = ""):
    """Return a factory that yields a ChatResponse re-issuing one identical call.

    Each call gets a fresh id (the wire protocol pairs a tool result to its
    call id) but the same name+arguments, so the repeat-call cap counts them as
    identical and the loop-guard eventually blocks them.

    *text* is the assistant prose that rides along with the call — empty by
    default (a tool-only round). Passing a fixed non-empty string reproduces the
    narration loop the identical-call tally cannot see: the same prose emitted
    round after round.
    """
    from llm import ChatResponse, ToolCall

    def factory():
        tc = ToolCall(id=_next_call_id("call"), name=name, arguments=dict(arguments))
        return ChatResponse(text=text, tool_calls=[tc])

    return factory


def _burst_call_response(name: str, arguments: dict, count: int):
    """Return a factory whose ONE ChatResponse carries *count* identical calls.

    The shape a runaway actually takes in the wild: not one repeated call per
    round, but a single assistant message holding thousands of them (measured:
    2,555 in one message). Every call is identical bar its id, so the per-call
    guards see the same key throughout while the round-level bound is what has
    to stop the batch.
    """
    from llm import ChatResponse, ToolCall

    def factory():
        return ChatResponse(
            text="",
            tool_calls=[
                ToolCall(id=_next_call_id("burst"), name=name, arguments=dict(arguments))
                for _ in range(count)
            ],
        )

    return factory


def _final_answer_response(text: str):
    """Return a factory that yields a no-tool-call ChatResponse (turn ends)."""
    from llm import ChatResponse

    def factory():
        return ChatResponse(text=text, tool_calls=[])

    return factory


def disable_memory_hooks() -> None:
    """Patch the agent's memory hooks to no-ops for offline evals.

    Eval scripts drive the real ``handle_user_message`` turn loop but must stay
    deterministic and offline. The orientation and consolidation hooks now run
    unconditionally (no ``--no-memory`` flag), so every eval that calls
    ``handle_user_message`` must call this once to prevent the hooks from
    spawning explorers, writing memory.db, or making LLM calls via the stub.
    """
    import agent

    agent.orientation_maybe_seed = lambda session: None  # noqa: E731
    agent.consolidation_maybe_extract = lambda session, client: None  # noqa: E731
