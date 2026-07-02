## Phase 11 — Context economy: pruned assembly + reactive compaction

**Goal:** Sessions of any length survive: completed turns collapse to `[user, final answer]` in the assembled view, and over-cap sends compact-then-retry — while the on-disk transcript stays full-fidelity.

**Scope (atomic chunks):**
- `session.assemble_context()` (same P1 seam): in-flight turn keeps full native scaffolding; completed prior turns collapse to `[user message, final assistant answer]`; `tool_calls` fields stripped when tool results are dropped (no dangling calls); mid-chain narration dropped; injected diagnostics summaries stripped at turn boundaries via **P4's pinned representation** (they live inside tool-result content, which collapses with the turn); **context-provider output preserved** (providers render outside the pruning window — seam #3 honored, P14 depends on this).
- `compaction.py`: single shared cap definition `cap = window − max(0.10·window, 8000)`; pre-flight token estimate — tiktoken when importable, else the spec's chars/4 heuristic (new code — Chalie's actual fallback is words×1.3); the estimate **counts tool schemas** (lifted estimator does); trigger on projected overflow **or** on P1's `OverCapError`; compact then retry inside the P1 agent-loop seam; **bounded retry:** after N compaction attempts without the estimate dropping below cap, surface a clear error to the REPL instead of looping.
- Summarization prompt: Chalie's `ChatHistoryCompactionSystemPrompt` re-themed to fixed sections **Task / State / Files-touched / Open / Decisions / Last**.
- Persistence (simplified vs Chalie): replace everything above a watermark with one summary message, keep the recent tail — assembled-view only; disk transcript untouched.
- `config.json`: add the `compaction` block (`reserve_ratio` 0.10, `reserve_min_tokens` 8000).
- Context log line per LLM call: assembled message count + token estimate (pruning and compaction become visible numbers).

**Out of scope:** memory (episodic memory is *not* required for pruning — the full record lives on disk).

**Dependencies:** P1 (seams), P4 (summary representation contract). Placed here so its live test can use heavy multi-tool turns from P2–P10.

**Live test:**
1. Run a heavy turn (`find and fix the failing test` — many tool calls). Next turn with `--verbose`: the prior turn shows as exactly `[user, final answer]`, no `tool_calls` field, no tool messages, no embedded diagnostics summaries; token estimate drops sharply; disk transcript still holds every tool call.
2. Set `context_limit` low (~16000, tuning upward as needed — note: the ~35 tool schemas present by P11 count toward the estimate, so pick a limit where schemas + summary + tail comfortably fit under cap); drive ~15 real turns of edits/test runs → watch the send that would exceed cap trigger compaction: log shows the retry; assembled context now opens with one summary message carrying the six sections; recent tail intact.
3. `what were we doing at the very start, and which files have we touched?` → correct answer sourced from the summary.
4. Hide tiktoken → chars/4 path exercised; compaction still fires at a sane point.
5. No provider 400s about dangling tool_call ids across the whole session.

**Acceptance criteria:**
- [ ] Pruning matches §4.4 exactly (in-flight full; completed collapsed; strip rule enforced; provider output preserved).
- [ ] Compaction fires on both triggers (pre-flight estimate; provider `context_length_exceeded`), retries successfully, and respects the bounded-retry rule.
- [ ] Summary carries the six fixed sections; post-compaction answers about early work are correct.
- [ ] Disk transcript remains full-fidelity through both mechanisms.

**Definition of done:** standard DoD.
