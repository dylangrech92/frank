# Spec refinement — dependable-senior-coder pass (H-series)

Five refinements identified from the 2026-07-06 live-test sessions (P17 + slice G).
Design philosophy carried over from P17: reactive, evidence-gated mechanisms that
fire only when a smell appears — never standing per-turn instructions.

## H1 — Post-mutation verification nudge

**Gap:** the agent edits files, receives injected LSP diagnostics, and declares done
without running anything. A senior runs the tests.

**Mechanism:** in `agent.py`, when a turn is about to end (model returned no tool
calls) and `_TURN_MUTATIONS`-tracked file mutations occurred this turn without any
subsequent `run_tests`/`run_command` call, inject one system-side line before
accepting the final answer: the agent must either verify or explicitly state the
change is unverified. Fires at most once per turn.

**Done when:** a live session that edits a file and answers immediately receives the
nudge and either runs the tests or says "unverified"; a turn that already ran tests
gets no nudge.

## H2 — Stale-read edit guard

**Gap:** multiple agents share one directory (by design), but write tools happily
overwrite files modified on disk after this session last read them.

**Mechanism:** track per-session `{path: mtime/hash}` at `read_file` time; on
`replace_one`/`replace_many`/`update_file`/`delete_file`, if the file changed on disk
since the last read (or was never read), return `file-changed-on-disk` /
`not-read-yet` error with a re-read hint instead of writing.

**Done when:** an offline probe that reads, externally modifies, then edits gets the
error; re-reading clears it; `create_file` (new files) is unaffected.

## H3 — State-assumptions prompt rule

**Gap:** working rules cover evidence, retries, and scope — not ambiguity. Silent
interpretation picks are invisible until they are wrong.

**Mechanism:** one line in `SYSTEM_PROMPT`'s working rules: when the request is
ambiguous, state the assumption being proceeded on in the answer.

**Done when:** an ambiguous live prompt yields an explicit "assuming X" in the reply.

## H4 — Repeatable eval harness

**Gap:** dependability claims rest on hand-rolled one-off scripts; prompt/guard
changes have no regression score.

**Mechanism:** `evals/` directory with a stdlib-only runner that replays the proven
scenarios against the live endpoint (config-driven): load_tool gate, JSON leniency,
oversize guard (central via run_command + read_file paging), loop-guard steer
(dispatch-level), compaction coherence, scope decline, search-focus nudge. Each
scenario = prompt(s) + pass-markers grepped from transcript/stderr. Output: pass/fail
table.

**Done when:** `python3 evals/run.py` executes all scenarios against the configured
endpoint and prints a scored table matching today's known-good results.

## H5 — Graph-memory usage nudge

**Gap:** P14's decision/spec graph layer is injected into context but rarely written
to — nothing prompts the model to record decisions during long tasks, so
cross-session resumes rely on transcript summaries alone.

**Mechanism:** reactive trigger in `agent.py`: after a turn whose mutations span 3+
files and no `record_decision`/`record_spec` call happened this turn, append a hint
to the final tool result suggesting recording the driving decision. At most once per
session.

**Done when:** a multi-file-edit live session surfaces the hint exactly once and a
single-file edit never does.

## Sequencing

H2 + H4 first (touch tools/evals only). H1 + H3 + H5 after the slice-G commit lands
(they edit `agent.py`/`main.py`). Every slice: offline probe first, then live
verification on the standard endpoint, then commit.

# Spec refinement — speed & robustness pass (F-series)

Seven improvements identified 2026-07-07 after the P19 streaming/parallel-dispatch
work. Work on F1–F7 is being executed by parallel agents; check `git log` before
picking one up to avoid double-work.

## F1 — HTTP connection reuse in llm.py

**Gap:** every `LLMClient.chat()` opens a fresh connection via
`urllib.request.urlopen`, so each of the 2–10 round-trips per turn pays TCP (+TLS)
setup.

**Mechanism:** persistent keep-alive connection per client — `http.client`
connection reused across calls (or `requests.Session`; requests is already a
dependency via web_read). Must preserve the SSE streaming path
(`_read_sse_response` iterates the raw response) and the OverCapError /
RuntimeError error contract.

**Done when:** an offline probe shows connection reuse (single TCP connect across
sequential chats) and a live one-shot turn still streams and answers correctly.

## F2 — KV-cache-stable context prefix

**Gap:** provider-side prompt caches (llama.cpp/Ollama) reuse the prefill only up
to the first changed byte. Context providers (catalog, graph, flashback) inject
blocks each turn; any early-position block that varies per turn forces a full
re-prefill of everything after it — most of the TTFT on a 27b model.

**Mechanism:** audit `session.assemble_context()` ordering with evidence (dump
two consecutive turns' assembled contexts and diff). Pin static blocks (system
prompt, catalog) at the top; move per-turn-varying blocks (flashback, graph,
diagnostics) as late in the message list as their semantics allow.

**Done when:** consecutive-turn context dumps share a byte-identical prefix
covering the system + catalog blocks, and a live 2-turn session shows reduced
prompt-eval time on turn 2 (or the audit proves ordering was already optimal).

## F3 — Thread-safe LSP doc sync → parallel-safe navigation tools

**Gap:** the LSP request layer is thread-safe (locked framed writes, per-request
Events) but `LSPManager` doc sync (`_open_docs`, didOpen/didChange) is unguarded,
which is the only reason read-only navigation tools are excluded from P19
parallel dispatch.

**Mechanism:** guard doc-sync state with a lock (reuse/extend `_spawn_lock`
discipline), then set `parallel_safe = True` on the read-only navigation tools:
go_to_definition, go_to_implementation, go_to_type_definition, find_references,
find_symbol, document_symbols, hover, signature_help, call_hierarchy.

**Done when:** a stub batch of 2+ navigation calls dispatches concurrently with
correct results and transcript order, and a concurrent didOpen stress probe shows
no corruption/exception.

## F4 — parallel-safe recall via fresh-per-call store

**Gap:** `recall` rides `get_memory`'s main-thread cached SQLite connection, so it
cannot join parallel batches.

**Mechanism:** open a fresh store per call inside the tool (the established
fresh-store-per-thread pattern from `_memory_maintenance` / the episodic writer),
then set `parallel_safe = True`.

**Done when:** recall works from a worker thread in a stub parallel batch and
returns identical results to the main-thread path.

## F5 — bounded retry on transient LLM errors

**Gap:** `chat()` raises `RuntimeError` on any non-2xx; one 502/connection-reset
kills the whole turn (exit 1 in one-shot, which an orchestrator reads as task
failure).

**Mechanism:** one retry with short backoff on 5xx and connection-level errors
(URLError/ConnectionError/timeouts). Never retry 4xx; never retry after streaming
has begun delivering deltas (non-idempotent display).

**Done when:** a stub endpoint that 502s once then succeeds yields a successful
turn; a 400 still fails immediately; OverCapError path unchanged.

## F6 — capture real token usage

**Gap:** compaction runs off a chars/4 estimate; both plain and SSE responses
carry a `usage` field that is currently dropped.

**Mechanism:** parse `usage` (prompt_tokens/completion_tokens) from the response
(SSE: the final chunk that carries it) onto `ChatResponse`; surface it in the
per-turn telemetry line next to the estimate so drift is visible; optionally feed
it back to calibrate the estimator.

**Done when:** the telemetry line shows actual vs estimated tokens on a live turn
against a provider that reports usage, and estimates remain the fallback when
usage is absent.

## F7 — graceful Ctrl-C mid-turn in the REPL

**Gap:** an interrupt during a long turn propagates out of
`handle_user_message` and is swallowed by the generic error handler (or exits),
losing the session.

**Mechanism:** catch `KeyboardInterrupt` at the REPL loop boundary: abandon the
in-flight turn cleanly (transcript remains consistent — no dangling tool_calls
without results), print an "interrupted" notice, and return to the prompt.
Double Ctrl-C at the prompt still exits.

**Done when:** a scripted REPL session interrupted mid-turn returns to the prompt
with a valid transcript and can run a follow-up turn; Ctrl-C at the idle prompt
still exits.

## F-series sequencing & concurrency rules

F1/F5/F6 share `llm.py` — one owner works them together. F3+F4 share the
parallel-dispatch surface. F2 is `session.py`; F7 is `main.py` REPL. Multiple
agents share this directory: never run `git checkout --`/`restore`/`stash`/
`clean`/`reset`; stage and commit only your own hunks; if a needed file carries
someone else's uncommitted work, stage your hunk with `git apply --cached`.
