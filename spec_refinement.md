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

# Spec refinement — static-analysis pass (I-series)

Goal: the harness provides analysis tooling well beyond what ad-hoc shell
commands offer, elevating what the model can deliver. Today the only analysis
surface is LSP diagnostics (`get_diagnostics`), `format`, and `code_actions` —
no linting, no dead-code detection, no deprecation scanning.

External analyzers are config-declared optional binaries (same posture as the
language servers and debug adapters): when one is missing the tool degrades to
a clear `lint-unavailable`-style error naming the missing binary, never a
crash. `requests` is the only accepted third-party *library*; analyzers are
external processes.

## I1 — `lint` tool + shared runner layer

**Gap:** style/bug-pattern linting only happens if the model thinks to run a
linter via `run_command`, and raw linter output is unbounded and unnormalized.

**Mechanism:** new `tools/_lint.py` shared layer + `tools/lint.py` tool.
Config gains a `linters` block mapping language → command template (defaults:
`ruff` for Python via `--output-format json`, `eslint` for JS/TS via
`--format json`, `phpstan` for PHP via `--error-format json`). The shared
layer exposes the pinned interface (I2 builds against it, do not change the
shape without updating I2):

    run_lint(paths, project_root) -> LintReport
    LintReport.issues: list[LintIssue]   # path, line, col, rule, severity, message, source
    LintReport.unavailable: dict[str, str]  # language -> reason (missing binary etc.)

`lint(path?)` tool: lints one file or the project, renders issues normalized
(`path:line:col rule severity message`), severity-sorted, capped at a count
that cannot blow the context budget, with a `+N more` tail line.

**Done when:** an offline probe on a file with known ruff findings returns the
normalized issues; a missing-binary language yields the unavailable note; a
clean file returns "no issues".

## I2 — reactive lint-delta injection

**Gap:** the model gets LSP diagnostics injected after edits, but no lint
feedback — and asking it to lint after every edit is a standing instruction
(the anti-pattern; steers must stay reactive and evidence-gated).

**Mechanism:** in `agent.py`, mirror the existing diagnostics-injection path:
before a write tool mutates a file, capture the file's lint issues; after the
write succeeds, lint again and append ONLY the issues new since the pre-edit
snapshot to the tool result (`[lint] path:line rule message`, capped).
No new issues → append nothing. Uses `tools/_lint.py:run_lint` (I1's pinned
interface). Skips silently when the language has no configured/available
linter. Delta-only is the point: pre-existing project lint noise must never
flood the context.

**Done when:** an offline probe that introduces a new ruff-detectable issue
via `replace_one` sees exactly the new issue appended to the tool result; an
edit that introduces nothing new appends nothing; projects with pre-existing
issues never see them injected.

## I3 — `find_dead_code` tool

**Gap:** proving a symbol dead today takes one `find_references` call per
symbol — nothing sweeps a file or project for unreachable/unused code.

**Mechanism:** new `tools/find_dead_code.py`. Primary adapters: `vulture`
(Python, JSON-ish parseable output) and — when present — `knip`/`ts-prune`
(TS/JS). Fallback for any LSP-supported language: walk `document_symbols` for
the target file and count `find_references` per symbol (excluding the
declaration itself); zero references → reported as potentially dead with a
"verify before deleting (dynamic dispatch, exports, reflection)" caveat in the
rendered output. Results normalized and capped like I1.

**Done when:** an offline probe on a fixture with a provably-unused function
reports it via vulture AND via the LSP fallback path; a fully-used fixture
reports none.

## I4 — deprecation surfacing

**Gap:** deprecated-API usage is invisible: LSP servers send
`DiagnosticTag.Deprecated` on symbols but the diagnostics store drops tags;
test runs swallow `DeprecationWarning`s.

**Mechanism:** two small cuts. (a) Preserve LSP diagnostic `tags` through the
diagnostics store and render `[deprecated]` markers in `get_diagnostics`
output and the post-edit injection line. (b) `run_tests` (pytest path): parse
the warnings summary for `DeprecationWarning`/`PendingDeprecationWarning`
entries and append a compact `deprecations:` section to the rendered result
(no `-W error` — never turn warnings into failures behind the user's back).
Additionally ruff's deprecation-adjacent rules ride in free via I1 defaults.

**Done when:** a fixture calling a deprecated API shows the `[deprecated]`
marker in `get_diagnostics` (verify pyright/typescript-language-server
actually emit the tag; if a server never does, document that and rely on the
other cuts); a pytest fixture raising a DeprecationWarning shows the
`deprecations:` section in `run_tests` output.

## I-series sequencing & ownership

I1 owns `tools/_lint.py`, `tools/lint.py`, the `linters` config block.
I2 owns the `agent.py` wiring ONLY (imports I1's pinned interface) and starts
after the H-series `agent.py` commit lands. I3 owns `tools/find_dead_code.py`.
I4 owns `tools/get_diagnostics.py`, `diagnostics.py` store, `tools/run_tests.py`.
No two slices share a file. Registry/config additions follow the existing
patterns (`load_tool` deferred schemas; config parsing in `config.py`).
Every slice: offline probe first, then live verification, then commit.

# Spec refinement — CC-parity pass (S-series)

Three structural upgrades identified 2026-07-07: close the gaps that make the
harness weaker than mainstream agent CLIs when driving a strong model, while
keeping its native advantages (DAP debugging, LSP refactors, injected
diagnostics). A fourth candidate (capability-tiered guardrail profile) was
explicitly deferred by the owner.

## S1 — subagent fan-out tool

**Gap:** the harness has exactly one context window. Every file read to answer
a broad question burns the parent's context, while P18 already made concurrent
one-shot instances safe (pid-unique sessions, WAL memory db, advisory locks)
— the fan-out capability exists but no tool exposes it.

**Mechanism:** new `spawn_agents` tool: accepts a list of `{prompt, cwd?}`
specs (cwd defaults to the project root), launches each as a child one-shot
(`python3 <install_dir>/main.py -p -`, prompt on stdin) via
subprocess, all children concurrent up to `subagents.max_concurrent` (config,
default 4), bounded per-child timeout (`subagents.timeout_s`, default 600).
Returns per-child: answer (stdout), exit code, stderr tail on failure.
Recursion guard: children get `CODING_AGENT_DEPTH=parent+1`; the tool refuses
at depth >= 2. Not `parallel_safe` (children may mutate files).

**Done when:** a live parent one-shot fans out 2 children answering different
questions about this repo, both answers come back correct, and wall-clock is
clearly under the sum of two sequential child runs; a depth-2 spawn attempt is
refused with a clear error.

## S2 — usage-driven context accounting

**Gap:** compaction triggers off a chars/4 estimate; F6 now captures real
`usage` (prompt/completion tokens) but only prints it in telemetry — the
number that matters is dropped where it matters most.

**Mechanism:** carry the latest real `prompt_tokens` on the session; use it as
the primary compaction-trigger signal when present (the estimate stays as the
fallback for providers that omit usage). Calibrate the estimator with an EMA
of observed real-vs-estimated ratio so the fallback drifts toward truth.
Depends on F6 being landed.

**Done when:** a live multi-turn session logs compaction decisions with real
token counts; a forced near-cap probe compacts at the real threshold; a stub
response without usage falls back to the (now calibrated) estimate.

## S3 — hard verify gate (upgrade of H1's nudge)

**Gap:** a nudge can be ignored; the harness still lets a turn that mutated
files end with an unverified "done". A senior never ships unverified work
silently.

**Mechanism:** harness-enforced in `agent.py`: when a final answer arrives on
a turn whose mutation tracker recorded file changes and no `run_tests`/
`run_command` ran this turn, bounce once — inject a system-side message
requiring the model to either verify now or state the work is unverified. If
the second final answer still has neither, accept it but prefix the answer
with a harness-side `[UNVERIFIED CHANGES]` marker so the caller (human or
orchestrator) sees the state. One bounce max per turn — no loops. If H1's
nudge already landed from the other session, upgrade it in place rather than
adding a parallel mechanism (one source of truth).

**Done when:** a live session that edits a file and immediately answers gets
bounced and then runs the tests (or declares unverified); a turn that already
ran tests is untouched; a stubborn double-refusal yields the marker; the
bounce fires at most once per turn.

## S4 — structured result envelope for one-shot mode

**Gap:** the executor returns prose on stdout. An orchestrator driving many
instances must re-read each executor's work to know what happened — which
files changed, whether anything verified the change, what it cost. The
`[UNVERIFIED CHANGES]` marker was the first byte of this contract; the rest
is missing.

**Mechanism:** new `--json` flag, valid only with `-p` (argparse error
otherwise). With it, stdout carries exactly ONE JSON object and nothing else;
stderr behavior (streaming deltas, telemetry) is unchanged; exit codes are
unchanged. Envelope shape (version-pinned):

    {
      "envelope": 1,
      "status": "ok" | "error",
      "error": null | "<message>",
      "answer": "<final answer text, WITHOUT the [UNVERIFIED CHANGES] prefix>",
      "verified": true | false,
      "declared_unverified": true | false,
      "files_changed": [{"path": "...", "tool": "replace_one"}, ...],
      "verification_runs": [{"tool": "run_tests", "status": "success",
                             "detail": "<command or short descriptor>"}, ...],
      "usage": {"prompt_tokens": N|null, "completion_tokens": N|null,
                "llm_calls": N},
      "session_id": "...",
      "duration_s": 12.3
    }

Collection happens in `agent.py`: a per-turn report accumulated alongside the
existing trackers — mutation records captured at the `_TURN_MUTATIONS` append
site (NOT by reading the list later; the diagnostics/lint injection drains
it), verification tool calls (`run_tests`/`run_command`) with their result
status, usage summed per LLM call, and the S3 gate outcome. `files_changed`
is deduped by path, first-tool-wins. In `--json` mode the S3 marker is not
prefixed onto the answer; the `verified`/`declared_unverified` fields carry
that state instead (prose mode keeps the marker exactly as today). On a turn
error the envelope is still emitted (status "error", answer null/partial,
exit 1). The report surface is exposed to `main.py` via the session object.

**Done when:** an offline probe covers: mutation + verification run →
`verified: true` with the run listed; mutation + none → `verified: false`;
model self-declares → `declared_unverified: true`; error path emits a valid
envelope with status "error" and exit 1; plain mode (no `--json`) behaves
byte-identically to today. The existing regression suite stays ALL PASS. A
live `--json` one-shot that edits a file parses with `json.loads`, shows the
edit in `files_changed`, the verification run, real usage numbers, and a
clean answer.

## S-series sequencing & concurrency rules

S1 (new tool file + registry + config) and S3 (`agent.py` final-answer path)
are disjoint — parallel owners. S3 coordinates with H1/I2 (same `agent.py`
surface): whoever lands second integrates, never duplicates. S2 waits for
F1/F5/F6 to land and the regression suite to pass, since it builds directly on
F6's usage capture. Same shared-directory rules as the F-series: never
`git checkout --`/`restore`/`stash`/`clean`/`reset`; stage and commit only
your own hunks (`git apply --cached` for shared files); leave other agents'
uncommitted hunks alone.
