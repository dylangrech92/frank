## Phase 4 — LSP spine & live diagnostics

**Goal:** Real language servers run under the harness and the agent permanently sees editor truth: `publishDiagnostics` flows into a store, a one-line summary is auto-injected after mutating steps, full detail is on demand, and missing engines degrade gracefully.

**Scope (atomic chunks):**
- Shared JSON-RPC framing module (seam #6): Content-Length framing + threaded reader, written once, explicitly reused by DAP in P8.
- `lsp/client.py`: JSON-RPC-over-stdio client — `initialize`/`initialized`/`shutdown` lifecycle, request/notify, server-notification callbacks; responds to server→client requests incl. `workspace/configuration` (the css/html servers won't publish diagnostics without it — P6 relies on this); raw-traffic debug flag.
- `lsp/manager.py`: language→server map from `config.language_servers` (all six §2.1 languages configured; css+scss share the css server); lazy start; launch-time pre-warm for languages detected in the tree; subscribed to **all four** P2 bus events — created/changed → `didOpen`/`didChange`; deleted → `didClose` + purge that URI from the diagnostics store; renamed → `didClose`(old) + `didOpen`(new) + purge old URI.
- `diagnostics.py`: `{uri: [diagnostics]}` store; one-line summary (`⚠ 3 errors, 2 warnings in 2 files`) — **representation pinned: appended to the content of the final tool-result message of the mutating step, never a standalone message** (P11's pruning strips against this exact contract); settle mechanism pinned: after the `didChange` for a mutated file, wait for the next `publishDiagnostics` for that URI up to a 2 s deadline, then summarise whatever the store holds.
- `tools/get_diagnostics.py`: full dump, optional file filter (§5, §6).
- `tools/hover.py`: type/signature/docs — proves the `(file, line, col)` → `textDocument/*` position translation layer (0-based lines, UTF-16 columns). *(Release valve: hover may slip to P5, which owns the position/rendering fleet — purely additive; step 6 below can exercise `get_diagnostics` instead.)*
- Startup engine report + graceful degradation: each missing/unstartable server reported once at launch; that language's code-intel tools return a clear `server not installed` error; file/text tools unaffected (§2.1).
- `config.json`: add the full `language_servers` block.

**Out of scope:** navigation fleet (P5), workspace edits (P6). Only pyright is *validated* today; ts-ls/intelephense are configured but proven (and lazy-start is validated) in P5.

**Dependencies:** P1, P2 (mutation bus).

**Live test:**
1. Launch in the scratch playground → startup line lists detected languages and server status (pyright started; any missing servers named once).
2. `what's the signature of create_app in app.py?` → hover returns real pyright type info.
3. `add a call to a function that doesn't exist in app.py` → after the edit, the mutating step's final tool-result (visible via `--verbose`) carries the injected `⚠ 1 error in 1 file` summary.
4. `show me the full diagnostics` → pyright's exact message with file/line/col.
5. `fix it` → after the fix, summary goes clean (absent).
6. `delete that file` → diagnostics for its URI purged; summary no longer mentions it (deleted-event handling).
7. Point config at a bogus pyright path, relaunch → one startup warning; a Python hover question returns the clean `server not installed` error; `create_file` still works.

**Acceptance criteria:**
- [ ] pyright pre-warms at launch when Python files are detected.
- [ ] All four mutation-bus events drive the correct LSP lifecycle calls; no stale-document answers; deleted/renamed files leave no stale diagnostics.
- [ ] Diagnostics summary auto-injects **only** after steps that mutated files, in the pinned representation, reflecting the mutation (settle mechanism works).
- [ ] `get_diagnostics` returns the store's full detail, filtered by file when asked.
- [ ] Missing-engine path: one startup report, per-tool clear error, no crash, unrelated tools unaffected.

**Definition of done:** standard DoD + raw LSP traffic flag documented for debugging later phases.
