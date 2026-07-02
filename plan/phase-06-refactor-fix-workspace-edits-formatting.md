## Phase 6 — Refactor & fix: workspace edits + formatting

**Goal:** The lightbulb and F2: type-aware cross-file rename, listing/applying code actions, formatting — and `move_file` now fires `workspace/willRenameFiles` so imports auto-update where the server supports it. HTML/CSS/SCSS complete their intelligence+formatting-only story.

**Scope (atomic chunks):**
- `lsp/edits.py` — shared WorkspaceEdit applier: handles both `changes` and `documentChanges`, applies TextEdits in reverse document order, writes **through the P2 shared fs-write helper** so mutation events (didChange sync + diagnostics) fire. **Atomicity pinned:** resolve and validate every target path/edit in memory first, then write; hold pre-write contents in memory and restore them on any per-file failure.
- `tools/rename_symbol.py`: `textDocument/rename` (with `prepareRename` where required) + multi-file apply.
- `tools/code_actions.py`: list quick-fixes / organize-imports / fix-all for file+range; apply a chosen action; `codeAction/resolve` support. **Protocol quirk pinned:** echo the stored diagnostics for the range in `CodeActionContext.diagnostics`, and request `source.*` kinds via `context.only` — servers return empty lists otherwise. **Capability facts (verified):** pyright's only code action is "create type stub" (organize-imports exists solely as the `pyright.organizeimports` executeCommand — optional to wire); intelephense code actions are premium. **JS (ts-ls) is the validation target** (`source.organizeImports.ts`, quickfixes).
- `tools/format.py`: `textDocument/formatting` where the server supports it (ts-ls, intelephense, css/html servers); configurable CLI fallback per language (e.g. `black` — pyright does not format).
- Fill `move_file`'s P2 pre-move hook: fire `workspace/willRenameFiles`, apply returned edits via the applier, then move; degrade to plain move + visible warning where the server lacks support. **Verified:** ts-ls supports willRenameFiles; **pyright does not (and has stated no plans to)** — Python moves demonstrate the degrade path by design.
- Validate html/css/scss end-to-end: diagnostics + formatting only (§2.1); requires the P4 client answering `workspace/configuration` (already built). CSS diagnostics anchored on a seeded syntax error; **html diagnostics are best-effort** (the html server's validation is minimal).
- *(Slack valve: `codeAction/resolve` is the safest deferral if the day runs hot — ts-ls returns fully-populated edits for the actions exercised here.)*

**Out of scope:** debug/test refusals for non-executable languages (P7/P9).

**Dependencies:** P2 (hook seam + write helper), P5 (proven servers).

**Live test:**
1. `rename fibonacci to fib everywhere` → definition, importers, and tests change together; next diagnostics summary stays clean (proves applier + resync).
2. `what quick fixes are available in js/order.js?` → ts-ls list; `apply the organize-imports one` → file rewritten (JS is the code-actions validation target).
3. `format styles.scss` (css server), `format php/index.php` (intelephense), `format app.py` (black fallback) → all produce formatted files.
4. `move js/order.js into js/lib/` → importing files (`js/index.js`, the jest test dir once it exists) auto-update via ts-ls willRenameFiles; confirm via `git diff`; diagnostics stay clean.
5. `move lib/fib.py into utils/` → plain move + **visible degrade warning** (pyright lacks willRenameFiles — this step *is* the degrade-path demo).
6. Introduce a syntax error into `styles.scss` → real css-server diagnostics; `any problems in index.html?` → best-effort.
7. Rename `parseOrder` in the JS package → ts-ls cross-file rename works (second rename language).

**Acceptance criteria:**
- [ ] Multi-file WorkspaceEdit applies with the pinned in-memory-validate/restore-on-failure discipline; writes route through the shared helper (diagnostics stay in sync).
- [ ] rename verified on ≥2 languages (Python + JS); code_actions list+apply verified on JS; format verified on ≥3 languages incl. one CLI fallback.
- [ ] `move_file` updates importers on JS and produces the visible degrade warning on Python.
- [ ] CSS server diagnostics + formatting proven (config-request handling works).

**Definition of done:** standard DoD.
