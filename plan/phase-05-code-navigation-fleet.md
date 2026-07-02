## Phase 5 — Code navigation fleet

**Goal:** The agent navigates code like an IDE — definitions, implementations, type definitions, references, call hierarchy, outlines, workspace symbols, signature help — across Python, JS, and PHP.

**Scope (atomic chunks):**
- Env setup (counts in the day): `npm install -g typescript-language-server typescript intelephense` (and `vscode-langservers-extracted` now if step 7 is to exercise a live css server; otherwise P6 installs it).
- Seed the **P5 fixture slice** (§0.4): `js/order.js` (exports `parseOrder`; small class hierarchy), `js/index.js`, `php/Cart.php` (`Cart::total`), `php/bin/run.php`, `php/index.php` — additive files only.
- Shared position/URI translation + uniform location rendering helper — handles **both** `Location | LocationLink` and **both** `DocumentSymbol` (hierarchical) | `SymbolInformation` (flat) response variants; `workspace/symbol` returns SymbolInformation/WorkspaceSymbol.
- `tools/go_to_definition.py`, `tools/go_to_implementation.py`, `tools/go_to_type_definition.py`.
- `tools/find_references.py`; `tools/call_hierarchy.py` (`direction`: incoming callers / outgoing callees).
- `tools/document_symbols.py` (outline); `tools/find_symbol.py` (`workspace/symbol` fuzzy search, ctrl+T).
- `tools/signature_help.py` (parameter hints).
- First live validation of typescript-language-server and intelephense; lazy-start validation (moved from P4).
- Server capability gaps surface as clear per-tool errors. **Known capability facts (verified):** pyright declares no `implementationProvider`; intelephense gates implementations/type-definitions behind its premium licence — `go_to_implementation`/`go_to_type_definition` are therefore *validated on JS* (ts-ls supports both), and a clear capability error from pyright/intelephense **is the correct result** for those tools on Python/PHP.

**Out of scope:** anything that *edits* (P6).

**Dependencies:** P4.

**Live test:**
1. `where is fibonacci defined?` → file:line via `go_to_definition`.
2. `who calls fibonacci?` → `find_references`, then `show incoming calls` → `call_hierarchy` listing `tests/test_fib.py`.
3. `outline app.py` → symbol tree. `find the symbol Cart anywhere` → workspace symbol hit.
4. `what's the return type of fib at that call site?` → hover; `what parameters does that call take?` → `signature_help`.
5. Definition + references on `parseOrder` in `js/order.js` and `Cart::total` in `php/Cart.php` → ts-ls and intelephense answer end-to-end.
6. `go to the implementation of` the JS base-class method → ts-ls returns the override (implementation validated on JS); the same question on Python → clear capability error (correct per pyright's capabilities).
7. Lazy start: launch the harness in a Python-only subdirectory, then `create a file notes.js with a small function and outline it` → ts-ls lazy-spawn observed in the log mid-session.
8. Ask for call hierarchy on a CSS file → clear capability/server error, not a crash.

**Acceptance criteria:**
- [ ] All 8 tools return correct, consistently-rendered results, each validated on at least one language where the server supports the capability (implementation/type-definition on JS).
- [ ] Definition + references verified on Python, JS, and PHP (three LSP servers proven).
- [ ] Both dual response shapes (Location|LocationLink, DocumentSymbol|SymbolInformation) handled.
- [ ] Lazy start observed live; capability-gap and server-missing errors are clear and non-fatal.

**Definition of done:** standard DoD + P5 fixture slice committed.
