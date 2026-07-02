## Phase 10 — Debugger adapters II: JS via js-debug

**Goal:** The six debug tools drive Node through js-debug's DAP-server mode (TCP, parent/child sessions), completing the Run & Debug engine matrix.

**Scope (atomic chunks):**
- **Adapter acquisition pinned (counts in the day):** download the `js-debug-dap` release tarball from the microsoft/vscode-js-debug releases page; run `node src/dapDebugServer.js <port>` — a **TCP** DAP server.
- Wire js-debug through P8's TCP transport + reverse-request seams: handle the `startDebugging` reverse request, open the child-session connection via the child-session registry, and bind breakpoints in the child session (where node code actually stops).
- Register the **js launch-config synthesizer** into P8's registry (node launch config for a plain `.js` target).
- Missing-adapter degradation for js validated live.

**Out of scope:** no new tools; no P8 modifications — registration + config only.

**Dependencies:** P8 (P5 fixture slice provides `js/index.js`).

**Live test:**
1. `set a breakpoint in js/order.js line 5 and debug js/index.js` → stopped in the child session; `inspect the local order` → real values; `step over`, `continue` to termination, `stop` → clean.
2. Python and PHP spot-checks → debugpy and Xdebug flows still work untouched.
3. Remove the js entry from `config.debug_adapters`, relaunch, ask to debug JS → single clear error; other languages unaffected.

**Acceptance criteria:**
- [ ] Full breakpoint→inspect→step→continue cycle on Node via js-debug DAP-server mode, including the startDebugging child-session flow, with zero changes to the six tools and zero edits to P8 modules.
- [ ] All three debug engines (debugpy, Xdebug, js-debug) proven side by side in one session.
- [ ] Degradation path verified live.

**Definition of done:** standard DoD + js-debug acquisition/launch quirks captured as config comments.
