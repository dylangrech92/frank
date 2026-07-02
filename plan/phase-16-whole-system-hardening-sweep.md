## Phase 16 — Whole-system hardening sweep

**Goal:** One dedicated day proving the finished harness end-to-end and fixing whatever the sweep surfaces — the final integration gate.

**Scope (atomic chunks):**
- Scripted-by-hand (not automated) full-pass live session plan covering every subsystem: sandbox attacks, terminal deny-list, destructive-git gate, SSRF + TLS, missing-engine degradation (LSP server, debug adapter, test plugin, ONNX model each removed once), diagnostics injection, nav/refactor across the three LSP servers, all three debug engines, all three test frameworks, pruning + a forced compaction, and the full memory loop (remember/recall/forget, episodic reconsolidation, rules injection, pivot supersession, flashback gates, boundary jobs).
- Fix every regression the sweep surfaces (budgeted: this is the day's real work).
- Polish pass on operator-facing texts surfaced by the sweep (startup engine report, degradation errors, gate/extractor logs).
- Delivery summary written: what was delivered, challenges overcome, key decisions and why.

**Out of scope:** new features; anything in §15.

**Dependencies:** P1–P15 (all).

**Live test:** the sweep itself — one long polyglot session (fresh scratch playground) driving the representative live-test happy path of every prior phase, in order, plus the degradation matrix (each engine removed once). The session's `--verbose` stream, transcript file, memory.db, and process table are the evidence trail.

**Acceptance criteria:**
- [ ] Every prior phase's representative live-test command passes in a single continuous session.
- [ ] Degradation matrix: each missing engine produces its one startup report + clean per-tool error, with all other subsystems unaffected.
- [ ] Zero regressions left open; every fix re-validated live within the day.

**Definition of done:** standard DoD + delivery summary written.
