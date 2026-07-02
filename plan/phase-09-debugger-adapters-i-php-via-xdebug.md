## Phase 9 — Debugger adapters I: PHP via Xdebug

**Goal:** The same six debug tools drive PHP through vscode-php-debug + Xdebug, with the non-executable-language refusals and missing-adapter degradation validated.

**Scope (atomic chunks):**
- **Adapter acquisition pinned (counts in the day):** clone `github.com/xdebug/vscode-php-debug`, `npm install && npm run build`, run `node out/phpDebug.js` over stdio. Xdebug install checklist: pecl/brew install; php.ini — `xdebug.mode=debug`, `xdebug.start_with_request=yes`, `xdebug.client_port=9003`; requirements documented as config comments.
- Listen-mode flow in `dap/manager.py` via the P8 attach-style seam: adapter waits for Xdebug; harness launches the PHP script with `XDEBUG_MODE`/`XDEBUG_SESSION` env so it connects back.
- Register the **php launch-config synthesizer** into P8's registry (user-facing input stays "just a target file").
- Debug refusal for non-executable languages: `debug styles.scss` → clear "no debug adapter for this language" error (§2.1).
- Missing-adapter degradation validated live.

**Out of scope:** js-debug (P10). No new tools, no P8 code modifications — registrations + config only.

**Dependencies:** P8 (P5 fixture slice provides `php/bin/run.php`).

**Live test:**
1. `set a breakpoint in php/Cart.php line 20 and debug php/bin/run.php` → stopped; `inspect $items` → real values; `step over`, `continue`, `stop` → clean.
2. Python spot-check → debugpy still works untouched.
3. Remove the php entry from `config.debug_adapters`, relaunch, ask to debug PHP → single clear "adapter not configured" error; python debugging unaffected.
4. `debug styles.scss` → clear refusal.

**Acceptance criteria:**
- [ ] Full breakpoint→inspect→step→continue cycle on PHP via Xdebug listen-mode with zero changes to the six tools and zero edits to P8 modules (registration only).
- [ ] Launch-config synthesis: user passes just the target file.
- [ ] Refusal + degradation paths verified live.

**Definition of done:** standard DoD + Xdebug/php.ini setup quirks captured as config comments.
