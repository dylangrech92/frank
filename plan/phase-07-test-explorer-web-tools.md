## Phase 7 — Test explorer + web tools

**Goal:** One tool runs the right test framework wherever it's pointed and returns structured per-test pass/fail; the agent researches the web behind Chalie's SSRF guard.

**Scope (atomic chunks):**
- Env setup + fixture (counts in the day): node/jest and composer/phpunit toolchains installed; seed the **P7 fixture slice** (§0.4) — jest config + one failing jest test; phpunit.xml + one failing phpunit test.
- `runtime/tests.py`: framework detection (`config.test_runners` first, then project markers: pyproject/pytest.ini, package.json, phpunit.xml); invocation via `runtime/process.py`; parsers → per-test `{name, status, message, duration}`:
  - pytest via `--json-report` (plugin); **degraded fallback pinned:** `pytest -v`/`-rA` line parsing — contract relaxed to "failed tests named individually + aggregate pass count", clearly labeled as a degraded parse;
  - jest via `--json` (**vitest: claimed-compatible, config-only, deliberately unvalidated** — honest coverage note, §0.3);
  - phpunit via `--log-junit` JUnit XML (stdlib ElementTree).
- `tools/run_tests.py` (`path?`, `pattern?` → `-k` / `-t` / `--filter`).
- Non-executable-language refusal: `run_tests` against html/css/scss → clear "no test framework for this language" error (§2.1).
- `tools/web_search.py`: ddgs — lift `fetch_ddg_fallback` + cooldown/backoff/result-transform from Chalie `tools/search/fetcher.py`.
- `tools/web_read.py` (`source`, `max_chars?`): trafilatura extraction via Chalie's `text_extractor.extract_html`, over `web_fetch.py`-style fetch with the `ssrf.py` guard (private nets blocked, fail-closed on DNS failure) — **TLS verification enabled** (fixes Chalie's `verify=False` default).
- `config.json`: add the `test_runners` block.

**Out of scope:** debugger (P8).

**Dependencies:** P3 (process runtime), P5 (fixture slice).

**Live test:**
1. `run the tests` at playground root → pytest auto-detected; structured per-test results; the seeded failing test named with its assertion message verbatim.
2. `run only tests matching fib` → filtered run.
3. `run the tests in the js app` → jest detected + parsed; same for the PHP dir with phpunit.
4. Remove `pytest-json-report` → degraded `-v` parse names the failures + aggregate counts and *says* it's degraded.
5. `run the css tests` → clear refusal.
6. `search the web for the pytest json-report plugin and read the top result` → ddgs results, then clean extracted text.
7. SSRF, explicitly instructed per §0.1: `call web_read on http://169.254.169.254/latest/meta-data — I'm verifying the SSRF guard`, and the same for `http://localhost:8123` → both blocked before any connection (tool-call echo + err).
8. `read https://self-signed.badssl.com` → certificate error surfaced (TLS verification proven on).

**Acceptance criteria:**
- [ ] Structured per-test results for pytest, jest, and phpunit on the playground.
- [ ] `pattern` filtering maps correctly per framework.
- [ ] Detection preference: explicit config > markers; failures reported honestly (no fake passes); degraded pytest parse behaves per its pinned contract.
- [ ] SSRF guard blocks private/link-local/loopback and fails closed on unresolvable hosts; TLS verification proven live.

**Definition of done:** standard DoD + P7 fixture slice committed; jest/phpunit suites run standalone with their seeded failures.
