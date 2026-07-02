## Phase 8 — Debugger core (DAP + debugpy)

**Goal:** A live, stateful debug session the agent drives across many REPL turns: breakpoints, launch, stop, step, inspect, continue, terminate — against real debugpy — with every seam P9/P10 need pre-built.

**Scope (atomic chunks):**
- `dap/client.py`: DAP seq/request/response/event protocol over the shared P4 framing module; event stream (`stopped`/`terminated`/`output`); capability handshake; **transport abstraction: stdio and TCP socket** (js-debug's DAP-server mode is TCP — seam #7); **reverse-request handling** (`startDebugging`, `runInTerminal`) with a **child-session registry** (multi-session support js-debug requires — seam #7).
- `dap/manager.py`: the only stateful subsystem — owns breakpoint registry (settable before or during a session), the adapter handle, and current stop state (thread, frame, location) **persisting between tool calls**; launch-style and adapter-mediated attach-style both supported; **per-language launch-config synthesizer registry** — python registered today; `debug_start` also accepts a raw launch-config dict (seam #7).
- **Correct DAP sequencing pinned (verified):** initialize request → initialize response → **launch request → wait for the adapter's `initialized` event** → setBreakpoints → configurationDone → launch response / first `stopped` event. debugpy defers `initialized` until after launch/attach — sending setBreakpoints before it yields hangs/ignored breakpoints.
- `tools/set_breakpoint.py` (`file`, `line`, `condition?`), `tools/clear_breakpoint.py`.
- `tools/debug_start.py` (`target|config`), `tools/debug_control.py` (`continue|step_over|step_into|step_out|pause`), `tools/debug_inspect.py` (evaluate expression / list scope variables / read call stack at current stop), `tools/debug_stop.py`.
- `stopped` events surfaced into tool results (location + top of stack) so the LLM sees where it is.
- Missing-adapter graceful error path; `config.json` gains the `debug_adapters` block (python validated; php/js entries added in P9/P10).

**Out of scope:** Xdebug (P9), js-debug (P10).

**Dependencies:** P4 (framing), P3 (process infrastructure).

**Live test (each step a separate REPL turn — proves state persistence; target is the seeded `bin/run_fib.py` driver, a plain runnable script):**
1. `set a breakpoint at lib/fib.py line 3`.
2. `debug bin/run_fib.py` → stopped at fib.py:3 with call stack in the reply.
3. `what is n right now?` → live evaluation.
4. `step over and show me the locals` → new line + variables.
5. `add a conditional breakpoint where n == 5 and continue` → stops with n = 5.
6. `continue` → runs to completion (terminated event surfaced).
7. `stop debugging` → clean teardown; a second `debug_start` works (no zombie adapter).
8. Ask to debug PHP (adapter not yet configured) → honest "adapter not configured" error, python debugging unaffected.

**Acceptance criteria:**
- [ ] Breakpoint/launch/stop/step/inspect/continue/terminate all work across separate turns; stop state survives between tool calls.
- [ ] Conditional breakpoints honored; breakpoints settable mid-session; the pinned initialize→launch→initialized→setBreakpoints→configurationDone sequence implemented.
- [ ] `stopped`/`terminated` events reflected in tool results.
- [ ] Session teardown is clean and restartable.
- [ ] Transport abstraction (stdio+TCP), reverse-request handling + child-session registry, and the synthesizer registry exist (exercised for real in P9/P10; TCP + reverse-request paths may be validated with a loopback smoke here).

**Definition of done:** standard DoD.
