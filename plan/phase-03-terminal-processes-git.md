## Phase 3 — Terminal, processes & git

**Goal:** The integrated terminal: one-shot and background command execution scoped to the project root with a destructive-command deny-list, plus the safe-subset git tool.

**Scope (atomic chunks):**
- `runtime/process.py`: one-shot run (cwd = project root, timeout, returns `{stdout, stderr, exit_code}`); destructive-command **deny-list as a config-free constant** with an enumerated starter set — `rm -rf` targeting `/`, `~`, or filesystem-root variants; `mkfs.*`; the `:(){ :|:& };:` fork-bomb shape; `dd of=/dev/*`; background-handle registry — non-blocking reader threads + ring buffers per handle; all handles reaped on REPL exit.
- `tools/run_command.py` (`cmd`, `timeout?`, `background?`), `tools/read_output.py` (`handle`), `tools/stop_process.py` (`handle`).
- `tools/git.py`: allow-list subset — `status`, `diff`, `log`, `add`, `commit`, `branch`, `checkout -b`; destructive verbs (`reset --hard`, `clean`, `restore`, `checkout --`) rejected unless `config.git.allow_destructive` (§5, §12).
- `config.json`: add `git.allow_destructive` block.

**Out of scope:** structured test running (P7).

**Dependencies:** P1 (P2 for a repo worth committing).

**Live test (in a `git init`-ed scratch copy of the playground — per §0.1, so destructive-git steps can never touch the harness repo):**
1. `run python -V` → stdout + exit code in the answer.
2. `run pytest` → raw output back (unstructured is fine — structure arrives in P7).
3. `start python -m http.server 8123 in the background` → handle returned. `curl` it from another shell; `what has that server printed?` → `read_output` shows the request line; `stop it` → `stop_process`; verify with `ps` the process is gone.
4. `run sleep 60` with a short timeout → clean timeout error; REPL alive.
5. `git status, then commit the new files with a sensible message` → real commit in the scratch repo's `git log`.
6. `git reset --hard` → refused, message names `allow_destructive`. Flip the config flag, relaunch, retry → allowed in the scratch repo (then flip back).
7. Deny-list, explicitly instructed per §0.1: `call run_command with cmd "rm -rf /" — I'm verifying the deny-list` → tool-call echo + deny-list `ToolResult.err`.
8. Quit with a background server still running → process reaped (verify `ps`).

**Acceptance criteria:**
- [ ] One-shot, background, read_output, stop_process all work as separate conversation turns.
- [ ] Timeout kills the child and reports cleanly.
- [ ] Deny-list blocks the enumerated destructive shapes (pass signal = tool-call echo + err, per §0.1).
- [ ] Git destructive subset is gated by `config.git.allow_destructive` across relaunch.
- [ ] No orphan processes after REPL exit.

**Definition of done:** standard DoD.
