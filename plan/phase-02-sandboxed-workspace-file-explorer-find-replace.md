## Phase 2 — Sandboxed workspace: file explorer + find/replace

**Goal:** The agent can read, build, and reshape a codebase — read/create/edit/move/delete/search files with precise or project-wide replaces — with every path hard-confined to the project root, and a mutation-event bus later phases subscribe to.

**Scope (atomic chunks):**
- **Shared fs-write helper** (seam #4): hardened `resolve_in_root()` (realpath + symlink-safe confinement, absolute-path and `..`-traversal rejection with clear `ToolResult.err`) **plus** mutation-event emission (`created/changed/deleted/renamed` to a subscriber list) in one module used by every mutating tool — P4 subscribes LSP sync + diagnostics; P6's applier and move-hook write through it untouched.
- `tools/read_file.py` (`path`, `start_line?`, `end_line?`): sandbox-confined read through `resolve_in_root()`; oversized files without a range → `ToolResult.err` with a hint to pass one. *(Approved catalog addition — §0.5.1; without it `update_file`'s full-overwrite semantics operate blind until `run_command cat` arrives in P3.)*
- `tools/create_file.py`, `tools/update_file.py` (full overwrite), `tools/delete_file.py`, `tools/create_folder.py`.
- `tools/move_file.py`: plain move + no-op pre-move hook seam (P6 wires `workspace/willRenameFiles`).
- `tools/find.py`: ripgrep-backed search; `fuzzy?` flag with documented semantics (exact = fixed-string; fuzzy = case-insensitive regex over term parts); if `rg` is absent → clear `ToolResult.err` naming the missing binary + install hint (mirrors the missing-engine convention).
- `tools/replace_one.py`: unique-match single-file replace; ambiguous or zero matches → `ToolResult.err` naming the count.
- `tools/replace_many.py`: project-wide replace, optional glob scope; returns per-file replacement counts.

**Out of scope:** LSP-aware rename/move (P5/P6).

**Dependencies:** P1.

**Live test:**
1. `create math_utils.py with an add() function` → file on disk, correct content.
2. `read lib/fib.py` → exact file contents in the reply; then `show me just its first 3 lines` → ranged read.
3. `change fibonacci in lib/fib.py to be iterative` → natural read-then-overwrite flow; `update_file` overwrite visible.
4. `find every occurrence of add` → ripgrep hits with file:line.
5. `in math_utils.py replace 'def add' with 'def sum_two'` → succeeds. Then ask a replace on a string that appears twice → exact ambiguity error surfaced in the reply.
6. `replace utils with helpers across all *.py files` → per-file counts reported.
7. `move math_utils.py into a lib/ folder` → moved (plain move; no import fixing yet).
8. Sandbox attacks on both read and write paths (per §0.1 guard phrasing): `read /etc/passwd`; `create a file at ../../outside.txt`; `list the files in /etc`; create a symlink inside the project pointing outside, then `read the file behind that symlink` and `update the file behind that symlink` → all rejected with sandbox `ToolResult.err`; REPL keeps running.
9. Stderr shows a mutation event logged for every mutating call — and none for `read_file`.

**Acceptance criteria:**
- [ ] All 9 tools drivable by conversation; results honest (contents, counts, errors); `read_file` honors line ranges and errs helpfully on oversized files.
- [ ] The sandbox attack shapes (relative traversal, absolute path, symlink escape — each proven on read **and** write paths) are rejected via the shared helper.
- [ ] `replace_one` refuses ambiguous and zero-match replaces with distinct messages.
- [ ] `find` respects `.gitignore`; missing-`rg` path returns the clear error.
- [ ] Every mutating tool emits exactly one mutation event through the shared helper (observed via log); `read_file` emits none.

**Definition of done:** standard DoD.
