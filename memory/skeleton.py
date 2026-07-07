"""memory/skeleton.py — S-DERIVED project skeleton (D3 / M3 of the memory redesign).

Deterministic, zero-LLM "where things live" map: a gitignore-aware file listing
personalised to a task, plus the top-ranked symbols per most task-relevant file.
Rebuilt fresh from source on every call (mtime-cached so re-rendering within a
session is cheap) rather than persisted as memory — structural facts go stale
silently, so they are *derived*, never remembered (MEMORY_REDESIGN.md SS2/SS5/SS6).

Ranking is deliberately v1-simple: a task-personalised token-overlap sort over
file paths and symbol names — NOT PageRank (MEMORY_REDESIGN.md SS12 decision D).

Never calls an LLM or embedder. Degrades loudly (stderr) instead of silently
when the LSP layer is unavailable or a warm server hasn't been spawned for a
language, falling back to a local AST (Python) or regex (other languages)
symbol extractor — the skeleton itself is never silently empty.
"""

from __future__ import annotations

import ast
import re
import sys
import threading
from pathlib import Path
from typing import Any

from tools.list_files import _load_gitignore_patterns, _should_skip

try:
    from lsp.manager import EXTENSION_LANGUAGES, path_to_uri
    from lsp.locations import flatten_symbols
except Exception:  # pragma: no cover - the lsp package should always import cleanly
    EXTENSION_LANGUAGES: dict[str, str] = {}
    path_to_uri = None
    flatten_symbols = None


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

CANDIDATE_WINDOW = 40        # top files (by cheap path score) considered for symbol extraction
FILES_WITH_SYMBOLS = 15      # how many ranked files get a symbol breakdown in the output
SYMBOLS_PER_FILE = 8         # symbols shown per file, matched-first
LSP_TIMEOUT = 1.5            # seconds; only ever queries an ALREADY-RUNNING server (spawn=False)
CHARS_PER_TOKEN_PROXY = 3.2  # conservative chars/token used only for the greedy fit loop

_KIND_PRIORITY = {
    'class': 0, 'struct': 0, 'enum': 0, 'interface': 0,
    'function': 1, 'constructor': 1,
    'method': 2,
}

_STOPWORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'but', 'by',
    'can', 'do', 'does', 'for', 'from', 'had', 'has', 'have', 'if', 'in',
    'into', 'is', 'it', 'its', 'of', 'on', 'onto', 'or', 'our', 'over',
    'please', 'should', 'so', 'than', 'that', 'the', 'their', 'them', 'then',
    'there', 'these', 'this', 'those', 'to', 'up', 'was', 'we', 'were',
    'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'would',
    'you', 'your',
})


# ---------------------------------------------------------------------------
# Tokenisation (shared by task text, file paths, and symbol names)
# ---------------------------------------------------------------------------

_CAMEL_BOUNDARY_1 = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_CAMEL_BOUNDARY_2 = re.compile(r'(?<=[A-Z])(?=[A-Z][a-z])')
_NON_ALNUM = re.compile(r'[^a-zA-Z0-9]+')


def _split_identifier(text: str) -> list[str]:
    """Split *text* into lowercase fragments across camelCase/snake_case/kebab-case/paths."""
    step = _CAMEL_BOUNDARY_1.sub(' ', text)
    step = _CAMEL_BOUNDARY_2.sub(' ', step)
    step = _NON_ALNUM.sub(' ', step)
    return [t.lower() for t in step.split() if t]


def _tokenize(text: str, *, min_len: int = 2) -> set[str]:
    """Tokenise *text* for relevance scoring, dropping stopwords and 1-char fragments."""
    return {t for t in _split_identifier(text) if len(t) >= min_len and t not in _STOPWORDS}


def _estimate_tokens(text: str) -> int:
    """Approximate token count: tiktoken (cl100k_base) when importable, else chars/4.

    Mirrors the estimator in ``compaction.py`` without importing it (that module
    pulls in ``session``/``llm`` — heavier than this module should depend on).
    """
    if not text:
        return 0
    try:
        import tiktoken  # local import: optional dependency, keep module import cheap

        enc = tiktoken.get_encoding('cl100k_base')
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# mtime-cached project walk (gitignore-aware; reuses tools.list_files' filter)
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_TREE_CACHE: dict[str, tuple[tuple, list[str]]] = {}          # root -> (signature, files)
_SYMBOL_CACHE: dict[str, tuple[float, list[dict], str]] = {}  # abs path -> (mtime, symbols, source)


def _tree_signature(root: Path, patterns: list[str]) -> tuple:
    """A cheap (non-recursive) change signature for *root*.

    Combines the root directory's own mtime with the ``(name, mtime)`` of each
    direct, non-ignored child. Catches files/dirs added, removed, or renamed at
    the root or immediately under it; it will not notice a content edit nested
    several levels deep with no directory-entry change at those two levels —
    an accepted simplification per D3 ("keep it simple and correct"): a stale
    hit just means the next call re-walks, which is itself only milliseconds.
    """
    try:
        root_mtime = root.stat().st_mtime
    except OSError:
        return (0.0, ())

    children: list[tuple[str, float]] = []
    try:
        for entry in sorted(root.iterdir(), key=lambda e: e.name):
            if _should_skip(entry.name, entry.is_dir(), patterns):
                continue
            try:
                children.append((entry.name, entry.stat().st_mtime))
            except OSError:
                continue
    except OSError:
        pass

    return (root_mtime, tuple(children))


def _walk_project(root: Path, patterns: list[str]) -> list[str]:
    """Gitignore-aware recursive file walk, reusing ``tools.list_files``' filter rules.

    Returns:
        Relative (posix-style) file paths only — directories are not listed —
        sorted directories-first then by name within each directory level.
    """
    files: list[str] = []

    def _recurse(dir_path: Path) -> None:
        try:
            entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        except OSError:
            return
        for entry in entries:
            if _should_skip(entry.name, entry.is_dir(), patterns):
                continue
            if entry.is_dir():
                _recurse(entry)
            else:
                files.append(str(entry.relative_to(root)))

    _recurse(root)
    return files


def _cached_walk(root: Path) -> list[str]:
    """Return the gitignore-aware file list for *root*, mtime-cached.

    A cache hit costs one ``stat`` on the root plus one per immediate child; a
    miss re-walks the whole tree (still fast — see module docstring).
    """
    key = str(root)
    patterns = _load_gitignore_patterns(root)
    signature = _tree_signature(root, patterns)

    with _CACHE_LOCK:
        cached = _TREE_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]

    files = _walk_project(root, patterns)
    with _CACHE_LOCK:
        _TREE_CACHE[key] = (signature, files)
    return files


# ---------------------------------------------------------------------------
# Symbol extraction: warm-LSP first, loud fallback to static analysis
# ---------------------------------------------------------------------------


def _check_lsp_manager() -> Any | None:
    """Return the running agent process's ``LSPManager``, or ``None`` when absent.

    Mirrors the ``import main as main_module`` seam already used by
    ``tools/find_symbol.py`` and ``tools/document_symbols.py`` — when this
    module runs inside the live agent process, ``main.MANAGER`` is the shared,
    already-prewarmed manager; in a standalone script (like this slice's
    live-test) or before startup, it is ``None`` and callers fall back.
    """
    try:
        import main as main_module  # pylint: disable=import-outside-toplevel
    except Exception:
        return None
    return getattr(main_module, 'MANAGER', None)


def _lsp_symbols(abs_path: Path, language: str, manager: Any) -> list[dict] | None:
    """Query an ALREADY-RUNNING language server for *abs_path*'s outline.

    Never spawns a server (``spawn=False``) — spawning costs seconds and would
    defeat the "cheap on every task" requirement. Returns ``None`` (not an
    empty list) whenever no warm server is available or the request fails, so
    callers can tell "no symbols" apart from "couldn't ask" and fall back.
    """
    if flatten_symbols is None or path_to_uri is None:
        return None

    try:
        client = manager.get_client(language, spawn=False)
    except Exception:
        return None  # no warm server for this language yet — normal, not logged per-file

    try:
        manager.ensure_document_open(str(abs_path))
        result = client.request(
            'textDocument/documentSymbol',
            {'textDocument': {'uri': path_to_uri(str(abs_path))}},
            timeout=LSP_TIMEOUT,
        )
    except Exception as exc:
        print(
            f"memory.skeleton: LSP documentSymbol failed for {abs_path} ({exc!r}); "
            f"falling back to static extraction",
            file=sys.stderr,
        )
        return None

    return [
        {'name': s['name'], 'kind': s['kind'], 'line': s['line'], 'depth': s['depth']}
        for s in flatten_symbols(result)
    ]


def _python_symbols(source: str, rel_path: str) -> list[dict]:
    """Top-level classes/functions plus one level of class methods, via ``ast``.

    Deliberately shallow (no nested-function descent) — this is an outline for
    orientation, not a full call graph.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        print(f"memory.skeleton: could not parse {rel_path} ({exc}); skipping its symbols", file=sys.stderr)
        return []

    out: list[dict] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.append({'name': node.name, 'kind': 'class', 'line': node.lineno - 1, 'depth': 0})
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append({'name': member.name, 'kind': 'method', 'line': member.lineno - 1, 'depth': 1})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append({'name': node.name, 'kind': 'function', 'line': node.lineno - 1, 'depth': 0})

    return out


# Conservative class/top-level-function declaration patterns for non-Python
# languages when no warm language server is available. Narrow on purpose (no
# generic brace-heuristic method detection) to avoid false positives — a known
# v1 gap versus the LSP path, which returns a fully accurate outline.
_GENERIC_DEF_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)'), 'class'),
    (re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s*\*?\s+([A-Za-z_$][\w$]*)'), 'function'),
    (re.compile(r'^\s*(?:public\s+|private\s+|protected\s+|static\s+)+function\s+([A-Za-z_]\w*)'), 'method'),
    (re.compile(r'^\s*function\s+([A-Za-z_]\w*)'), 'function'),
]


def _regex_symbols(source: str) -> list[dict]:
    """Line-scan fallback symbol extractor for non-Python files (see module note)."""
    out: list[dict] = []
    for i, line in enumerate(source.splitlines()):
        for pattern, kind in _GENERIC_DEF_PATTERNS:
            m = pattern.match(line)
            if m:
                out.append({'name': m.group(1), 'kind': kind, 'line': i, 'depth': 0})
                break
    return out


def _extract_symbols(root: Path, rel_path: str, manager: Any | None) -> tuple[list[dict], str]:
    """Return ``(symbols, source)`` for *rel_path*, mtime-cached per file.

    *source* is one of ``'lsp' | 'ast' | 'regex' | 'none'``. LSP is tried first
    (only against an already-warm server); any miss or failure falls back to a
    local static extractor rather than leaving the file symbol-less silently.
    """
    abs_path = root / rel_path
    try:
        mtime = abs_path.stat().st_mtime
    except OSError:
        return [], 'none'

    cache_key = str(abs_path)
    with _CACHE_LOCK:
        cached = _SYMBOL_CACHE.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1], cached[2]

    suffix = abs_path.suffix.lower()
    language = EXTENSION_LANGUAGES.get(suffix)

    symbols: list[dict] = []
    source = 'none'

    if language is not None and manager is not None:
        lsp_syms = _lsp_symbols(abs_path, language, manager)
        if lsp_syms:
            symbols, source = lsp_syms, 'lsp'

    if not symbols:
        try:
            text = abs_path.read_text(encoding='utf-8', errors='replace')
        except OSError as exc:
            print(f"memory.skeleton: could not read {rel_path} ({exc!r}); skipping its symbols", file=sys.stderr)
            text = None

        if text is not None:
            if suffix == '.py':
                symbols, source = _python_symbols(text, rel_path), 'ast'
            else:
                symbols, source = _regex_symbols(text), 'regex'

    with _CACHE_LOCK:
        _SYMBOL_CACHE[cache_key] = (mtime, symbols, source)

    return symbols, source


# ---------------------------------------------------------------------------
# Ranking: v1 task-personalised token overlap (NOT PageRank — see module docstring)
# ---------------------------------------------------------------------------


_TEST_DIR_NAMES = frozenset({'test', 'tests', '__tests__', 'spec', 'specs'})
_TEST_PATH_PENALTY = 0.5  # de-prioritise (not exclude) test files vs. same-scoring implementation


def _is_test_path(rel_path: str) -> bool:
    """Heuristic: does *rel_path* look like a test file (dir name or filename convention)?

    Test files often repeat task keywords more densely than the implementation
    they exercise (descriptive test names), which would otherwise out-rank the
    file a "senior who knows this codebase" would actually reach for first.
    This only de-weights tests (see ``_TEST_PATH_PENALTY``); it never excludes
    them — a task that is itself about tests still surfaces them.
    """
    p = Path(rel_path)
    if any(part.lower() in _TEST_DIR_NAMES for part in p.parts[:-1]):
        return True
    stem = p.stem.lower()
    return (
        stem.startswith('test_')
        or stem.endswith(('_test', '.test', '_spec', '.spec'))
    )


def _path_score(rel_path: str, task_tokens: set[str]) -> float:
    """Score *rel_path* by task-token overlap: filename weighs 3x, directory 1x."""
    if not task_tokens:
        return 0.0
    p = Path(rel_path)
    filename_tokens = _tokenize(p.stem)
    dir_tokens = _tokenize('/'.join(p.parts[:-1])) if len(p.parts) > 1 else set()
    return 3.0 * len(task_tokens & filename_tokens) + 1.0 * len(task_tokens & dir_tokens)


def _rank_symbols(symbols: list[dict], task_tokens: set[str]) -> list[dict]:
    """Order *symbols*: task-token matches first, then class/function before method."""
    def keyfn(sym: dict) -> tuple:
        unmatched = 0 if (task_tokens & _tokenize(sym.get('name', ''))) else 1
        return (unmatched, _KIND_PRIORITY.get(sym.get('kind', ''), 3), sym.get('line', 0))

    return sorted(symbols, key=keyfn)


def _matched_symbol_count(symbols: list[dict], task_tokens: set[str]) -> int:
    return sum(1 for s in symbols if task_tokens & _tokenize(s.get('name', '')))


# ---------------------------------------------------------------------------
# Rendering + budget fitting
# ---------------------------------------------------------------------------


def _render_file_block(entry: dict[str, Any], max_symbols: int) -> str:
    lines = [entry['path']]
    for sym in entry['symbols'][:max_symbols]:
        indent = '  ' * (1 + min(sym.get('depth', 0), 2))
        lines.append(f"{indent}{sym['kind']} {sym['name']} — line {sym['line'] + 1}")
    return '\n'.join(lines)


def _fit_blocks(items: list[dict[str, Any]], char_budget: int) -> tuple[list[str], bool]:
    """Greedily keep *items* (already in priority order) within *char_budget*.

    Each item is either ``{'kind': 'text', 'text': str}`` (atomic — a section
    header or a single project-shape line) or ``{'kind': 'file', 'entry': ...}``
    (a ranked file whose symbol list is progressively trimmed — most symbols,
    then fewer, then bare path — before the whole file is dropped). This is
    what makes truncation drop the lowest-ranked *symbols* before the lowest-
    ranked *files*, and the lowest-ranked files before higher-ranked ones.
    """
    kept: list[str] = []
    used = 0
    truncated = False

    for item in items:
        if item['kind'] == 'text':
            text = item['text']
            if text.startswith('##') and kept:
                text = '\n' + text
            cost = len(text) + 1
            if kept and used + cost > char_budget:
                truncated = True
                continue
            kept.append(text)
            used += cost
            continue

        entry = item['entry']
        placed = False
        for n in range(len(entry['symbols']), -1, -1):
            candidate = _render_file_block(entry, n)
            cost = len(candidate) + 1
            if not kept or used + cost <= char_budget:
                kept.append(candidate)
                used += cost
                placed = True
                if n < len(entry['symbols']):
                    truncated = True
                break
        if not placed:
            truncated = True

    return kept, truncated


def build_skeleton(project_root: str, task_text: str, token_budget: int = 1200) -> str:
    """Render a deterministic, task-personalised project skeleton.

    Zero LLM/embedder calls, always. Combines a gitignore-aware file walk of
    *project_root* (mtime-cached) with the top task-relevant symbols per most
    relevant file, fit within *token_budget* — truncating the lowest-ranked
    symbols, then files, first, and noting when it had to.

    Args:
        project_root: Absolute path to the project to render a skeleton for.
        task_text: The current task description; drives the relevance ranking.
        token_budget: Approximate max tokens for the returned text (chars/4 or
            tiktoken cl100k_base, whichever is available).

    Returns:
        A compact text block ready for injection into a system message. Never
        raises and never returns an empty string — unexpected failures yield a
        clearly labelled degraded block instead, logged loudly to stderr.
    """
    try:
        return _build_skeleton_inner(project_root, task_text, token_budget)
    except Exception as exc:  # loud, non-raising: this runs on the hot path
        print(
            f"memory.skeleton: build_skeleton failed for {project_root!r} "
            f"({type(exc).__name__}: {exc}); returning a minimal degraded skeleton",
            file=sys.stderr,
        )
        return (
            "# Project skeleton (degraded)\n"
            f"Task: {task_text}\n"
            f"Could not derive a skeleton for {project_root} — {type(exc).__name__}: {exc}\n"
        )


def _build_skeleton_inner(project_root: str, task_text: str, token_budget: int) -> str:
    root = Path(project_root).resolve()
    if not root.is_dir():
        print(
            f"memory.skeleton: {root} is not a directory; returning a minimal degraded skeleton",
            file=sys.stderr,
        )
        return f"# Project skeleton (degraded)\nTask: {task_text}\n{root} is not a directory.\n"

    files = _cached_walk(root)
    task_tokens = _tokenize(task_text)

    manager = _check_lsp_manager()
    if manager is None:
        print(
            "memory.skeleton: no LSP manager available in this process — using static "
            "AST (Python) / regex (other languages) symbol extraction",
            file=sys.stderr,
        )

    ranked_by_path = sorted(files, key=lambda f: (-_path_score(f, task_tokens), f))
    candidates = ranked_by_path[:CANDIDATE_WINDOW]

    scored: list[dict[str, Any]] = []
    for rel_path in candidates:
        path_score = _path_score(rel_path, task_tokens)
        symbols, source = _extract_symbols(root, rel_path, manager)
        matched = _matched_symbol_count(symbols, task_tokens)
        final_score = path_score + 2.0 * min(6, matched)
        if _is_test_path(rel_path):
            final_score *= _TEST_PATH_PENALTY
        scored.append({
            'path': rel_path,
            'score': final_score,
            'symbols': _rank_symbols(symbols, task_tokens)[:SYMBOLS_PER_FILE],
            'source': source,
        })

    scored.sort(key=lambda e: (-e['score'], e['path']))
    relevant = [e for e in scored if e['score'] > 0][:FILES_WITH_SYMBOLS]

    dir_counts: dict[str, int] = {}
    for f in files:
        parts = Path(f).parts
        top = parts[0] if len(parts) > 1 else '.'
        dir_counts[top] = dir_counts.get(top, 0) + 1

    header = (
        f"# Project skeleton — {root.name}\n"
        f"Task: {task_text}\n"
        f"{len(files)} files tracked (gitignore-aware)\n"
    )

    items: list[dict[str, Any]] = [
        {'kind': 'text', 'text': f"## Most relevant files ({len(relevant)} of {len(files)}, ranked)"},
    ]
    if relevant:
        for entry in relevant:
            items.append({'kind': 'file', 'entry': entry})
    else:
        items.append({
            'kind': 'text',
            'text': '(no strong path/symbol match for this task — see project shape below)',
        })

    items.append({'kind': 'text', 'text': '## Project shape (all directories, gitignore-aware)'})
    for d, n in sorted(dir_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        label = '(root)' if d == '.' else f"{d}/"
        items.append({'kind': 'text', 'text': f"{label} — {n} file{'s' if n != 1 else ''}"})

    char_budget = int(token_budget * CHARS_PER_TOKEN_PROXY)
    kept, truncated = _fit_blocks(items, char_budget)
    text = header + '\n'.join(kept)

    # Safety net: the char proxy is conservative but not infallible for every
    # tokenizer — verify with the real estimator and trim further if needed.
    while kept and _estimate_tokens(text) > token_budget:
        kept.pop()
        truncated = True
        text = header + '\n'.join(kept)

    if truncated:
        text += (
            f"\n\n… output truncated to fit the {token_budget}-token budget "
            "(lowest-ranked files/symbols dropped first) …"
        )

    return text
