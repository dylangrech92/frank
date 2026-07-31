"""Lazy gap-fill explorer (D5 / M5): a bounded, memory-less exploration pass
over a cold task area.

Runs entirely in-process, reusing the parent's already-imported tool
registry, but touches NO session/memory state itself -- it is handed a
derived skeleton (``memory.skeleton``) as a starting map, drives its own tiny
LLM+tool loop hard-capped at ``MAX_ITERATIONS`` rounds (the provider is a
slow local model, ~20-40s/call, so this bounds wall-clock, not just token
spend), and returns the model's final structured brief as plain text. It
never calls ``memory.orientation``/``memory.consolidation`` and never writes
to the facts store itself, so it cannot recurse and cannot be the "child"
that needs its own gap-fill.

The caller (``memory.orientation._maybe_explore``) is the one that persists
the brief as anchored knowledge atoms (``persist_brief`` below) and injects
it into the current turn -- this module only explores and parses.

"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory.recall import MemoryContext

# Hard ceiling on LLM+tool rounds. The provider is a slow local model
# (~20-40s/call) -- this is a wall-clock budget as much as a tool-call budget.
MAX_ITERATIONS = 5

# Below this self-rated confidence, a parsed bullet is dropped instead of
# persisted (poisoning defence) -- matches memory.consolidation.CONFIDENCE_FLOOR.
CONFIDENCE_FLOOR = 0.45

# Fallback confidence for a bullet whose self-rating is missing/unparseable --
# conservative (below the "meh but plausible" band) without being the floor
# itself, so a model that simply forgets the tag isn't auto-dropped outright.
_DEFAULT_ITEM_CONFIDENCE = 0.5

# Deliberately minimal and LSP-free: list_files/read_file/find always work
# regardless of whether a language server happens to be running for this
# project, so a tight iteration budget is never wasted on lsp-unavailable
# errors from find_symbol/document_symbols.
_READ_ONLY_TOOLS = ("list_files", "read_file", "find")

_BRIEF_SECTIONS = ("WHERE", "VOCABULARY", "CONCEPTS", "FLOWS", "WHY")

_SYSTEM_PROMPT = f"""You are a codebase explorer building a one-time orientation brief for a coding agent that is about to work on this project. You are memory-less -- this is your only pass, budgeted at {MAX_ITERATIONS} tool-calling rounds, then you MUST stop calling tools and answer with your final brief.

You are given a task area of interest and a derived project-skeleton map (file tree + ranked symbols) as your starting point -- use it instead of re-listing the whole tree. Use list_files/find/read_file to confirm and deepen your understanding of THAT task area specifically, not the whole repo.

When ready (or when your budget is nearly spent), respond with ONLY the brief, no other prose, in EXACTLY this section format (omit a section only if you truly found nothing for it):

## WHERE
- <what lives here, one line> (see <relative/path/to/file>) [confidence: <0.0-1.0>]

## VOCABULARY
- <term>: <one-line definition> (see <relative/path/to/file>) [confidence: <0.0-1.0>]

## CONCEPTS
- <concept>: <one-line explanation> (see <relative/path/to/file>) [confidence: <0.0-1.0>]

## FLOWS
- <flow/pipeline name>: <one-line sequence of what happens> (see <relative/path/to/file>) [confidence: <0.0-1.0>]

## WHY
- <design insight>: <one-line rationale> (see <relative/path/to/file>) [confidence: <0.0-1.0>]

Rules: every bullet is ONE durable, reusable fact -- never narration of what you just did (never "I read file X"). Every bullet MUST end with `(see <path>) [confidence: <0.0-1.0>]` -- <path> names the single most representative file for that fact, as a path relative to the project root; <0.0-1.0> is YOUR OWN rating of how durable, correct, and reusable this fact is for a future task (1.0 = a core fact you verified directly by reading the file; 0.5 = plausible but unconfirmed). Be conservative -- narration, guesses, and filler score LOW and are discarded, never stored. Prefer a few high-value bullets over many trivial ones."""


def _log(msg: str) -> None:
    print(f"explorer: {msg}", file=sys.stderr, flush=True)


def _default_client():
    """Build a standalone ``LLMClient`` from this process's config.

    Mirrors the construction ``main.py`` does at startup -- reads the same
    ``CODING_AGENT_CONFIG`` env var main.py exports (absolute path, so this
    works regardless of the explorer's temporary chdir), falling back to a
    relative ``config.json`` for callers outside the normal CLI entry point.
    """
    from config import load as config_load
    from llm import LLMClient

    config_path = os.environ.get("CODING_AGENT_CONFIG", "config.json")
    cfg = config_load(config_path)
    return LLMClient(cfg.llm)


def _tool_schemas() -> list[dict[str, Any]]:
    from tools.registry import discover, get_tool

    discover()
    schemas: list[dict[str, Any]] = []
    for name in _READ_ONLY_TOOLS:
        tool = get_tool(name)
        if tool is None:
            continue
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
        )
    return schemas


def _assistant_entry(text: str, tool_calls: list) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": "assistant", "content": text or ""}
    if tool_calls:
        entry["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tool_calls
        ]
    return entry


def _dispatch_call(call, last_read: dict[str, str]) -> str:
    """Execute one explorer tool call and render its result, tracking the last
    successfully-read file as a fallback anchor for bullets that omit one."""
    from turn.rendering import render_tool_result
    from tools.registry import dispatch

    if call.name not in _READ_ONLY_TOOLS:
        return (
            f"[{call.name}(error code=not-allowed)]\n"
            f"only read-only exploration tools are available: {', '.join(_READ_ONLY_TOOLS)}"
        )
    # No per-tool activation step any more: the run's mode is fixed at startup and
    # every mode carries all three _READ_ONLY_TOOLS via modes._COMMON_TOOLS, so
    # they are already dispatch-eligible here.
    try:
        result = dispatch(call.name, call.arguments)
    except Exception as exc:
        return f"[{call.name}(error code=explorer-crash)]\n{exc}"

    if call.name == "read_file" and result.status == "success":
        path = call.arguments.get("path")
        if isinstance(path, str) and path.strip():
            last_read["path"] = path.strip()

    return render_tool_result(call.name, result)


def explore(
    project_root: str, task_text: str, skeleton: str, max_iterations: int = MAX_ITERATIONS
) -> tuple[str | None, str | None]:
    """Run one bounded, memory-less exploration pass over *project_root*.

    Drives its own LLM+tool loop (list_files/read_file/find only) for at most
    *max_iterations* rounds; when the budget runs out without a final
    text-only answer, forces one last no-tools wrap-up call asking the model
    to write the brief from whatever it already learned.

    Returns ``(brief_text, fallback_anchor_path)``. *brief_text* is the raw
    structured brief, or ``None`` when nothing usable came back (no LLM
    reachable, no read-only tools registered, every call errored, or the
    model never produced non-empty text) -- this function never raises.
    *fallback_anchor_path* is the last file this pass actually read (relative
    to *project_root*), used by the caller to anchor bullets missing their
    own ``(see <path>)`` annotation.
    """
    try:
        client = _default_client()
    except Exception as exc:
        _log(f"no LLM client available: {exc}")
        return None, None

    tool_schemas = _tool_schemas()
    if not tool_schemas:
        _log("no read-only tools registered; aborting")
        return None, None

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task area of interest: {task_text!r}\n\n"
                f"Starting map (derived project skeleton):\n{skeleton}\n\n"
                "Explore this task area and produce the structured brief."
            ),
        },
    ]

    # tools/list_files.py, read_file.py and find.py all resolve paths against
    # Path.cwd() (not an explicit root parameter) -- so for this pass to
    # actually explore *project_root* regardless of the caller's own cwd, the
    # process must be chdir'd there for the duration of the loop. In the real
    # hot path this is already a no-op (main.py's cwd IS the project root for
    # the whole run); it matters for callers (e.g. a test harness) that point
    # session.project_root somewhere other than the current directory.
    prev_cwd = os.getcwd()
    target_cwd = os.path.abspath(project_root) if project_root else prev_cwd
    chdir_needed = target_cwd != prev_cwd and os.path.isdir(target_cwd)
    last_read: dict[str, str] = {}
    final_text = ""

    if chdir_needed:
        os.chdir(target_cwd)
    try:
        for iteration in range(max_iterations):
            try:
                response = client.chat(messages, tool_schemas)
            except Exception as exc:
                _log(f"llm-error at round {iteration + 1}/{max_iterations}: {exc}")
                break

            if not response.tool_calls:
                final_text = response.text or ""
                break

            _log(f"round {iteration + 1}/{max_iterations}: {len(response.tool_calls)} tool call(s)")
            messages.append(_assistant_entry(response.text, response.tool_calls))
            for call in response.tool_calls:
                rendered = _dispatch_call(call, last_read)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": rendered}
                )
        else:
            # Budget exhausted without a final text-only turn -- force a
            # no-tools wrap-up so the pass still yields a usable brief.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Budget exhausted -- do not call any more tools. Write the "
                        "structured brief now with whatever you have already "
                        "learned, in the exact section format from your instructions."
                    ),
                }
            )
            try:
                response = client.chat(messages, tools=None)
                final_text = response.text or ""
            except Exception as exc:
                _log(f"wrap-up llm-error: {exc}")
    finally:
        if chdir_needed:
            os.chdir(prev_cwd)

    final_text = final_text.strip()
    if not final_text:
        _log("produced no usable brief")
        return None, last_read.get("path")
    _log(f"finished: {len(final_text)} chars, last_read={last_read.get('path')!r}")
    return final_text, last_read.get("path")


# ---------------------------------------------------------------------------
# Brief parsing + persistence
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^#{1,3}\s*(" + "|".join(_BRIEF_SECTIONS) + r")\b", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[-*]\s+(.*)$")
_ANCHOR_RE = re.compile(r"\(\s*see\s+([^)]+?)\s*\)\s*$", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"[\(\[]\s*confidence\s*:\s*([^)\]]*?)\s*[\)\]]\s*$", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def parse_brief(brief_text: str) -> list[dict[str, Any]]:
    """Parse the explorer's structured brief into ``{section, label, text,
    anchor_path, confidence}`` items.

    Best-effort against a weak local model's formatting: recognizes ``#``/
    ``##``/``###`` section headers named after ``_BRIEF_SECTIONS`` and
    ``-``/``*`` bullets under them. A bullet's optional trailing
    ``[confidence: <0-1>]`` (parens also accepted) becomes *confidence* --
    falling back to ``_DEFAULT_ITEM_CONFIDENCE`` when absent or unparseable,
    clamped to ``[0, 1]`` otherwise -- stripped out of *text* before the
    ``(see <path>)`` anchor (also trailing) is parsed the same way into
    *anchor_path*. *label* is the text before the first ``:`` (or the first
    60 characters when there is none), used to derive a stable ``remember()``
    key. Malformed or out-of-section lines are silently skipped -- never
    raises.
    """
    items: list[dict[str, Any]] = []
    section: str | None = None
    for raw_line in (brief_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sec_match = _SECTION_RE.match(line)
        if sec_match:
            section = sec_match.group(1).upper()
            continue
        bullet_match = _BULLET_RE.match(line)
        if not bullet_match or section is None:
            continue
        body = bullet_match.group(1).strip()

        confidence = _DEFAULT_ITEM_CONFIDENCE
        conf_match = _CONFIDENCE_RE.search(body)
        if conf_match:
            try:
                parsed_confidence = float(conf_match.group(1))
                if not math.isfinite(parsed_confidence):
                    raise ValueError("non-finite confidence")
                confidence = max(0.0, min(1.0, parsed_confidence))
            except (TypeError, ValueError):
                confidence = _DEFAULT_ITEM_CONFIDENCE
            body = body[: conf_match.start()].strip()

        anchor_match = _ANCHOR_RE.search(body)
        anchor_path = anchor_match.group(1).strip() if anchor_match else None
        if anchor_match:
            body = body[: anchor_match.start()].strip()
        if not body:
            continue
        label = body.split(":", 1)[0].strip() if ":" in body else body[:60].strip()
        items.append(
            {
                "section": section,
                "label": label,
                "text": body,
                "anchor_path": anchor_path,
                "confidence": confidence,
            }
        )
    return items


def _slug(section: str, label: str) -> str:
    """Derive a short, stable ``remember()`` key from a section + label pair."""
    raw = f"explorer-{section.lower()}-{label.lower()}"
    slug = _SLUG_RE.sub("-", raw).strip("-")
    return (slug or "explorer-item")[:60]


def _resolve_anchor(project_root: str, raw_path: str | None) -> str | None:
    """Resolve a model-supplied (possibly relative) path to an absolute anchor.

    Anchors must be absolute -- ``memory.anchor.is_stale`` re-opens the path
    directly at recall time, regardless of the caller's cwd at that point.
    Returns ``None`` (repo-wide) for anything not a non-empty string.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = raw_path.strip()
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(project_root, path))


def persist_brief(
    ctx: "MemoryContext",
    project_root: str,
    brief_text: str,
    fallback_anchor: str | None = None,
) -> list[int]:
    """Persist a parsed brief as anchored, durable knowledge atoms.

    Each bullet becomes one ``remember()`` call under kind ``"project"``
    (durable, no TTL -- see ``memory.atomic.KINDS``), keyed by a slug of its
    section+label so a later re-exploration or consolidation pass supersedes
    it via the same ``(kind, key)`` instead of duplicating. A bullet missing
    its own ``(see <path>)`` annotation falls back to *fallback_anchor* (the
    last file this pass actually read); with neither, the atom is repo-wide
    (``anchor_path=None``). Confidence is the model's OWN self-rating for
    that bullet (``parse_brief``'s ``confidence`` field, conservative default
    when the model omits it) rather than a flat value for the whole brief --
    this is LLM-attested, not diff-verified, so a bullet below
    ``CONFIDENCE_FLOOR`` (matching ``memory.consolidation``'s poisoning
    defence) is skipped instead of written.

    Returns the list of new fact ids actually written (low-confidence
    bullets are not counted). Never raises -- a single bullet's write
    failure is logged and skipped, the rest still proceed.
    """
    from memory import anchor as _anchor
    from memory.atomic import remember

    written: list[int] = []
    skipped_low_confidence = 0
    for item in parse_brief(brief_text):
        item_confidence = item["confidence"]
        if item_confidence < CONFIDENCE_FLOOR:
            skipped_low_confidence += 1
            continue
        raw_path = item.get("anchor_path") or fallback_anchor
        anchor_path = _resolve_anchor(project_root, raw_path)
        anchor_hash, learned_commit = _anchor.anchor_for(anchor_path)
        key = _slug(item["section"], item["label"])
        try:
            fact_id = remember(
                ctx,
                "project",
                key,
                item["text"],
                anchor_path=anchor_path,
                anchor_hash=anchor_hash,
                learned_commit=learned_commit,
                confidence=item_confidence,
                source="explorer",
            )
            written.append(fact_id)
        except Exception as exc:
            _log(f"write-error key={key!r}: {exc}")
    _log(f"persisted {len(written)} atom(s), skipped {skipped_low_confidence} low-confidence bullet(s)")
    return written
