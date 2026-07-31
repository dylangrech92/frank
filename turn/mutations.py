"""How a turn learns which files it actually changed.

The bookkeeping behind ``turn_report["files_changed"]`` and the reverted-file
annotation: ``_capture_preimage`` records a target's turn-start hash at the
dispatch seam before a mutating tool writes, ``_record_file_mutations`` folds the
resulting mutation events into the log, and ``_scan_new_mutations`` classifies a
raw event slice into "did anything relevant change, and where".

The mutation bus itself — the ``_TURN_MUTATIONS`` sink and the ``_mutate_tracker``
callback subscribed to it at import time — deliberately stays in ``agent``: that
subscription is an import-time side effect, and moving it somewhere nothing
imports at module level would kill file tracking silently.
"""

from __future__ import annotations

import hashlib

from llm import ToolCall
from tools.registry import get_tool
from turn.lint_delta import _lint_resolve_call_path
from turn.state import _TurnState


def _scan_new_mutations(events: list[dict]) -> tuple[bool, set[str]]:
    """Inspect newly observed mutation events for H1/H5 tracking.

    Args:
        events: A slice of ``_TURN_MUTATIONS`` added since the last check.

    Returns:
        ``(any_relevant, paths)`` — whether any event's ``kind`` is one of
        ``created``/``changed``/``renamed`` (the kinds that count as a real file
        mutation for verification/graph-memory purposes), and the set of
        distinct ``path`` values among those relevant events.
    """
    any_relevant = False
    paths: set[str] = set()
    for event in events:
        if event.get("kind") in ("created", "changed", "renamed"):
            any_relevant = True
            path_val = event.get("path")
            if path_val:
                paths.add(path_val)
    return any_relevant, paths


# Files larger than this are not pre-imaged for net-change detection: hashing
# them on every mutating call would tax the dispatch hot path, and the reverted
# flag is a best-effort truthful signal, not a guarantee — an un-hashed path
# stays UNKNOWN (no reverted key), never guessed.
_PREIMAGE_MAX_BYTES = 5 * 1024 * 1024


def _capture_preimage(state: "_TurnState", call: ToolCall, project_root: str) -> None:
    """Record the turn-start state of *call*'s target file, once per path.

    Called at the dispatch seam BEFORE a mutating tool writes, so the recorded
    hash reflects the file as it was when the turn first touched it. The key is
    the resolved absolute path the mutation event (and thus the ``files_changed``
    entry) uses — via the same ``_lint_resolve_call_path`` resolution — so the
    two join exactly at annotation time.

    Records nothing (leaving the path UNKNOWN) unless the call carries a
    resolvable ``path`` argument on a non-parallel_safe tool: a ``run_command``
    shell side effect has no ``path`` argument, so a file it mutates first has no
    capturable pre-image and stays unknown, never guessed. A nonexistent target
    records ``None`` (did-not-exist at turn start); an oversize
    (> ``_PREIMAGE_MAX_BYTES``) or unreadable file records nothing (unknown).

    A path already mutated this turn is skipped (``files_changed_seen``): its
    turn-start state is gone, so a pre-image taken now would be a mid-turn state,
    not the start — this is what keeps a path whose FIRST mutation was a
    pre-imageless ``run_command`` side effect unknown even when a later edit tool
    touches it. Combined with the ``preimages`` guard, first capture per path
    wins and a later write never overwrites a recorded turn-start state.
    """
    if getattr(get_tool(call.name), "parallel_safe", False):
        return  # read-only tools never mutate — nothing to pre-image
    key = _lint_resolve_call_path(call, project_root)
    if key is None or key in state.preimages or key in state.files_changed_seen:
        return
    from pathlib import Path as _Path

    target = _Path(key)
    if not target.exists():
        state.preimages[key] = None  # did not exist at turn start
        return
    try:
        if target.stat().st_size > _PREIMAGE_MAX_BYTES:
            return  # too large to hash cheaply — leave unknown, never guess
        state.preimages[key] = hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError:
        return  # unreadable — unknown


def _record_file_mutations(
    state: "_TurnState", call: ToolCall, new_events: list[dict]
) -> None:
    """Fold this call's mutation events into the S4 ``files_changed`` log.

    Correlated with the dispatched call's own slice of ``_TURN_MUTATIONS`` (NOT
    the module-level list read later — ``diagnostics_inject_summary`` drains it at
    the end of the dispatch round). Counts EVERY relevant mutation event (before
    the path dedup) so a second edit to an already-mutated path still bumps
    ``mutation_events`` — the stamp the verification dedup gates on, which
    ``len(mutated_paths)`` would miss. Appends to ``files_changed`` deduped by
    path, first-tool-wins, in order of first mutation.
    """
    for event in new_events:
        if event.get("kind") not in ("created", "changed", "renamed"):
            continue
        state.mutation_events += 1
        path = event.get("path")
        if path and path not in state.files_changed_seen:
            state.files_changed_seen.add(path)
            state.turn_report["files_changed"].append(
                {"path": path, "tool": call.name}
            )
