"""Pure parsers for V8 perf and heap profile JSON formats."""

from __future__ import annotations

import json
# Profile-artifact parsers — V8 .cpuprofile JSON and Xdebug cachegrind.
# ---------------------------------------------------------------------------


def parse_cpuprofile(path: str) -> list[dict]:
    """Parse a V8 ``.cpuprofile`` JSON file (produced by ``node --cpu-prof``).

    The profile is a flat list of nodes with children referenced by id,
    plus ``samples`` and ``timeDeltas`` that together define the sampling
    cadence.  We compute per-node self/total time from the mean sample
    interval and return the flat list sorted by self time descending.

    Raises:
        ValueError: On malformed JSON or a missing ``nodes`` key.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read cpuprofile {path}: {exc}") from exc

    if not isinstance(data, dict) or "nodes" not in data:
        raise ValueError("cpuprofile JSON is missing the 'nodes' key")

    nodes_list = data["nodes"]
    if not isinstance(nodes_list, list):
        raise ValueError("cpuprofile 'nodes' is not a list")

    start_time = data.get("startTime", 0) or 0
    end_time = data.get("endTime", 0) or 0
    samples = data.get("samples") or []

    num_samples = max(len(samples), 1)
    mean_interval_us = (end_time - start_time) / num_samples
    mean_interval_s = mean_interval_us / 1_000_000

    # id -> node map; missing ids are simply skipped (not a crash).
    node_map: dict[int, dict] = {}
    for node in nodes_list:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id is None:
            continue
        node_map[node_id] = node

    # Recursive total_s with memoisation.
    total_cache: dict[int, float] = {}

    def _total(node_id: int) -> float:
        cached = total_cache.get(node_id)
        if cached is not None:
            return cached
        node = node_map.get(node_id)
        if node is None:
            total_cache[node_id] = 0.0
            return 0.0
        hit_count = node.get("hitCount", 0) or 0
        children = node.get("children") or []
        self_s = hit_count * mean_interval_s
        child_total = sum(_total(cid) for cid in children if cid in node_map)
        result = self_s + child_total
        total_cache[node_id] = result
        return result

    results: list[dict] = []
    for node in nodes_list:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id is None:
            continue
        hit_count = node.get("hitCount", 0) or 0
        total_s = _total(node_id)
        if hit_count > 0 or total_s > 0:
            call_frame = node.get("callFrame") or {}
            results.append(
                {
                    "function": call_frame.get("functionName") or "(anonymous)",
                    "file": call_frame.get("url") or "",
                    "line": call_frame.get("lineNumber", 0) or 0,
                    "self_s": hit_count * mean_interval_s,
                    "total_s": total_s,
                    "hits": int(hit_count),
                }
            )

    results.sort(key=lambda r: r["self_s"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# parse_heapprofile — V8 .heapprofile JSON (produced by node --heap-prof).
# ---------------------------------------------------------------------------


def parse_heapprofile(path: str) -> list[dict]:
    """Parse a V8 ``.heapprofile`` JSON file (produced by ``node --heap-prof``).

    Unlike ``.cpuprofile``'s flat id-referenced node list, a heap profile is a
    single recursive tree rooted at ``"head"``. Each node's ``selfSize`` is
    the bytes allocated and attributed directly to that frame; a node's total
    is its own ``selfSize`` plus the total of all its children. Nodes sharing
    the same ``(functionName, url, lineNumber)`` are merged by summing, and
    the flat result is sorted by self bytes descending.

    Raises:
        ValueError: On malformed JSON or a missing ``head`` key.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read heapprofile {path}: {exc}") from exc

    if not isinstance(data, dict) or "head" not in data:
        raise ValueError("heapprofile JSON is missing the 'head' key")

    root = data["head"]
    if not isinstance(root, dict):
        raise ValueError("heapprofile 'head' is not an object")

    merged: dict[tuple[str, str, int], dict[str, int]] = {}

    def _walk(node: dict) -> int:
        """Merge *node* and its subtree into ``merged``; return its total_bytes."""
        call_frame = node.get("callFrame") or {}
        function_name = call_frame.get("functionName") or "(anonymous)"
        url = call_frame.get("url") or ""
        line_number = call_frame.get("lineNumber", 0) or 0
        self_bytes = node.get("selfSize", 0) or 0

        children_total = 0
        for child in node.get("children") or []:
            if isinstance(child, dict):
                children_total += _walk(child)

        total_bytes = self_bytes + children_total

        key = (function_name, url, line_number)
        entry = merged.get(key)
        if entry is None:
            entry = {"self_bytes": 0, "total_bytes": 0}
            merged[key] = entry
        entry["self_bytes"] += self_bytes
        entry["total_bytes"] += total_bytes

        return total_bytes

    _walk(root)

    results: list[dict] = []
    for (function_name, url, line_number), sizes in merged.items():
        results.append(
            {
                "function": function_name,
                "file": url,
                "line": line_number,
                "self_bytes": sizes["self_bytes"],
                "total_bytes": sizes["total_bytes"],
            }
        )

    results.sort(key=lambda r: r["self_bytes"], reverse=True)
    return results
