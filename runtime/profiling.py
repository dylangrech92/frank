"""Measured-subprocess core for performance-profiling tools.

Provides ``run_measured`` which runs a shell command and captures kernel-level
CPU/RSS (via :func:`os.wait4`'s ``rusage``) plus a sampled view of the entire
process tree via optional ``psutil``.  Differs from :mod:`runtime.process`'s
``run_one_shot`` in that this module reaps the child itself via ``os.wait4`` so
that ``rusage`` is available — ``Popen.communicate()`` reaps the child first and
cannot return it.

The module imports cleanly even when ``psutil`` is not installed: the sampler
thread simply stays idle and sampled fields default to zero.
"""

from __future__ import annotations

import gzip
import json
import dataclasses
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import IO, TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from subprocess import Popen

# ---------------------------------------------------------------------------
# Optional psutil import — never raise at import time.
# ---------------------------------------------------------------------------

try:
    import psutil  # type: ignore[import-untyped]

    PSUTIL_AVAILABLE: bool = True
except ImportError:
    psutil = cast(Any, None)
    PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# MeasuredRun — frozen snapshot of a single measured subprocess run.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MeasuredRun:
    """A complete, immutable snapshot of a measured subprocess run."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    wall_s: float
    cpu_user_s: float
    cpu_sys_s: float
    max_rss_bytes: int
    sampled_peak_rss_bytes: int
    sampled_mean_cpu_pct: float
    max_procs: int


# ---------------------------------------------------------------------------
# run_measured — the heart of the module.
# ---------------------------------------------------------------------------


def run_measured(
    cmd: str,
    root: str,
    timeout_seconds: int,
    sample_interval_s: float = 0.1,
) -> MeasuredRun:
    """Run *cmd* as a one-shot shell process and capture kernel rusage.

    Args:
        cmd: Shell command string to execute.
        root: Working directory for the subprocess (cwd).
        timeout_seconds: Max seconds before forcible termination.
        sample_interval_s: Sampling interval for the optional psutil sampler.

    Returns:
        A :class:`MeasuredRun` with stdout, stderr, exit code, timing, and
        rusage-derived CPU/RSS metrics plus optional sampled process-tree data.
    """
    t0 = time.monotonic()
    proc: Popen[bytes] = subprocess.Popen(
        cmd,
        shell=True,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # --- Daemon reader threads drain pipes so the child never blocks. ---
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    stdout_lock = threading.Lock()
    stderr_lock = threading.Lock()

    def _drain(stream: IO[bytes], buf: bytearray, lock: threading.Lock) -> None:
        for chunk in stream:
            with lock:
                buf.extend(chunk)
        stream.close()

    stdout_reader = threading.Thread(
        target=_drain,
        args=(proc.stdout, stdout_buf, stdout_lock),
        daemon=True,
        name="prof-stdout",
    )
    stderr_reader = threading.Thread(
        target=_drain,
        args=(proc.stderr, stderr_buf, stderr_lock),
        daemon=True,
        name="prof-stderr",
    )
    stdout_reader.start()
    stderr_reader.start()

    # --- Optional psutil sampler thread. ---
    sampled_peak_rss = 0
    sampled_mean_cpu = 0.0
    max_procs_seen = 0
    sampler_done = threading.Event()

    _cpu_samples: list[float] = []
    if PSUTIL_AVAILABLE:
        _peak_lock = threading.Lock()

        def _sampler() -> None:
            nonlocal sampled_peak_rss, max_procs_seen
            first_tick = True
            while not sampler_done.is_set():
                try:
                    parent = psutil.Process(proc.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    break
                try:
                    children = parent.children(recursive=True)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    children = []
                members = [parent, *children]
                total_rss = 0
                cpu_sum = 0.0
                for member in members:
                    try:
                        total_rss += member.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                        continue
                    try:
                        pct = member.cpu_percent(None)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                        continue
                    if first_tick:
                        # First call to cpu_percent always returns 0.0; skip it.
                        continue
                    cpu_sum += pct
                if first_tick:
                    first_tick = False
                else:
                    _cpu_samples.append(cpu_sum)
                with _peak_lock:
                    if total_rss > sampled_peak_rss:
                        sampled_peak_rss = total_rss
                max_procs_seen = max(max_procs_seen, len(members))
                sampler_done.wait(sample_interval_s)

        sampler_thread = threading.Thread(
            target=_sampler,
            daemon=True,
            name="prof-sampler",
        )
        sampler_thread.start()
    else:
        sampler_thread = None  # type: ignore[assignment]

    # --- Timeout killer (mirrors process.py's killpg pattern). ---
    timed_out = False

    def _kill_on_timeout() -> None:
        nonlocal timed_out
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            timed_out = True
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout_seconds, _kill_on_timeout)
    timer.start()

    # --- Reap via os.wait4 to capture rusage. ---

    try:
        _, status, rusage = os.wait4(proc.pid, 0)
    except ChildProcessError as exc:
        raise RuntimeError(
            "run_measured lost the child's exit status: something else reaped "
            f"pid {proc.pid} before os.wait4 could (rusage is unrecoverable)"
        ) from exc

    # Mark returncode so later proc.poll() does NOT try to reap again.
    proc.returncode = os.waitstatus_to_exitcode(status)
    exit_code = proc.returncode

    # Signal the sampler to stop and wait for it to finish its current tick.
    if sampler_thread is not None:
        sampler_done.set()
        sampler_thread.join(timeout=sample_interval_s * 2 + 0.5)

    # Cancel the timeout timer (no-op if it already fired).
    timer.cancel()

    # Compute wall time after reap.
    wall_s = time.monotonic() - t0

    # Compute CPU from rusage.
    cpu_user_s = rusage.ru_utime
    cpu_sys_s = rusage.ru_stime

    # max_rss: darwin reports bytes; Linux/other platforms report KiB.
    if sys.platform == "darwin":
        max_rss_bytes = rusage.ru_maxrss
    else:
        max_rss_bytes = rusage.ru_maxrss * 1024

    # Join reader threads first so we don't miss trailing output.
    stdout_reader.join(timeout=1)
    stderr_reader.join(timeout=1)

    # Decode captured bytes.
    with stdout_lock:
        stdout_str = bytes(stdout_buf).decode("utf-8", errors="replace")
    with stderr_lock:
        stderr_str = bytes(stderr_buf).decode("utf-8", errors="replace")

    # Compute sampled mean CPU.
    if PSUTIL_AVAILABLE and _cpu_samples:
        sampled_mean_cpu = sum(_cpu_samples) / len(_cpu_samples)

    return MeasuredRun(
        stdout=stdout_str,
        stderr=stderr_str,
        exit_code=exit_code,
        timed_out=timed_out,
        wall_s=wall_s,
        cpu_user_s=cpu_user_s,
        cpu_sys_s=cpu_sys_s,
        max_rss_bytes=max_rss_bytes,
        sampled_peak_rss_bytes=sampled_peak_rss,
        sampled_mean_cpu_pct=sampled_mean_cpu,
        max_procs=max_procs_seen,
    )


# ---------------------------------------------------------------------------
# which_interpreter — locate a language runtime.
# ---------------------------------------------------------------------------

_INTERPRETER_ENV = {
    "python": "CODING_AGENT_PY_BIN",
    "node": "CODING_AGENT_NODE_BIN",
    "php": "CODING_AGENT_PHP_BIN",
}

_INTERPRETER_WHICH = {
    "python": ("python3", "python"),
    "node": ("node",),
    "php": ("php",),
}


def which_interpreter(language: str) -> str | None:
    """Return the path to an interpreter for *language*, or ``None``.

    Args:
        language: One of ``"python"``, ``"node"``, ``"php"``.

    Raises:
        ValueError: If *language* is not one of the supported interpreters.
    """
    env_key = _INTERPRETER_ENV.get(language)
    if env_key is None:
        raise ValueError(f"Unknown interpreter language: {language!r}")

    # Environment override takes precedence.
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val

    # Fall back to shutil.which for each candidate in priority order.
    for name in _INTERPRETER_WHICH[language]:
        found = shutil.which(name)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# php_xdebug_status — detect whether xdebug is loaded/available for a PHP bin.
# ---------------------------------------------------------------------------

_xdebug_cache: dict[str, str] = {}


def php_xdebug_status(php_bin: str) -> str:
    """Return ``"loaded"`` | ``"available"`` | ``"missing"`` for *php_bin*.

    Results are cached in a module-level dict keyed by *php_bin*.
    """
    cached = _xdebug_cache.get(php_bin)
    if cached is not None:
        return cached

    result = "missing"
    try:
        proc = subprocess.run(
            [php_bin, "-m"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if "xdebug" in proc.stdout.lower():
            result = "loaded"
        else:
            # Try loading via -dzend_extension.
            proc2 = subprocess.run(
                [php_bin, "-dzend_extension=xdebug", "-m"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if "xdebug" in proc2.stdout.lower():
                result = "available"
    except (subprocess.SubprocessError, FileNotFoundError):
        result = "missing"

    _xdebug_cache[php_bin] = result
    return result


# ---------------------------------------------------------------------------
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


def parse_cachegrind(path: str) -> dict:
    """Parse an Xdebug 3 cachegrind profile file.

    The file may be gzip-compressed (detected by the ``\\x1f\\x8b`` magic
    bytes); otherwise it is opened as plain text with ``errors="replace"``.

    Supports the grammar subset used by ``profile_hotspots``: an
    ``events:`` header, ``fl=(id)`` / ``fn=(id)`` declarations with the
    compressed-name ``(id)`` alias mechanism, cost lines after ``fn=``
    blocks, ``cfn=`` / ``calls=`` edges, and a ``summary:`` line.

    Raises:
        ValueError: If the file has no ``events:`` header.
    """
    try:
        with open(path, "rb") as fh:
            magic = fh.read(2)
    except OSError as exc:
        raise ValueError(f"Failed to read cachegrind file {path}: {exc}") from exc

    is_gzip = magic == b"\x1f\x8b"

    events: list[str] = []
    file_table: dict[int, str] = {}
    fn_table: dict[int, str] = {}
    functions: dict[str, dict] = {}
    summary: dict[str, int] = {}

    current_file: str = ""
    current_fn_name: str | None = None
    current_fn_file: str = ""

    # State that bridges a ``cfl=``/``cfn=``/``calls=`` call-edge sequence to
    # the cost line that follows it.
    cfl_pending: str | None = None
    pending_cfn_name: str | None = None
    pending_cfn_file: str = ""
    awaiting_edge_cost = False

    def _open_input() -> IO[str]:
        if is_gzip:
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        return open(path, "r", encoding="utf-8", errors="replace")

    def _ensure_fn(name: str, file_: str) -> dict:
        """Return the per-name aggregate dict, creating it if needed."""
        entry = functions.get(name)
        if entry is None:
            entry = {
                "function": name,
                "file": file_,
                "calls": 0,
                "self": {e: 0 for e in events},
                "inclusive": {e: 0 for e in events},
            }
            functions[name] = entry
        return entry

    def _parse_alias(rest: str) -> tuple[int | None, str | None]:
        """Parse a compressed-name payload: ``(id) value``, bare ``(id)``,
        or a plain literal.

        Returns ``(id, value)``. ``id`` is the compressed-name integer when
        the payload starts with ``(id)``, else ``None``. ``value`` is the
        trailing definition text when present, else ``None`` — a bare
        ``(id)`` with no trailing text is a REFERENCE to a previously
        defined id, never a redefinition to an empty string.
        """
        rest = rest.strip()
        if rest.startswith("(") and ")" in rest:
            idx = rest.index(")")
            try:
                id_ = int(rest[1:idx])
            except ValueError:
                id_ = None
            value = rest[idx + 1 :].strip() or None
            return id_, value
        return None, (rest or None)

    with _open_input() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("events:"):
                events = line[len("events:"):].split()
                continue

            if line.startswith("fl="):
                id_, value = _parse_alias(line[len("fl="):])
                if id_ is not None:
                    if value is not None:
                        file_table[id_] = value
                        current_file = value
                    else:
                        current_file = file_table.get(id_, current_file)
                elif value is not None:
                    current_file = value
                continue

            if line.startswith("fn="):
                id_, value = _parse_alias(line[len("fn="):])
                if id_ is not None:
                    if value is not None:
                        fn_table[id_] = value
                        current_fn_name = value
                    else:
                        current_fn_name = fn_table.get(id_, f"({id_})")
                elif value is not None:
                    current_fn_name = value
                else:
                    current_fn_name = None
                current_fn_file = current_file
                pending_cfn_name = None
                awaiting_edge_cost = False
                continue

            if line.startswith("cfl="):
                id_, value = _parse_alias(line[len("cfl="):])
                if id_ is not None:
                    if value is not None:
                        file_table[id_] = value
                        cfl_pending = value
                    else:
                        cfl_pending = file_table.get(id_, current_file)
                elif value is not None:
                    cfl_pending = value
                continue

            if line.startswith("cob=") or line.startswith("ob="):
                # Object-file annotations — not part of the cost model.
                continue

            if line.startswith("cfn="):
                id_, value = _parse_alias(line[len("cfn="):])
                if id_ is not None:
                    if value is not None:
                        fn_table[id_] = value
                        resolved = value
                    else:
                        resolved = fn_table.get(id_, f"({id_})")
                elif value is not None:
                    resolved = value
                else:
                    resolved = "(unknown)"
                pending_cfn_name = resolved
                pending_cfn_file = cfl_pending if cfl_pending is not None else current_file
                cfl_pending = None
                awaiting_edge_cost = False
                continue

            if line.startswith("calls="):
                parts = line.split()
                try:
                    count = int(parts[0].split("=", 1)[1])
                except (ValueError, IndexError):
                    count = 0
                # The remaining tokens (e.g. the ``20`` in ``calls=4 20``)
                # are target POSITIONS, not costs — deliberately ignored.
                name = pending_cfn_name if pending_cfn_name is not None else "(unknown)"
                fn_data = _ensure_fn(name, pending_cfn_file)
                fn_data["calls"] += count
                pending_cfn_name = None
                awaiting_edge_cost = True
                continue

            if line.startswith("summary:"):
                parts = line[len("summary:"):].split()
                for i, cost_str in enumerate(parts):
                    if i < len(events):
                        try:
                            summary[events[i]] = summary.get(events[i], 0) + int(
                                cost_str
                            )
                        except ValueError:
                            pass
                continue

            # Cost line: ``<line> <cost1> [<cost2>]``.
            parts = line.split()
            if len(parts) < 2:
                continue
            if awaiting_edge_cost:
                # The cost line right after a cfn=/calls= pair is the
                # INCLUSIVE cost of that call edge, attributed to the
                # caller's inclusive total — never to the caller's self
                # cost, and never double-counted into the callee (whose
                # own self/inclusive comes from its own fn= block).
                awaiting_edge_cost = False
                if current_fn_name is not None:
                    fn_data = _ensure_fn(current_fn_name, current_fn_file)
                    for i, cost_str in enumerate(parts[1:]):
                        if i < len(events):
                            try:
                                fn_data["inclusive"][events[i]] += int(cost_str)
                            except ValueError:
                                pass
                continue
            if current_fn_name is not None:
                fn_data = _ensure_fn(current_fn_name, current_fn_file)
                for i, cost_str in enumerate(parts[1:]):
                    if i < len(events):
                        try:
                            cost = int(cost_str)
                        except ValueError:
                            continue
                        fn_data["self"][events[i]] += cost
                        fn_data["inclusive"][events[i]] += cost

    if not events:
        raise ValueError("cachegrind file has no 'events:' header")

    func_list = list(functions.values())
    first_event = events[0]
    func_list.sort(key=lambda f: f["self"].get(first_event, 0), reverse=True)

    return {
        "events": events,
        "functions": func_list,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Render helpers — small formatting utilities for profiling output.
# ---------------------------------------------------------------------------


def fmt_bytes(n: int) -> str:
    """Format *n* bytes as B / KB / MB / GB with one decimal place."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.1f} GB"


def fmt_seconds(s: float) -> str:
    """Format *s* seconds as ``"1.234s"`` or milliseconds for sub-second values."""
    if s < 1.0:
        return f"{s * 1000:.1f}ms"
    return f"{s:.3f}s"


def fmt_count(n: int) -> str:
    """Format *n* with thousands separators (e.g. ``1,234,567``)."""
    return f"{n:,}"


def render_top_table(
    headers: list[str],
    rows: list[tuple],
    top: int,
    total_label: str,
) -> str:
    """Render a fixed-width table truncated to *top* rows.

    Args:
        headers: Column headers.
        rows: Row tuples (one per table row).
        top: Maximum number of rows to display.
        total_label: Label for the total count (e.g. ``"files"``).

    Returns:
        A string with fixed-width columns, truncated to *top* rows, followed by
        a summary line like ``"(top 10 of 42 files)"``.
    """
    if not headers:
        return ""

    # Compute column widths from headers and rows.
    num_cols = len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            cell_str = str(cell)
            if i < num_cols and len(cell_str) > col_widths[i]:
                col_widths[i] = len(cell_str)

    # Format header row.
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))

    # Truncate rows.
    display_rows = rows[:top]
    total_rows = len(rows)

    lines = [header_line]
    for row in display_rows:
        line = "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)

    # Summary line.
    if total_rows > top:
        summary = f"(top {top} of {total_rows} {total_label})"
    else:
        summary = f"({total_rows} {total_label})"
    lines.append(summary)

    return "\n".join(lines)
