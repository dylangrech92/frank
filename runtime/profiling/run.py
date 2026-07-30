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

