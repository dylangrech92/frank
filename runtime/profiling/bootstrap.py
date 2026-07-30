"""Self-contained Python profiler bootstrap script templates.

Two templates are provided:
- ``_TRACEMALLOC_BOOTSTRAP_TEMPLATE`` — stdlib ``tracemalloc`` memory tracking.
- ``_TRACE_BOOTSTRAP_TEMPLATE`` — ``sys.settrace`` call counter with optional
  per-line hitting on a focus file.

Both render into a standalone `.py` bootstrap that invokes the target through
``runpy`` so it keeps running past ``sys.exit``/exceptions and writes its
JSON result from a ``finally`` block.
"""

from __future__ import annotations

import json
import os
# write_python_bootstrap — self-contained Python profiler bootstrap scripts.
# ---------------------------------------------------------------------------

# Both templates embed the config as a single ``json.loads(<literal>)`` call
# via a placeholder token, substituted with ``repr(json.dumps(config))`` —
# this sidesteps any collision between str.format()-style braces and the
# literal ``{``/``}`` characters that pervade the generated code itself.

_TRACEMALLOC_BOOTSTRAP_TEMPLATE = '''"""Auto-generated tracemalloc bootstrap. Do not edit directly."""

import json
import runpy
import sys
import tracemalloc

_CONFIG = json.loads(@@CONFIG_JSON@@)

_target = _CONFIG["target"]
_is_module = _CONFIG["is_module"]
_args = _CONFIG["args"]
_result_path = _CONFIG["result_path"]

sys.argv = [_target] + list(_args)

tracemalloc.start(25)

_target_error = None

try:
    # The returned namespace is kept alive (not discarded) so any
    # module-level state the target leaves behind is still resident when
    # we measure/snapshot in the finally block below.
    if _is_module:
        _target_globals = runpy.run_module(
            _target, run_name="__main__", alter_sys=True
        )
    else:
        _target_globals = runpy.run_path(_target, run_name="__main__")
except SystemExit as exc:
    _target_error = str(exc) or None
except BaseException as exc:
    _target_error = str(exc) or None
finally:
    # Snapshot BEFORE stop() -- stopping first would discard the traces.
    _current, _peak = tracemalloc.get_traced_memory()
    _snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    _top = []
    for _stat in _snapshot.statistics("lineno")[:100]:
        _frame = _stat.traceback[0]
        _top.append(
            {
                "file": _frame.filename,
                "line": _frame.lineno,
                "size_bytes": _stat.size,
                "count": _stat.count,
            }
        )

    _result = {
        "kind": "tracemalloc",
        "current_bytes": _current,
        "peak_bytes": _peak,
        "top": _top,
        "target_error": _target_error,
    }
    with open(_result_path, "w", encoding="utf-8") as _fh:
        json.dump(_result, _fh)
'''

_TRACE_BOOTSTRAP_TEMPLATE = '''"""Auto-generated sys.settrace bootstrap. Do not edit directly."""

import json
import runpy
import sys

_CONFIG = json.loads(@@CONFIG_JSON@@)

_target = _CONFIG["target"]
_is_module = _CONFIG["is_module"]
_args = _CONFIG["args"]
_result_path = _CONFIG["result_path"]
_focus_file = _CONFIG.get("focus_file") or ""

sys.argv = [_target] + list(_args)

_BOOTSTRAP_PATH = __file__

_call_counts = {}
_line_hits = {}
_depth = 0
_max_depth = 0


def _local_tracer(frame, event, arg):
    """Per-frame tracer: counts focus-file line hits, pops depth on return."""
    global _depth
    if event == "line":
        _filename = frame.f_code.co_filename
        if _filename == _focus_file:
            _key = (_filename, frame.f_lineno)
            _line_hits[_key] = _line_hits.get(_key, 0) + 1
    elif event == "return":
        _depth -= 1
    return _local_tracer


def _global_tracer(frame, event, arg):
    """Global tracer: counts calls and stack depth for every traced call.

    A local tracer is attached to every non-excluded frame -- even ones
    outside focus_file -- because returning None here would also silence
    that frame's "return" event, breaking depth bookkeeping. Per-line
    overhead for non-focus frames is instead avoided via f_trace_lines,
    which suppresses "line" event dispatch entirely at the interpreter
    level (verified: no per-line callback is made when it is False).
    """
    global _depth, _max_depth
    if event != "call":
        return _global_tracer

    _code = frame.f_code
    _filename = _code.co_filename
    if _filename.startswith("<") or _BOOTSTRAP_PATH in _filename:
        return None

    _depth += 1
    if _depth > _max_depth:
        _max_depth = _depth

    _key = (_filename, _code.co_firstlineno, _code.co_name)
    _call_counts[_key] = _call_counts.get(_key, 0) + 1

    if _filename != _focus_file:
        frame.f_trace_lines = False
    return _local_tracer


_target_error = None

sys.settrace(_global_tracer)
try:
    # The returned namespace is kept alive (not discarded) so any
    # module-level state the target leaves behind is still resident when
    # we measure/snapshot in the finally block below.
    if _is_module:
        _target_globals = runpy.run_module(
            _target, run_name="__main__", alter_sys=True
        )
    else:
        _target_globals = runpy.run_path(_target, run_name="__main__")
except SystemExit as exc:
    _target_error = str(exc) or None
except BaseException as exc:
    _target_error = str(exc) or None
finally:
    sys.settrace(None)

    _calls = [
        {"file": _f, "line": _ln, "function": _fn, "count": _c}
        for (_f, _ln, _fn), _c in _call_counts.items()
    ]
    _calls.sort(key=lambda r: r["count"], reverse=True)
    _calls = _calls[:200]

    _hits = [
        {"file": _f, "line": _ln, "count": _c}
        for (_f, _ln), _c in _line_hits.items()
    ]
    _hits.sort(key=lambda r: r["count"], reverse=True)
    _hits = _hits[:200]

    _result = {
        "kind": "trace",
        "calls": _calls,
        "max_depth": _max_depth,
        "line_hits": _hits,
        "target_error": _target_error,
    }
    with open(_result_path, "w", encoding="utf-8") as _fh:
        json.dump(_result, _fh)
'''


def write_python_bootstrap(kind: str, config: dict, tmpdir: str) -> str:
    """Render a self-contained Python profiler bootstrap script into *tmpdir*.

    The bootstrap runs ``config["target"]`` (a script path, or a module name
    when ``config["is_module"]`` is true) via :mod:`runpy`, profiles it with
    stdlib-only instrumentation, and writes a JSON result to
    ``config["result_path"]`` from a ``finally`` block -- so the result
    survives the target calling ``sys.exit`` or raising.

    Args:
        kind: ``"tracemalloc"`` or ``"trace"``.
        config: Bootstrap configuration. Keys: ``"target"``, ``"is_module"``
            (bool), ``"args"`` (list of str), ``"result_path"`` (absolute
            path); ``kind="trace"`` additionally reads ``"focus_file"`` (str,
            may be empty -- when set, per-line hit counts are collected only
            for frames whose filename matches it).
        tmpdir: Directory the bootstrap script is written into.

    Returns:
        The absolute path to the written bootstrap script.

    Raises:
        ValueError: If *kind* is not ``"tracemalloc"`` or ``"trace"``.
    """
    if kind == "tracemalloc":
        template = _TRACEMALLOC_BOOTSTRAP_TEMPLATE
    elif kind == "trace":
        template = _TRACE_BOOTSTRAP_TEMPLATE
    else:
        raise ValueError(f"Unknown bootstrap kind: {kind!r}")

    config_literal = repr(json.dumps(config))
    source = template.replace("@@CONFIG_JSON@@", config_literal)

    bootstrap_path = os.path.join(
        os.path.abspath(tmpdir), f"_coding_agent_{kind}_bootstrap.py"
    )
    with open(bootstrap_path, "w", encoding="utf-8") as fh:
        fh.write(source)

    return bootstrap_path
