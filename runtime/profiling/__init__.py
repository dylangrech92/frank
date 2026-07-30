"""Profiling subpackage.

This package consolidates the runtime profiling module. Consumer tools import
from this package directly:
- Subprocess runners: :func:`run_measured`, :func:`which_interpreter`,
  :func:`php_xdebug_status`, :class:`MeasuredRun`.
- Parser functions: :func:`parse_cpuprofile`, :func:`parse_heapprofile`,
  :func:`parse_cachegrind`.
- Formatting helpers: :func:`fmt_bytes`, :func:`fmt_seconds`,
  :func:`fmt_count`, :func:`render_top_table`.
- Bootstrap generation: :func:`write_python_bootstrap`.

All public names are re-exported at package level so consumers can import:

    >>> from runtime.profiling import MeasuredRun, run_measured
"""

from __future__ import annotations

from .bootstrap import write_python_bootstrap
from .cachegrind import parse_cachegrind
from .format import fmt_bytes, fmt_count, fmt_seconds, render_top_table
from .parsers import parse_cpuprofile, parse_heapprofile
from .run import (
    MeasuredRun,
    php_xdebug_status,
    run_measured,
    which_interpreter,
)

__all__ = [
    "MeasuredRun",
    "run_measured",
    "which_interpreter",
    "php_xdebug_status",
    "parse_cpuprofile",
    "parse_heapprofile",
    "parse_cachegrind",
    "fmt_bytes",
    "fmt_seconds",
    "fmt_count",
    "render_top_table",
    "write_python_bootstrap",
]
