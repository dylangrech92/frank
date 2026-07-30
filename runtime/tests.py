"""Test framework detection and structured test running for the coding agent."""

from __future__ import annotations

import os
import re
from pathlib import Path

from runtime.frameworks.jest_runner import run_jest
from runtime.frameworks.phpunit_runner import run_phpunit
from runtime.frameworks.pytest_runner import run_pytest

FRAMEWORKS = ("pytest", "jest", "phpunit")


def detect_framework(
    target_dir: str,
    project_root: str,
    test_runners: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    """Return *(framework, reason)*.

    *framework* is one of ``FRAMEWORKS`` or ``None`` when nothing is detected.

    Detection priority (first match wins):

    1. Explicit config via *test_runners*.
    2. Marker files / directories walking upward from *target_dir*.
    3. Unknown → ``(None, "no test framework markers found")``.

    Args:
        target_dir: Directory to inspect for a test framework.
        project_root: Project root directory; the walk stops after checking this
            level (never above it).
        test_runners: Optional mapping of project-root-relative directory strings
            to framework names (e.g. ``{"js": "jest", ".": "pytest"}``).

    Returns:
        A ``(framework, reason)`` tuple as described above.
    """

    # ---- 1. Explicit config first ----
    if test_runners is not None and test_runners:
        rel = os.path.relpath(target_dir, project_root)
        walked = []
        while True:
            walked.append("." if rel in ("", os.sep) else rel)
            if rel in ("", ".", os.sep):
                break
            rel = os.path.dirname(rel)

        for key in walked:
            value = test_runners.get(key)
            if value is None:
                continue
            if value not in FRAMEWORKS:
                return (None, f"config test_runners maps to unknown framework '{value}'")
            return (value, f"configured in test_runners for '{key}'")

    # ---- 2. Marker files ----
    current = os.path.abspath(target_dir)
    project_abs = os.path.abspath(project_root)

    while True:
        # PHPUnit — most specific
        for name in ("phpunit.xml", "phpunit.xml.dist"):
            if os.path.isfile(os.path.join(current, name)):
                return ("phpunit", f"found {name} in {current}")

        # Jest configs
        jest_files = (
            "jest.config.js",
            "jest.config.cjs",
            "jest.config.mjs",
        )
        for jf in jest_files:
            path_val = os.path.join(current, jf)
            if os.path.isfile(path_val):
                return ("jest", f"found {jf} in {current}")

        # package.json with "jest" dependency
        pkg_json = os.path.join(current, "package.json")
        if os.path.isfile(pkg_json):
            try:
                with open(pkg_json, encoding="utf-8") as fh:
                    text_content = fh.read()
            except OSError:
                pass
            else:
                if '"jest"' in text_content:
                    return ("jest", f"found \"jest\" dependency in {pkg_json}")

        # Pytest marker files
        pytest_marker_files = (
            "pytest.ini",
            "pyproject.toml",
            "setup.cfg",
            "conftest.py",
        )
        for pmf in pytest_marker_files:
            if os.path.isfile(os.path.join(current, pmf)):
                return ("pytest", f"found {pmf} in {current}")

        # tests/ subdirectory with test_*.py files
        tests_dir = os.path.join(current, "tests")
        if os.path.isdir(tests_dir):
            py_files = [
                entry.name
                for entry in Path(tests_dir).iterdir()
                if entry.is_file() and re.search(r"^test_.*\.py$", entry.name)
            ]
            if py_files:
                return ("pytest", f"found test file {py_files[0]} in tests/ directory")

        # Move up; stop when current >= project_abs (checked via parent)
        parent_dir = os.path.dirname(current)
        if parent_dir == current or current == project_abs:
            break
        current = parent_dir

    return (None, "no test framework markers found")


def _make_zeroed() -> dict:
    """Return a zero-filled pytest result dict."""
    return {
        "framework": "pytest",
        "degraded": True,
        "tests": [],
        "summary": {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
        "output_tail": "",
        "note": "internal error",
    }


def run_framework(
    framework: str,
    target_dir: str,
    project_root: str,
    pattern: str | None = None,
    timeout: int = 120,
) -> dict:
    """Dispatch to the runner for *framework* (one of FRAMEWORKS)."""

    try:
        if framework == "pytest":
            return run_pytest(target_dir, project_root, pattern, timeout)
        elif framework == "jest":
            return run_jest(target_dir, project_root, pattern, timeout)
        elif framework == "phpunit":
            return run_phpunit(target_dir, project_root, pattern, timeout)
        else:
            return dict(
                framework=framework,
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail="",
                note=f"unknown framework '{framework}'",
            )
    except Exception as exc:
        return dict(
            framework=framework,
            degraded=True,
            tests=[],
            summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
            output_tail=f"internal error: {exc}",
            note="internal error",
        )
