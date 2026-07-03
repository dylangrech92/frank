"""Test framework detection and structured test running for the coding agent."""

from __future__ import annotations

import json
import os
import re
import shlex
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from runtime.process import run_one_shot

FRAMEWORKS = ("pytest", "jest", "phpunit")

_STATUS_MAP = {
    "passed": "passed",
    "failed": "failed",
    "skipped": "skipped",
    "error": "error",
    "xfailed": "skipped",
    "xpassed": "passed",
}


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


def run_pytest(
    target_dir: str,
    project_root: str,
    pattern: str | None = None,
    timeout: int = 120,
) -> dict:
    """Run pytest and return structured results.

    Primary path uses ``--json-report``; falls back to a degraded *"-v -rA"*
    parse when the plugin is unavailable.

    Args:
        target_dir: Directory passed to ``pytest`` as the test root.
        project_root: Working directory for the subprocess.
        pattern: Optional ``-k`` string (e.g. ``"not slow"``).
        timeout: Max seconds before the command is killed. Defaults to 120.

    Returns:
        A dict with keys ``framework``, ``degraded``, ``tests``, ``summary``,
        ``output_tail``, and ``note`` as described in the module docstring.
    """

    # Build base command with shlex-quoted target.
    target = shlex.quote(target_dir) if isinstance(target_dir, str) else target_dir
    cmd_base = f"python3 -m pytest {target} -q"

    try:
        # --------------------------------------------------- Primary path: --json-report
        tmp_handle, tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(tmp_handle)

        report_cmd = f"{cmd_base} --json-report --json-report-file={tmp_path}"
        if pattern is not None:
            report_cmd += f" -k {shlex.quote(pattern)}"

        result = run_one_shot(report_cmd, project_root, timeout_seconds=timeout)
        stdout_part = str(result.get("stdout", ""))
        stderr_part = str(result.get("stderr", ""))
        combined_output = stdout_part + "\n" + stderr_part
        timed_out = bool(result.get("timed_out"))

        # ----- timed out (primary run) --------------------------------------
        if timed_out:
            return dict(
                framework="pytest",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=f"TIMED OUT after {timeout}s\n{combined_output.strip()[-2000:]}",
                note="",
            )

        # ----- try to open & parse the report -------------------------------
        content = ""
        try:
            with open(tmp_path, encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            pass

        if content.strip():
            try:
                report = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                report = {}
        else:
            report = {}

        os.unlink(tmp_path)

        # ----- successfully parsed — primary path ---------------------------
        if "tests" in report and report["tests"]:
            tests_list = []
            parsed_statuses = {
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
            }

            for entry in report["tests"]:
                name = entry.get("nodeid", "")
                outcome_raw = entry.get("outcome", "error")
                status = _STATUS_MAP.get(outcome_raw, outcome_raw)

                # Duration: prefer the top-level field; fall back through phases.
                dur = float(entry.get("duration", 0.0))
                if dur == 0.0:
                    for phase_name in ("setup", "call", "teardown"):
                        phase_val = entry.get(phase_name, {})
                        if isinstance(phase_val, dict):
                            phase_dur = phase_val.get("duration")
                            if phase_dur is not None:
                                dur += float(phase_dur)

                message = ""
                if status != "passed":
                    message = (
                        str(entry.get("call", {}).get("longrepr", ""))
                        or str(entry.get("setup", {}).get("longrepr", ""))
                        or ""
                    )

                tests_list.append({
                    "name": name,
                    "status": status,
                    "message": message,
                    "duration": dur,
                })

                if status in ("passed",):
                    parsed_statuses["passed"] += 1
                elif status in ("failed",):
                    parsed_statuses["failed"] += 1
                elif status in ("skipped",):
                    parsed_statuses["skipped"] += 1
                elif status == "error":
                    parsed_statuses["errors"] += 1

            parsed_statuses["total"] = len(tests_list)

            return dict(
                framework="pytest",
                degraded=False,
                tests=tests_list,
                summary=parsed_statuses,
                output_tail=combined_output[-2000:],
                note="",
            )

        # ----- report is missing / empty / unparseable --
        # "unrecognized arguments: --json-report" also triggers degraded.
        unrecognized = "unrecognized arguments: --json-report" in combined_output
        if not unrecognized and os.path.isfile(tmp_path):
            try:
                with open(tmp_path, encoding="utf-8") as fh:
                    check_content = fh.read()
                if check_content.strip():
                    try:
                        json.loads(check_content)
                    except (json.JSONDecodeError, ValueError):
                        unrecognized = True
            except OSError:
                unrecognized = True

        # Degraded fallback
        degraded_cmd = f"python3 -m pytest {target} -v -rA"
        if pattern is not None:
            degraded_cmd += f" -k {shlex.quote(pattern)}"

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        degraded_result = run_one_shot(degraded_cmd, project_root, timeout_seconds=timeout)
        fb_stdout = str(degraded_result.get("stdout", ""))
        fb_stderr = str(degraded_result.get("stderr", ""))
        fb_output = fb_stdout + "\n" + fb_stderr
        combined_output = fb_output
        degraded_timed_out = bool(degraded_result.get("timed_out"))

        if degraded_timed_out:
            return dict(
                framework="pytest",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=f"TIMED OUT after {timeout}s\n{fb_output.strip()[-2000:]}",
                note="degraded pytest parse (pytest-json-report plugin unavailable) -- per-test durations unavailable",
            )

        # Parse verbose lines.
        parsed_statuses = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        }
        tests_list = []
        test_names_seen: list[str] = []

        # The main verbose output lines pattern.
        _verbose_line_re = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")

        # Short test summary section lines for failed/error messages.
        _summary_line_re = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.*))?$")

        in_summary_section = False
        summary_messages: dict[str, str] = {}

        # pytest may emit ANSI color codes even when piped; strip them so the
        # line regexes see plain text.
        plain_output = re.sub(r"\x1b\[[0-9;]*m", "", fb_output)

        for line in plain_output.splitlines():
            if "short test summary info" in line.lower() or "===== " + "=" * 10 in line:
                in_summary_section = True
                continue

            if re.match(r"^={3,}", line):
                in_summary_section = False
                continue

            if in_summary_section:
                m = _summary_line_re.match(line)
                if m:
                    s_name = m.group(1)
                    s_msg = m.group(2) or ""
                    summary_messages[s_name] = s_msg
                continue

            vm = _verbose_line_re.match(line)
            if vm:
                test_name = vm.group(1)
                raw_status = vm.group(2).lower()
                if raw_status == "xfail":
                    raw_status = "skipped"
                elif raw_status == "xpass":
                    raw_status = "passed"

                tests_list.append({
                    "name": test_name,
                    "status": raw_status,
                    "message": "",
                    "duration": 0.0,
                })
                test_names_seen.append(test_name)

                if raw_status == "passed":
                    parsed_statuses["passed"] += 1
                elif raw_status == "failed":
                    parsed_statuses["failed"] += 1
                elif raw_status == "skipped":
                    parsed_statuses["skipped"] += 1
                elif raw_status == "error":
                    parsed_statuses["errors"] += 1

        # Attach per-test messages from the summary section.
        for test_name in test_names_seen:
            short = test_name.rsplit("::", 1)[-1] if "::" in test_name else test_name
            s_msg = summary_messages.get(test_name) or summary_messages.get(short)
            if s_msg:
                for t in tests_list:
                    if t["name"] == test_name:
                        t["message"] = s_msg
                        break

        parsed_statuses["total"] = len(tests_list)

        return dict(
            framework="pytest",
            degraded=True,
            tests=tests_list,
            summary=parsed_statuses,
            output_tail=fb_output[-2000:],
            note="degraded pytest parse (pytest-json-report plugin unavailable) -- per-test durations unavailable",
        )

    except Exception as exc:
        # Safety net — never let an exception escape.
        return dict(
            framework="pytest",
            degraded=True,
            tests=[],
            summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
            output_tail=f"internal error: {exc}",
            note="internal error",
        )


def run_jest(
    target_dir: str,
    project_root: str,
    pattern: str | None = None,
    timeout: int = 120,
) -> dict:
    """Run jest with ``--json`` and return structured results.

    The parser reads the jest ``--json`` schema; vitest claims output
    compatibility with this schema but that is config-only and deliberately
    unvalidated here.

    Args:
        target_dir: Directory passed to ``jest`` as the test root (also used
            as cwd so local ``jest.config.*`` is picked up).
        project_root: Project root directory.
        pattern: Optional ``-t`` string for filtering test names.
        timeout: Max seconds before the command is killed. Defaults to 120.

    Returns:
        A dict with keys ``framework``, ``degraded``, ``tests``, ``summary``,
        ``output_tail``, and ``note`` as described in the module docstring.
    """

    try:
        parts = ["jest", "--json"]
        if pattern is not None:
            parts.append(f"-t {shlex.quote(pattern)}")
        cmd = " ".join(parts)

        result = run_one_shot(cmd, target_dir, timeout_seconds=timeout)
        stdout_part = str(result.get("stdout", ""))
        stderr_part = str(result.get("stderr", ""))
        combined_output = stdout_part + "\n" + stderr_part
        timed_out = bool(result.get("timed_out"))

        if timed_out:
            return dict(
                framework="jest",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=f"TIMED OUT after {timeout}s\n{combined_output.strip()[-2000:]}",
                note=f"TIMED OUT after {timeout}s",
            )

        # Parse the JSON document from stdout.
        if not stdout_part.strip():
            return dict(
                framework="jest",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=combined_output[-2000:],
                note="jest produced no parseable --json output (is jest installed?)",
            )

        try:
            data = json.loads(stdout_part)
        except (json.JSONDecodeError, ValueError):
            return dict(
                framework="jest",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=combined_output[-2000:],
                note="jest produced no parseable --json output (is jest installed?)",
            )

        status_map = {
            "passed": "passed",
            "failed": "failed",
            "pending": "skipped",
            "todo": "skipped",
            "disabled": "skipped",
        }

        tests_list: list[dict] = []
        parsed_statuses = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        for suite in data.get("testResults", []):
            suite_path = suite.get("name") or suite.get("filePath") or ""
            if not suite_path:
                continue
            relative = os.path.relpath(suite_path, project_root)

            for entry in suite.get("assertionResults", []):
                full_name = f"{relative}::{entry.get('fullName', '')}"
                raw_status = entry.get("status", "unknown")
                status = status_map.get(raw_status, "error")
                message = "\n".join(entry.get("failureMessages", []))
                duration = float(entry.get("duration") or 0.0) / 1000.0

                tests_list.append({
                    "name": full_name,
                    "status": status,
                    "message": message,
                    "duration": duration,
                })

                if status == "passed":
                    parsed_statuses["passed"] += 1
                elif status == "failed":
                    parsed_statuses["failed"] += 1
                elif status == "skipped":
                    parsed_statuses["skipped"] += 1
                elif status == "error":
                    parsed_statuses["errors"] += 1

        parsed_statuses["total"] = (
            data.get("numTotalTests", 0)
            if data.get("numTotalTests", 0) > 0
            else len(tests_list)
        )

        return dict(
            framework="jest",
            degraded=False,
            tests=tests_list,
            summary=parsed_statuses,
            output_tail=combined_output[-2000:],
            note="",
        )

    except Exception as exc:
        return dict(
            framework="jest",
            degraded=True,
            tests=[],
            summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
            output_tail=f"internal error: {exc}",
            note="internal error",
        )


def run_phpunit(
    target_dir: str,
    project_root: str,
    pattern: str | None = None,
    timeout: int = 120,
) -> dict:
    """Run phpunit with ``--log-junit`` and return structured results.

    Args:
        target_dir: Directory passed to ``phpunit`` as the test root (also used
            as cwd so local ``phpunit.xml`` is discovered).
        project_root: Project root directory (not directly used by phpunit
            but retained for API consistency).
        pattern: Optional ``--filter`` string for test name filtering.
        timeout: Max seconds before the command is killed. Defaults to 120.

    Returns:
        A dict with keys ``framework``, ``degraded``, ``tests``, ``summary``,
        ``output_tail``, and ``note`` as described in the module docstring.
    """

    tmp_handle: int | None = None
    report_path: str | None = None

    try:
        tmp_handle, report_path = tempfile.mkstemp(suffix=".xml")
        os.close(tmp_handle)
        tmp_handle = None

        parts = ["phpunit", "--log-junit", shlex.quote(report_path)]
        if pattern is not None:
            parts.append(f"--filter {shlex.quote(pattern)}")
        cmd = " ".join(parts)

        result = run_one_shot(cmd, target_dir, timeout_seconds=timeout)
        stdout_part = str(result.get("stdout", ""))
        stderr_part = str(result.get("stderr", ""))
        combined_output = stdout_part + "\n" + stderr_part
        timed_out = bool(result.get("timed_out"))

        if timed_out:
            return dict(
                framework="phpunit",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=f"TIMED OUT after {timeout}s\n{combined_output.strip()[-2000:]}",
                note="",
            )

        # Read and parse the JUnit XML report.
        if not os.path.isfile(report_path):
            return dict(
                framework="phpunit",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=combined_output[-2000:],
                note="phpunit produced no JUnit XML (is phpunit installed?)",
            )

        try:
            with open(report_path, encoding="utf-8") as fh:
                file_content = fh.read()
        except OSError:
            return dict(
                framework="phpunit",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=combined_output[-2000:],
                note="phpunit produced no JUnit XML (is phpunit installed?)",
            )

        if not file_content.strip():
            return dict(
                framework="phpunit",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=combined_output[-2000:],
                note="phpunit produced no JUnit XML (is phpunit installed?)",
            )

        try:
            tree = ET.parse(report_path)
        except ET.ParseError:
            return dict(
                framework="phpunit",
                degraded=True,
                tests=[],
                summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
                output_tail=combined_output[-2000:],
                note="phpunit produced no JUnit XML (is phpunit installed?)",
            )

        root = tree.getroot()

        tests_list: list[dict] = []
        parsed_statuses = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

        for tc in root.iter("testcase"):
            classname = tc.get("classname", "") or ""
            name = tc.get("name", "") or ""
            if classname:
                full_name = f"{classname}::{name}"
            else:
                full_name = name

            duration_str = tc.get("time", "0") or "0"
            duration = float(duration_str)

            failure_elem = tc.find("failure")
            error_elem = tc.find("error")
            skipped_elem = tc.find("skipped")

            if failure_elem is not None:
                status = "failed"
                message = (failure_elem.text or "").strip()
            elif error_elem is not None:
                status = "error"
                message = (error_elem.text or "").strip()
            elif skipped_elem is not None:
                status = "skipped"
                message = ""
            else:
                status = "passed"
                message = ""

            tests_list.append({
                "name": full_name,
                "status": status,
                "message": message,
                "duration": duration,
            })

            if status == "passed":
                parsed_statuses["passed"] += 1
            elif status == "failed":
                parsed_statuses["failed"] += 1
            elif status == "skipped":
                parsed_statuses["skipped"] += 1
            elif status == "error":
                parsed_statuses["errors"] += 1

        parsed_statuses["total"] = len(tests_list)

        return dict(
            framework="phpunit",
            degraded=False,
            tests=tests_list,
            summary=parsed_statuses,
            output_tail=combined_output[-2000:],
            note="",
        )

    except Exception as exc:
        return dict(
            framework="phpunit",
            degraded=True,
            tests=[],
            summary={"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0},
            output_tail=f"internal error: {exc}",
            note="internal error",
        )

    finally:
        if report_path is not None:
            try:
                os.unlink(report_path)
            except OSError:
                pass


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
