from __future__ import annotations

import os
import shlex
import tempfile
import xml.etree.ElementTree as ET

from runtime.process import run_one_shot


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
