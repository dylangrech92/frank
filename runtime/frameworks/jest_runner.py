from __future__ import annotations

import json
import os
import shlex

from runtime.process import run_one_shot


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
