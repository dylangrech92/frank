from __future__ import annotations

import json
import os
import re
import shlex
import tempfile

from runtime.process import run_one_shot


_STATUS_MAP = {
    "passed": "passed",
    "failed": "failed",
    "skipped": "skipped",
    "error": "error",
    "xfailed": "skipped",
    "xpassed": "passed",
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
