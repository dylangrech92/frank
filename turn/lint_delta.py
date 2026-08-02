from __future__ import annotations


from llm import ToolCall



# Write tools that mutate exactly the file named by their own `path` argument
# (unlike `move_file`, whose mutation event fires on the *destination* path
# while `path` names the *source* -- a move changes a file's location, never
# its content, so it can never introduce a lint issue and is deliberately
# excluded here). Limiting reactive lint-delta injection to this single-path
# set keeps the pre-edit snapshot cheap and exact: one lint call on one known
# path before dispatch, one after.
_LINT_TRACKED_TOOLS = frozenset({"format", "write_file", "edit_file"})

# Cap on how many new lint issues are appended per call, mirroring the
# `lint` tool's own `_MAX_RENDERED_ISSUES` guard against flooding context.
_LINT_DELTA_CAP = 20


def _lint_resolve_call_path(call: ToolCall, project_root: str) -> str | None:
    """Resolve *call*'s ``path`` argument to the absolute string mutation events use.

    Tools resolve their ``path`` argument via ``resolve_in_root(Path.cwd(), path)``
    and emit mutation events keyed by that resolved absolute path (see
    ``tools/_sandbox.py``) -- not by the raw, project-root-relative argument the
    model passed. This mirrors that same resolution so the argument can be
    matched against ``_scan_new_mutations``'s path set.

    Returns ``None`` on any resolution failure (escapes the root, wrong type,
    etc.) -- callers treat that identically to "nothing to snapshot".
    """
    path_val = call.arguments.get("path")
    if not isinstance(path_val, str) or not path_val:
        return None
    try:
        from tools._sandbox import resolve_in_root

        return str(resolve_in_root(project_root, path_val))
    except Exception:
        return None


def _lint_pre_snapshot(call: ToolCall, project_root: str):
    """Capture the pre-edit lint issues for *call*'s target file, if lintable.

    Returns ``None`` when the call isn't one of ``_LINT_TRACKED_TOOLS``, its
    ``path`` argument is missing/not a string/escapes the root, or its
    extension has no configured/available linter -- in every such case
    reactive lint-delta injection silently does nothing for this call.

    Args:
        call: The about-to-be-dispatched tool call.
        project_root: Absolute project root, as required by ``run_lint``.

    Returns:
        A list of ``LintIssue`` (the pre-edit snapshot for the path), or
        ``None`` when no snapshot could or should be taken.
    """
    if call.name not in _LINT_TRACKED_TOOLS:
        return None

    resolved_path = _lint_resolve_call_path(call, project_root)
    if resolved_path is None:
        return None

    try:
        from tools._lint import EXTENSION_LANGUAGE, run_lint
        from pathlib import Path as _Path

        if _Path(resolved_path).suffix.lower() not in EXTENSION_LANGUAGE:
            return None

        report = run_lint([resolved_path], project_root)
        if report.unavailable:
            return None  # no configured/available linter for this language
        return report.issues
    except Exception:
        return None


def _lint_delta_suffix(pre_issues, call: ToolCall, project_root: str) -> str:
    """Return an appended ``[lint] ...`` block for issues new since *pre_issues*.

    Re-lints the same path lint-snapshotted before dispatch and diffs against
    *pre_issues* by identity (path, line, col, rule, message). Only issues
    absent from the pre-edit snapshot are surfaced -- pre-existing project
    lint noise is never injected (the whole point of a delta, not a full
    report). Capped at ``_LINT_DELTA_CAP`` lines plus a "+N more" tail.

    Args:
        pre_issues: The pre-edit issue list returned by ``_lint_pre_snapshot``
            (never ``None`` when this is called).
        call: The tool call that was just dispatched (successfully, and
            confirmed via ``_scan_new_mutations`` to have mutated this path).
        project_root: Absolute project root, as required by ``run_lint``.

    Returns:
        An empty string when there is nothing new to report (including on any
        internal error), or a ``"\\n\\n[lint] ..."`` block ready to append to
        the rendered tool result.
    """
    resolved_path = _lint_resolve_call_path(call, project_root)
    if resolved_path is None:
        return ""

    try:
        from tools._lint import run_lint

        report = run_lint([resolved_path], project_root)
        if report.unavailable:
            return ""

        pre_keys = {(i.path, i.line, i.col, i.rule, i.message) for i in pre_issues}
        new_issues = [
            i for i in report.issues
            if (i.path, i.line, i.col, i.rule, i.message) not in pre_keys
        ]
        if not new_issues:
            return ""

        shown = new_issues[:_LINT_DELTA_CAP]
        lines = [f"[lint] {i.path}:{i.line} {i.rule} {i.message}" for i in shown]
        remaining = len(new_issues) - len(shown)
        if remaining > 0:
            lines.append(f"[lint] +{remaining} more")
        return "\n\n" + "\n".join(lines)
    except Exception:
        return ""
