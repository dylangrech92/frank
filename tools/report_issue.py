"""Report a failure of the agent's own tools or of the harness itself.

The model is the only observer present at the moment a tool lies -- an edit that
reports success without writing, a search that misses a symbol already seen, a
message naming a tool the active mode does not carry. This tool lets it write
that down instead of the failure surviving only in a transcript nobody re-reads.

Entries are appended to ``issues.md`` beside this package (the coding-agent
install dir), NOT in the target project's cwd, because a single agent run
operates on arbitrary projects and a harness defect is a property of the
harness, not of whichever codebase it happened to be pointed at -- the same
reasoning ``stats.py`` gives for ``stats.json``. Parallel subagent runs share
this one file, so the append is guarded by an exclusive ``flock``.
"""

from __future__ import annotations

import fcntl
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.base import Tool
from tools.result import ToolResult

# tools/ -> install dir. Read through the module global at call time (never
# bound as a default argument) so the contract eval can rebind it to a temp
# file instead of writing into the real log.
ISSUES_PATH = Path(__file__).resolve().parent.parent / "issues.md"

# A log entry has a job; a model's stack-trace dump does not need to be kept
# whole to do it.
MAX_ISSUE_CHARS = 8000

# Characters that change how a plain YAML scalar parses when they lead a value.
_YAML_INDICATORS = "-?:,[]{}#&*!|>'\"%@`"


def _yaml_quoted(value: str) -> str:
    """Render *value* as a double-quoted one-line YAML scalar.

    Used for the harness-stamped context fields, which are identifiers and
    paths rather than prose: a model id like ``ornith:35b`` or a session id
    like ``2026-07-31T14-01-55-88213`` has no readable plain form worth the
    per-value reasoning about how YAML would resolve it. JSON string syntax is
    a valid YAML double-quoted scalar, so ``json.dumps`` is a correct escaper
    here rather than an approximation of one.

    ``ensure_ascii=False`` is required, not cosmetic: the default escapes
    astral characters as a surrogate pair (``\\ud83d\\ude80``), and YAML
    decodes the two escapes separately instead of recombining them, so an
    emoji would come back out of the log as broken surrogates. The file is
    written as UTF-8, so the raw characters are safe to emit.
    """
    return json.dumps(value, ensure_ascii=False)


def _needs_quoting(text: str) -> bool:
    """True when *text* cannot be written as a plain one-line YAML scalar.

    Deliberately conservative -- over-quoting costs two characters, while
    under-quoting either corrupts the entry or silently changes its type. Each
    rule closes a case observed to break a real round-trip:

    * a colon anywhere can open a mapping (``replace_one: not found`` is a
      realistic report and a YAML syntax error);
    * a tab or control character cannot start a plain scalar at all;
    * a single bare token with no space is what every typed YAML scalar looks
      like -- ``true``, ``123``, ``null``, ``0x1f``, ``2026-07-31`` all load as
      something other than a string, and a whole report that is one token is
      not a report worth keeping plain.
    """
    if not text or text != text.strip():
        return True
    if not text.isprintable():  # tabs, newlines, control characters
        return True
    if text[0] in _YAML_INDICATORS:
        return True
    if ":" in text or " #" in text:
        return True
    return " " not in text


def _yaml_issue(text: str) -> str:
    """Render the issue text as the entry's ``issue:`` field.

    Single-line text stays on one line -- plain when it parses cleanly, quoted
    when it does not -- so the common entry reads exactly as specified.
    Multi-line text becomes a literal block scalar, which also neutralises the
    one input that could otherwise corrupt the log: a line of ``---`` inside
    the model's own text, which lands indented inside the block instead of
    splitting the entry.

    The explicit ``|2`` indentation indicator is load-bearing. Without it YAML
    infers the block's indentation from its first non-empty line, so text whose
    first line is itself indented sets a baseline the later, less-indented
    lines fall out of, and loading the entry raises a parser error instead of
    returning the report. Stating the indentation removes the dependency on
    what the caller happened to pass.
    """
    if "\n" not in text:
        return f"issue: {_yaml_quoted(text) if _needs_quoting(text) else text}"
    body = "\n".join(f"  {line}" if line.strip() else "" for line in text.split("\n"))
    return f"issue: |2\n{body}"


def _harness_context() -> dict[str, str]:
    """Collect what the harness already knows about this run.

    Every field is optional and omitted entirely when unavailable -- never
    written as ``unknown`` -- so a field that IS present in the log can be
    trusted, and its absence is itself the signal that context was not
    reachable.  Collection failures are contained for the same reason: losing a
    defect report because optional decoration could not be gathered would be
    strictly worse than filing it undecorated.
    """
    context: dict[str, str] = {}

    from tools.registry import current_mode  # pylint: disable=import-outside-toplevel

    mode = current_mode()
    if mode:
        context["mode"] = mode

    # Imported lazily and by name -- main imports this package, so a top-level
    # import would be circular.  Same seam the LSP tools use to reach MANAGER.
    try:
        import main as main_module  # pylint: disable=import-outside-toplevel

        session = getattr(main_module, "SESSION", None)
        if session is not None:
            context["project"] = str(session.project_root)
            context["model"] = str(session.model)
            context["session"] = str(session.session_id)
    except (ImportError, AttributeError):
        pass

    return context


def _build_entry(issue: str, now: datetime, context: dict[str, str]) -> str:
    """Assemble one ``---``-delimited log entry, ending in a newline."""
    lines = ["---", f"date: {now.strftime('%Y-%m-%dT%H:%M:%S')}"]
    for key in ("mode", "project", "model", "session"):
        value = context.get(key)
        if value:
            lines.append(f"{key}: {_yaml_quoted(value)}")
    lines.append(_yaml_issue(issue))
    lines.append("---")
    return "\n".join(lines) + "\n"


def _append_entry(entry: str, path: Path) -> None:
    """Append *entry* to *path* under an exclusive lock.

    ``"a"`` creates the file when absent and never truncates; the lock keeps
    concurrent agent processes from interleaving partial entries into each
    other's blocks.
    """
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(entry)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class ReportIssue(Tool):
    """Record a tool or harness failure to the maintainer's issue log."""

    name = 'report_issue'
    description = (
        'Report a failure of your own tools or of this harness. Use it when a tool '
        'result contradicts reality — an edit reports success but the file is '
        'unchanged, a search returns nothing for something you have already seen, a '
        'tool errors in a way its own message does not explain — or when the harness '
        'itself obstructs you: an instruction contradicts another, a message names a '
        'tool you do not have, a guard blocks work that was legitimate. Describe what '
        'you were doing and what went wrong. The report goes to the harness '
        'maintainer; it does not fix the problem or reach the user, so carry on with '
        'your task afterwards.'
    )
    action = 'record the issue'
    oversize_hint = 'describe the failure in fewer words'
    alternative = 'stating the problem in your answer instead'

    # Writes a file, so it must never ride the concurrent dispatch path: that
    # path skips mutation attribution, pre-image capture and the loop guard on
    # the documented invariant that parallel_safe tools never mutate (agent.py).
    parallel_safe = False

    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'issue': {
                'type': 'string',
                'description': (
                    'What failed and what you were doing at the time. Name the tool '
                    'or instruction involved, what you expected, and what actually '
                    'happened.'
                ),
            },
        },
        'required': ['issue'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Append one issue report to the install-dir ``issues.md``.

        Args:
            **kwargs: Parsed from the LLM function-call payload. Requires
                ``issue``, a non-empty string.

        Returns:
            A ``ToolResult`` naming the log the entry landed in, or an error
            when *issue* is missing or blank.
        """
        raw = kwargs.get('issue')
        issue = raw.strip() if isinstance(raw, str) else ''
        if not issue:
            return ToolResult.err(
                'The "issue" argument is required and must be a non-empty string '
                'describing what failed.',
                code='missing-argument',
            )

        truncated = len(issue) > MAX_ISSUE_CHARS
        if truncated:
            issue = issue[:MAX_ISSUE_CHARS].rstrip() + '\n… report truncated.'

        entry = _build_entry(issue, datetime.now(), _harness_context())

        # Deliberately no emit_mutation: the log lives outside the project tree,
        # and publishing it on the mutation bus would list a file the user never
        # asked to change and arm the verification gate for a report that
        # changed no code.
        path = ISSUES_PATH
        try:
            _append_entry(entry, path)
        except OSError as exc:
            return ToolResult.err(
                f'Could not write the issue log at {path}: {exc}',
                code='issue-log-unwritable',
            )

        return ToolResult.ok(
            f'Issue recorded in {path.name}. This does not fix the problem — '
            f'continue with your task.',
            path=str(path),
            truncated=truncated,
        )
