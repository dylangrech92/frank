"""Dispatch-level contract check for the report_issue tool (no LLM).

Drives the real production hot path -- ``tools.registry.dispatch`` -- exactly as
a live model turn does per tool call, so this exercises the mode gate, argument
validation and ``ReportIssue.run`` together against a real file on disk. Zero
mocks: entries are appended to a real temp log with the real ``flock`` write
path, and read back with a real YAML parser. Asserts:

a. report_issue is reachable in every mode (a failure can happen in any of
   them, so the channel for reporting it has to exist in all of them) — the
   check iterates ``modes.MODES``, so a newly added mode is covered the moment
   it is declared, without this file being touched;
b. happy path: the log is created when absent and the entry parses as YAML;
c. a second call appends -- the first entry survives byte-for-byte;
d. multi-line text becomes an indented ``|2`` block scalar, and a line of
   ``---`` inside the model's own text does not split the log into extra
   entries;
e. single-line text containing ``: `` round-trips instead of producing
   ambiguous YAML;
f. the ``mode`` field in the written entry matches the activated mode;
g. filing a report emits ZERO mutation events -- the check that a report never
   pollutes ``files_changed`` or arms the verification gate;
h. blank / missing / non-string ``issue`` is rejected with a kebab-case code;
i. oversize text is capped and flagged ``truncated``;
j. the tool is not parallel_safe and joins no agent.py name-set (so the
   repeat-call cap still applies to byte-identical duplicate reports).

Exits 0 on success, prints each ``FAIL: <reason>`` to stderr and exits 1
otherwise. Runs with the repo root on ``sys.path`` (evals/run.py inserts it;
running directly needs PYTHONPATH=<repo root>) and restores both the active
mode and the real log path it rebound.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

LOG: Path = Path()  # rebound to the temp log in main()


def _dispatch(issue: object):
    """Call report_issue through the real dispatcher, ``issue`` unvalidated."""
    from tools.registry import dispatch

    args: dict[str, object] = {} if issue is None else {'issue': issue}
    return dispatch('report_issue', args)


def _entries() -> list[dict]:
    """Parse the temp log into one dict per entry.

    Each entry is delimited by ``---`` above and below, so consecutive entries
    put two ``---`` lines back to back and YAML sees an empty document between
    them; dropping the ``None`` documents leaves exactly the real entries.
    """
    import yaml

    text = LOG.read_text(encoding='utf-8')
    return [doc for doc in yaml.safe_load_all(text) if doc is not None]


# ---------------------------------------------------------------------------
# a. reachable in every mode
# ---------------------------------------------------------------------------


def check_all_modes() -> list[str]:
    import modes
    from tools import registry

    failures: list[str] = []
    for name in modes.MODES:
        registry.activate_mode(name)
        listed = [s['function']['name'] for s in registry.schemas()]
        if 'report_issue' not in listed:
            failures.append(f"mode {name!r}: report_issue missing from schemas(): {listed}")
    return failures


# ---------------------------------------------------------------------------
# b/c/f. happy path, append, mode stamp
# ---------------------------------------------------------------------------


def check_append_and_stamp() -> list[str]:
    from tools import registry

    failures: list[str] = []
    registry.activate_mode('code')

    if LOG.exists():
        failures.append(f"precondition: temp log already exists at {LOG}")
        return failures

    first = 'edit_file reported success but the file was unchanged'
    result = _dispatch(first)
    if result.status != 'success':
        failures.append(
            f"happy path: expected success, got {result.status!r} "
            f"code={result.code!r} body={result.body!r}"
        )
        return failures
    if not LOG.exists():
        failures.append(f"happy path: log was not created at {LOG}")
        return failures

    body = result.body if isinstance(result.body, str) else str(result.body)
    # The world fact in the render is what stops report_issue reading as an
    # escape hatch (E21/E24/E26/E27 render pattern) -- assert it is there.
    if 'does not fix the problem' not in body:
        failures.append(f"happy path: success body drops the 'does not fix' world fact: {body!r}")

    after_first = LOG.read_text(encoding='utf-8')

    second = 'find_symbol returned nothing for a symbol I had already read'
    result = _dispatch(second)
    if result.status != 'success':
        failures.append(f"append: expected success, got {result.status!r} body={result.body!r}")
        return failures

    if not LOG.read_text(encoding='utf-8').startswith(after_first):
        failures.append("append: the second call did not preserve the first entry byte-for-byte")

    entries = _entries()
    if len(entries) != 2:
        failures.append(f"append: expected 2 parsed entries, got {len(entries)}: {entries!r}")
        return failures

    for i, expected in enumerate((first, second)):
        entry = entries[i]
        if entry.get('issue') != expected:
            failures.append(f"entry {i}: issue round-trip mismatch: {entry.get('issue')!r} != {expected!r}")
        if not entry.get('date'):
            failures.append(f"entry {i}: no date field: {entry!r}")
        if entry.get('mode') != 'code':
            failures.append(f"entry {i}: expected mode='code' (the activated mode), got {entry.get('mode')!r}")

    return failures


# ---------------------------------------------------------------------------
# d. multi-line text with an embedded --- line
# ---------------------------------------------------------------------------


def check_multiline_block_scalar() -> list[str]:
    failures: list[str] = []
    before = len(_entries())

    text = 'run_command hung with no output.\n---\n  indented second line\n\nlast line'
    result = _dispatch(text)
    if result.status != 'success':
        failures.append(f"multi-line: expected success, got {result.status!r} body={result.body!r}")
        return failures

    raw = LOG.read_text(encoding='utf-8')
    if 'issue: |2' not in raw:
        failures.append(f"multi-line: no '|2' block scalar in the log: {raw!r}")

    entries = _entries()
    if len(entries) != before + 1:
        failures.append(
            f"multi-line: an embedded '---' split the log -- expected {before + 1} entries, "
            f"got {len(entries)}"
        )
        return failures
    if entries[-1].get('issue', '').rstrip('\n') != text:
        failures.append(f"multi-line: text did not round-trip: {entries[-1].get('issue')!r} != {text!r}")

    return failures


# ---------------------------------------------------------------------------
# e. single-line text that would otherwise be ambiguous YAML
# ---------------------------------------------------------------------------


def check_ambiguous_single_line() -> list[str]:
    failures: list[str] = []
    for text in ('edit_file: old_text not found, but I had just read it', '- leading dash then: a colon'):
        before = len(_entries())
        blocks_before = LOG.read_text(encoding='utf-8').count('issue: |2')
        result = _dispatch(text)
        if result.status != 'success':
            failures.append(f"ambiguous {text!r}: expected success, got {result.status!r}")
            continue
        entries = _entries()
        if len(entries) != before + 1:
            failures.append(f"ambiguous {text!r}: expected {before + 1} entries, got {len(entries)}")
            continue
        if entries[-1].get('issue') != text:
            failures.append(f"ambiguous {text!r}: did not round-trip: {entries[-1].get('issue')!r}")
        if LOG.read_text(encoding='utf-8').count('issue: |2') != blocks_before:
            failures.append(f"ambiguous {text!r}: single-line text was written as a block scalar")
    return failures


# ---------------------------------------------------------------------------
# e2. adversarial corpus -- every input round-trips as an exact string
# ---------------------------------------------------------------------------

# Each of these broke a real round-trip during development, or guards the class
# one of them belongs to: a trailing colon and a tab are YAML syntax errors that
# corrupt the entry, and a bare single token loads as bool/int/date/null rather
# than as the reported text.
_ADVERSARIAL = (
    'edit failed:',
    'edit_file: old_text not found',
    'key:value with no space',
    'value #hash',
    '@mention at start',
    'text with "double quotes" inside',
    "text with 'single quotes' inside",
    'ends with a backslash \\',
    'has\ttab inside',
    'true', '123', 'null', 'yes', 'inf', '0x1f', '0o777', '1_000', '.5',
    '2026-07-31', '2026-07-31 12:00:00',
    '- dash start', '? question start', '%directive', '*anchor', '&anchor',
    '!tag', '|pipe', '>gt', '{brace}', '[bracket]',
    'emoji and unicode: \U0001f680 …',
    # Multi-line shapes. The leading-indent case is the whole reason the block
    # scalar carries an explicit '|2' indicator rather than a bare '|'.
    'multi\nline\nwith --- inside\n  and indentation',
    'line one\n\tstarts with a tab\nline three',
    'a\n\n\nb',
    'traceback:\n  File "x.py", line 3\n    raise ValueError(\'x: y\')\nValueError: x: y',
)


def check_adversarial_corpus() -> list[str]:
    failures: list[str] = []
    for text in _ADVERSARIAL:
        before = len(_entries())
        result = _dispatch(text)
        if result.status != 'success':
            failures.append(f"{text!r}: expected success, got {result.status!r} body={result.body!r}")
            continue
        entries = _entries()
        if len(entries) != before + 1:
            failures.append(f"{text!r}: log split -- expected {before + 1} entries, got {len(entries)}")
            continue
        got = entries[-1].get('issue')
        if not isinstance(got, str):
            failures.append(f"{text!r}: type drift -- loaded as {type(got).__name__} ({got!r})")
            continue
        if got.rstrip('\n') != text:
            failures.append(f"{text!r}: did not round-trip: {got!r}")
    return failures


# ---------------------------------------------------------------------------
# e3. the block scalar states its own indentation
# ---------------------------------------------------------------------------


def check_block_indent_indicator() -> list[str]:
    """A bare ``|`` would make an indented first line unloadable; ``|2`` does not.

    Exercised at the renderer rather than through dispatch because ``run()``
    strips the text first -- which is exactly the upstream invariant the
    explicit indicator exists not to depend on.
    """
    import yaml

    from tools.report_issue import _yaml_issue

    failures: list[str] = []
    text = '  indented first line\nless indented second line'
    rendered = _yaml_issue(text)

    if not rendered.startswith('issue: |2'):
        failures.append(f"expected an explicit '|2' indicator, got: {rendered.splitlines()[0]!r}")

    try:
        got = yaml.safe_load(rendered + '\n')['issue']
    except Exception as exc:
        failures.append(f"indented first line did not load: {type(exc).__name__}: {exc}")
        return failures
    if got.rstrip('\n') != text:
        failures.append(f"indented first line did not round-trip: {got!r} != {text!r}")

    # Prove the indicator is what saves it: inferred indentation cannot.
    bare = rendered.replace('issue: |2', 'issue: |', 1)
    try:
        yaml.safe_load(bare + '\n')
    except Exception:
        pass
    else:
        failures.append(
            "a bare '|' loads this text fine, so the '|2' indicator is no longer "
            "guarding anything -- re-derive whether it is still needed"
        )
    return failures


# ---------------------------------------------------------------------------
# g. zero mutation events
# ---------------------------------------------------------------------------


def check_no_mutation_events() -> list[str]:
    from tools import _sandbox

    failures: list[str] = []
    events: list[dict] = []

    def _recorder(event: dict) -> None:
        events.append(event)

    _sandbox.subscribe_mutations(_recorder)
    try:
        result = _dispatch('get_diagnostics returned stale diagnostics after an edit')
        if result.status != 'success':
            failures.append(f"mutation check: expected success, got {result.status!r}")
            return failures
        if events:
            failures.append(
                f"mutation check: report_issue emitted {len(events)} mutation event(s) -- a report "
                f"would land in files_changed and arm the verification gate: {events!r}"
            )
    finally:
        if _recorder in _sandbox.MUTATION_SUBSCRIBERS:
            _sandbox.MUTATION_SUBSCRIBERS.remove(_recorder)
    return failures


# ---------------------------------------------------------------------------
# h. bad arguments
# ---------------------------------------------------------------------------


def check_bad_arguments() -> list[str]:
    import re

    failures: list[str] = []
    before = LOG.read_text(encoding='utf-8')

    for label, value, expected_code in (
        ('empty string', '', 'missing-argument'),
        ('whitespace only', '   \n\t ', 'missing-argument'),
        ('missing key', None, 'bad-arguments'),
        ('non-string', 42, 'bad-arguments'),
    ):
        result = _dispatch(value)
        if result.status != 'error':
            failures.append(f"{label}: expected error, got {result.status!r} body={result.body!r}")
            continue
        if result.code != expected_code:
            failures.append(f"{label}: expected code={expected_code!r}, got {result.code!r}")
        if not re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)*', result.code or ''):
            failures.append(f"{label}: code {result.code!r} is not kebab-case")

    if LOG.read_text(encoding='utf-8') != before:
        failures.append("bad arguments: a rejected call still wrote to the log")

    return failures


# ---------------------------------------------------------------------------
# i. oversize text
# ---------------------------------------------------------------------------


def check_oversize() -> list[str]:
    from tools import report_issue as mod

    failures: list[str] = []
    text = 'x' * (mod.MAX_ISSUE_CHARS + 1000)
    result = _dispatch(text)
    if result.status != 'success':
        failures.append(f"oversize: expected success, got {result.status!r} body={result.body!r}")
        return failures
    if result.meta.get('truncated') is not True:
        failures.append(f"oversize: expected meta truncated=True, got {result.meta.get('truncated')!r}")

    written = _entries()[-1].get('issue', '')
    if len(written) > mod.MAX_ISSUE_CHARS + 100:
        failures.append(f"oversize: written issue is {len(written)} chars, cap is {mod.MAX_ISSUE_CHARS}")
    if 'report truncated' not in written:
        failures.append("oversize: truncated text carries no truncation marker")

    ok = _dispatch('a short report that fits well under the cap')
    if ok.meta.get('truncated') is not False:
        failures.append(f"under cap: expected meta truncated=False, got {ok.meta.get('truncated')!r}")

    return failures


# ---------------------------------------------------------------------------
# j. dispatch-path and name-set membership
# ---------------------------------------------------------------------------


def check_agent_wiring() -> list[str]:
    import agent
    from tools import registry

    failures: list[str] = []
    tool = registry.get_tool('report_issue')
    if tool is None:
        failures.append("wiring: report_issue is not registered")
        return failures

    # parallel_safe tools ride a ThreadPoolExecutor path that skips mutation
    # attribution, pre-image capture and the loop guard on the documented
    # invariant that they never mutate. This one writes a file.
    if tool.parallel_safe is not False:
        failures.append("wiring: report_issue must not be parallel_safe -- it writes a file")

    # Left out of the repeat-cap exemption on purpose: three byte-identical
    # reports in one turn is duplicate spam, and the existing cap blocks it.
    from turn.verification import _REPEAT_CAP_EXEMPT

    if 'report_issue' in agent._VERIFICATION_TOOLS:
        failures.append("wiring: report_issue must not count as a verification tool")
    if 'report_issue' in _REPEAT_CAP_EXEMPT:
        failures.append("wiring: report_issue must not be exempt from the repeat-call cap")

    return failures


CHECKS = (
    ('all-modes', check_all_modes),
    ('append-and-stamp', check_append_and_stamp),
    ('multiline-block-scalar', check_multiline_block_scalar),
    ('ambiguous-single-line', check_ambiguous_single_line),
    ('adversarial-corpus', check_adversarial_corpus),
    ('block-indent-indicator', check_block_indent_indicator),
    ('no-mutation-events', check_no_mutation_events),
    ('bad-arguments', check_bad_arguments),
    ('oversize', check_oversize),
    ('agent-wiring', check_agent_wiring),
)


def main() -> int:
    global LOG

    # PyYAML is a declared dev dependency (pyproject.toml), so its absence is a
    # broken environment rather than a reason to pass. Every check below parses
    # the log back with it: skipping all ten and returning 0 reported a gate that
    # never ran as a gate that passed, which is how the whole contract sat
    # unexercised while smoke counted it green.
    try:
        import yaml  # noqa: F401
    except ImportError:
        print(
            'FAIL: PyYAML is not importable, so no check in this contract ran. '
            'Run the evals under `uv run` so the declared dev dependencies are '
            'present.',
            file=sys.stderr,
        )
        return 1

    from tools import registry, report_issue

    saved_mode = registry.current_mode()
    saved_path = report_issue.ISSUES_PATH
    tmp = tempfile.mkdtemp(prefix='report_issue_contract_')
    LOG = Path(tmp) / 'issues.md'
    report_issue.ISSUES_PATH = LOG

    failures: list[str] = []
    try:
        for label, check in CHECKS:
            try:
                failures.extend(f"[{label}] {f}" for f in check())
            except Exception as exc:  # pragma: no cover - unexpected
                failures.append(f"[{label}] unexpected error: {type(exc).__name__}: {exc}")
    finally:
        report_issue.ISSUES_PATH = saved_path
        shutil.rmtree(tmp, ignore_errors=True)
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(
        "PASS: report_issue reaches every mode, appends parseable YAML entries "
        "(block scalar for multi-line, --- safe), stamps the mode, emits no mutation "
        "events, rejects blank input, and caps oversize reports"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
