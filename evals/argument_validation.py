"""Dispatch-level contract check for tool argument validation (no LLM).

``registry.validate_arguments`` is the gate every tool call passes through, and
until this file existed nothing exercised it. Two defects lived there from the
day it was written and only surfaced when ``report`` became the first tool with
four required parameters:

1. the missing-parameter message joined names and types as two separate lists,
   so two missing parameters rendered as ``'verdict', 'observations'
   (string)(string)`` — every type detached from the name it described;
2. the validator returned after the first *class* of problem it found, so a
   call with an unknown key, a missing key and a wrong type was corrected one
   class per round-trip. In a live run that cost four rejected ``report`` calls
   and tripped the repeat-call guard before the model got there.

The checks below assert the message *shape* against hardcoded expectations
rather than anything derived from the schema under test — an expectation
computed from the code it is checking always passes. Asserts:

a. two or more missing parameters keep each name adjacent to its own type, and
   the detached ``)(`` rendering never comes back;
b. an unknown key, missing keys and wrong types are all reported in one
   message, one problem per line;
c. a caller that applies exactly the corrections the message names converges in
   a single retry — the property the short-circuit broke;
d. valid arguments pass, including a property with no declared type and one
   whose declared type the validator has no mapping for;
e. ``bool`` is still rejected for ``integer``/``number`` (it is an ``int``
   subclass in Python) while genuinely passing for ``boolean``;
f. an unknown key produces exactly one problem, not one per check;
g. end to end through the real ``registry.dispatch`` on the real ``report``
   tool: the exact payload from the live run is rejected as ``bad-arguments``
   naming both its violations at once.

Exits 0 on success, prints each ``FAIL: <reason>`` to stderr and exits 1
otherwise. Runs with the repo root on ``sys.path`` (evals/run.py inserts it;
running directly needs PYTHONPATH=<repo root>) and restores the active mode.
"""

from __future__ import annotations

import re
import sys

from tools.base import Tool
from tools.result import ToolResult


class _Fixture(Tool):
    """A schema pinned here on purpose, so a real tool's schema drifting cannot
    quietly change what these message-shape assertions mean."""

    name = 'argument_validation_fixture'
    description = 'Never registered — it lives in evals/, which discover() does not walk.'
    parameters = {
        'type': 'object',
        'properties': {
            'title': {'type': 'string'},
            'count': {'type': 'integer'},
            'ratio': {'type': 'number'},
            'enabled': {'type': 'boolean'},
            'items': {'type': 'array'},
            'meta': {'type': 'object'},
            'freeform': {},  # no declared type at all
            'exotic': {'type': 'null'},  # a type _PY_TYPE_MAP does not carry
        },
        'required': ['title', 'count', 'items', 'meta'],
    }

    def run(self, **kwargs: object) -> ToolResult:  # pragma: no cover - never dispatched
        raise AssertionError('the fixture tool must never be executed')


FIXTURE = _Fixture()

# Known-good values for every fixture property, used to correct a rejected call.
# Hardcoded rather than generated from the schema.
_GOOD: dict[str, object] = {
    'title': 'a title',
    'count': 3,
    'ratio': 1.5,
    'enabled': True,
    'items': [],
    'meta': {},
    'freeform': object(),
    'exotic': None,
}


def _validate(arguments: dict) -> str | None:
    from tools.registry import validate_arguments

    return validate_arguments(FIXTURE, arguments)


# ---------------------------------------------------------------------------
# a. each missing name carries its own type
# ---------------------------------------------------------------------------


def check_missing_name_type_pairing() -> list[str]:
    failures: list[str] = []

    message = _validate({'ratio': 1.0})
    expected = (
        "missing required parameter(s) 'title' (string), 'count' (integer), "
        "'items' (array), 'meta' (object)"
    )
    if message != expected:
        failures.append(f"four missing params: expected {expected!r}, got {message!r}")

    # The specific regression: names joined, then types joined separately.
    if message and ')(' in message:
        failures.append(
            f"missing-params message has adjacent type groups — types are detached "
            f"from their names again: {message!r}"
        )

    # Every declared name must be immediately followed by its own type.
    for name, jtype in (('title', 'string'), ('count', 'integer'), ('items', 'array')):
        if f"'{name}' ({jtype})" not in (message or ''):
            failures.append(f"missing-params message does not pair {name!r} with ({jtype}): {message!r}")

    # A single missing parameter must read the same way.
    one = _validate({'title': 'x', 'count': 1, 'items': []})
    if one != "missing required parameter(s) 'meta' (object)":
        failures.append(f"single missing param: got {one!r}")

    return failures


# ---------------------------------------------------------------------------
# b. all three problem classes in one message
# ---------------------------------------------------------------------------


def check_all_classes_reported() -> list[str]:
    failures: list[str] = []

    # unknown 'bogus'; missing 'items'/'meta'; wrong types on 'title' and 'count'
    message = _validate({'bogus': 1, 'title': 5, 'count': 'three'})
    if message is None:
        failures.append('a payload with four violations validated clean')
        return failures

    lines = message.splitlines()
    if len(lines) != 4:
        failures.append(f"expected 4 problem lines (unknown, missing, 2 type), got {len(lines)}: {message!r}")

    for fragment in (
        "unknown parameter(s) 'bogus'",
        "missing required parameter(s) 'items' (array), 'meta' (object)",
        "parameter 'title' must be string, got int",
        "parameter 'count' must be integer, got str",
    ):
        if fragment not in message:
            failures.append(f"message is missing the {fragment!r} problem: {message!r}")

    if 'allowed parameters are' not in message:
        failures.append(f"unknown-key problem does not list the allowed names: {message!r}")

    return failures


# ---------------------------------------------------------------------------
# c. one corrected retry clears every violation
# ---------------------------------------------------------------------------


def _correct(arguments: dict, message: str) -> dict:
    """Apply exactly the corrections *message* names — nothing more.

    This models the caller the validator is written for: it can only fix what
    it was told about, so the number of round-trips is a direct measure of how
    much the validator reports per call.
    """
    fixed = dict(arguments)
    for line in message.splitlines():
        if line.startswith('unknown parameter(s)'):
            named = line.split(';')[0]
            for name in re.findall(r"'([^']+)'", named):
                fixed.pop(name, None)
        elif line.startswith('missing required parameter(s)'):
            for name in re.findall(r"'([^']+)'", line):
                fixed[name] = _GOOD[name]
        elif line.startswith('parameter '):
            match = re.match(r"parameter '([^']+)'", line)
            if match:
                fixed[match.group(1)] = _GOOD[match.group(1)]
    return fixed


def check_converges_in_one_retry() -> list[str]:
    failures: list[str] = []

    for label, payload in (
        ('four violations, three classes', {'bogus': 1, 'title': 5, 'count': 'three'}),
        ('every class at once', {'x': 0, 'y': 0, 'title': [], 'count': 1.5, 'items': {}}),
        ('the live report payload shape', {'title': 5}),
    ):
        arguments = dict(payload)
        rounds = 0
        while True:
            message = _validate(arguments)
            if message is None:
                break
            rounds += 1
            if rounds > 5:
                failures.append(f"{label}: did not converge in 5 rounds, still: {message!r}")
                break
            arguments = _correct(arguments, message)

        if rounds > 1:
            failures.append(
                f"{label}: took {rounds} rejected calls to converge — the validator is "
                f"reporting one problem class per round again, which is what tripped the "
                f"repeat-call guard live"
            )
    return failures


# ---------------------------------------------------------------------------
# d. valid arguments pass
# ---------------------------------------------------------------------------


def check_valid_passes() -> list[str]:
    failures: list[str] = []

    for label, arguments in (
        ('required only', {'title': 'x', 'count': 1, 'items': [], 'meta': {}}),
        ('every property', dict(_GOOD)),
        ('untyped property carries anything', {'title': 'x', 'count': 1, 'items': [], 'meta': {}, 'freeform': 42}),
        ('unmapped type is not policed', {'title': 'x', 'count': 1, 'items': [], 'meta': {}, 'exotic': 'anything'}),
        ('int is acceptable for number', {'title': 'x', 'count': 1, 'items': [], 'meta': {}, 'ratio': 2}),
    ):
        message = _validate(arguments)
        if message is not None:
            failures.append(f"{label}: valid arguments were rejected: {message!r}")

    return failures


# ---------------------------------------------------------------------------
# e. bool is not an integer
# ---------------------------------------------------------------------------


def check_bool_is_not_a_number() -> list[str]:
    """``isinstance(True, int)`` is True in Python, so this guard is load-bearing."""
    failures: list[str] = []
    base = {'title': 'x', 'count': 1, 'items': [], 'meta': {}}

    for field, expected_type in (('count', 'integer'), ('ratio', 'number')):
        message = _validate({**base, field: True})
        wanted = f"parameter '{field}' must be {expected_type}, got bool"
        if message != wanted:
            failures.append(f"bool for {field}: expected {wanted!r}, got {message!r}")

    if _validate({**base, 'enabled': True}) is not None:
        failures.append('a genuine bool was rejected for a boolean property')
    if _validate({**base, 'enabled': 1}) != "parameter 'enabled' must be boolean, got int":
        failures.append('an int was accepted for a boolean property')

    return failures


# ---------------------------------------------------------------------------
# f. an unknown key is reported once
# ---------------------------------------------------------------------------


def check_unknown_reported_once() -> list[str]:
    failures: list[str] = []
    base = {'title': 'x', 'count': 1, 'items': [], 'meta': {}}

    message = _validate({**base, 'bogus': 1})
    if message is None:
        failures.append('an unknown key was accepted')
        return failures
    if len(message.splitlines()) != 1:
        failures.append(f"an unknown key produced {len(message.splitlines())} problems, expected 1: {message!r}")
    if 'must be' in message:
        failures.append(f"an unknown key was also type-checked against a schema it has no entry in: {message!r}")

    two = _validate({**base, 'bogus': 1, 'alsobad': 2})
    if two is None or "'bogus', 'alsobad'" not in two:
        failures.append(f"two unknown keys are not listed together: {two!r}")

    return failures


# ---------------------------------------------------------------------------
# g. end to end through the real dispatcher, on the tool that exposed this
# ---------------------------------------------------------------------------


def check_dispatch_on_report() -> list[str]:
    """The literal payload from the live run, through the production path."""
    from tools import registry

    failures: list[str] = []
    registry.activate_mode('verify')

    # What the model actually sent: observations absent, and plan serialised as
    # a string rather than an array. The old validator named only the first.
    result = registry.dispatch(
        'report',
        {'verdict': 'fail', 'plan': 'I planned to check the reset page', 'assertions': []},
    )

    if result.status != 'error':
        failures.append(f"expected an error, got {result.status!r}")
        return failures
    if result.code != 'bad-arguments':
        failures.append(f"expected code='bad-arguments', got {result.code!r}")

    body = result.body if isinstance(result.body, str) else str(result.body)
    if "'observations'" not in body:
        failures.append(f"the missing 'observations' key is not named: {body!r}")
    if "must be array" not in body:
        failures.append(
            f"the wrong-typed 'plan' is not named alongside the missing key — this is the "
            f"one-problem-per-round-trip behaviour that cost four calls live: {body!r}"
        )

    return failures


CHECKS = (
    ('missing-name-type-pairing', check_missing_name_type_pairing),
    ('all-classes-reported', check_all_classes_reported),
    ('converges-in-one-retry', check_converges_in_one_retry),
    ('valid-passes', check_valid_passes),
    ('bool-is-not-a-number', check_bool_is_not_a_number),
    ('unknown-reported-once', check_unknown_reported_once),
    ('dispatch-on-report', check_dispatch_on_report),
)


def main() -> int:
    from tools import registry

    saved_mode = registry.current_mode()

    failures: list[str] = []
    try:
        for label, check in CHECKS:
            try:
                failures.extend(f"[{label}] {f}" for f in check())
            except Exception as exc:  # pragma: no cover - unexpected
                failures.append(f"[{label}] unexpected error: {type(exc).__name__}: {exc}")
    finally:
        if saved_mode is not None:
            registry.activate_mode(saved_mode)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(
        "PASS: argument validation pairs each missing name with its own type, reports "
        "unknown/missing/wrong-typed arguments together so one corrected retry clears "
        "them, keeps bool out of the numeric types, and rejects the live report payload "
        "naming both its violations at once"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
