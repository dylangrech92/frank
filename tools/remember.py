"""Remember tool: persist a durable (kind, key, value) fact into project memory."""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult


class Remember(Tool):
    """Persist a durable fact as a (kind, key, value) atom.

    Use this tool to save important information about the project, conventions,
    discoveries, or miscellaneous facts. Always reuse the *key* when updating or
    correcting an existing fact so that the old value is superseded rather than
    duplicated.  Supported kinds are: **project**, **convention**, **discovery**,
    **misc**.
    """

    name = 'remember'
    summary = 'Persist a durable (kind, key, value) fact to memory.'
    description = (
        'Save a durable fact as a (kind, key, value) atom in the project memory. '
        'Always reuse the same *key* when updating an existing fact so it is '
        'superseded rather than duplicated. Supported kinds: project, convention, '
        'discovery, misc.'
    )
    parameters: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'kind': {
                'type': 'string',
                'description': 'The category of the fact. Must be one of: project, convention, discovery, misc.',
                'enum': ['project', 'convention', 'discovery', 'misc'],
            },
            'key': {
                'type': 'string',
                'description': (
                    'A short, stable identifier for this fact. Reuse the same key when '
                    'updating or correcting an existing fact so the old value is superseded.'
                ),
            },
            'value': {
                'type': 'string',
                'description': 'The fact text itself -- the content to persist.',
            },
        },
        'required': ['kind', 'key', 'value'],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the remember tool.

        Args:
            **kwargs: Parsed from LLM function-call payload. Requires ``kind``,
                ``key``, and ``value`` -- all non-empty strings; *kind* must be one
                of the four supported values.

        Returns:
            A ``ToolResult`` with the saved fact metadata on success, or an error
            when validation fails.
        """
        kind_raw = kwargs.get('kind') if isinstance(kwargs.get('kind'), str) else ''
        key_raw = kwargs.get('key') if isinstance(kwargs.get('key'), str) else ''
        value_raw = kwargs.get('value') if isinstance(kwargs.get('value'), str) else ''

        # --- validate kind ----------------------------------------------------
        from memory.atomic import KINDS  # pylint: disable=import-outside-toplevel

        if kind_raw not in KINDS:
            return ToolResult.err(
                f"'kind' must be one of {KINDS}, got {kind_raw!r}.",
                code='invalid-kind',
                hint=f"Supported kinds are {KINDS}.",
            )

        # --- validate key and value ------------------------------------------
        if not key_raw:
            return ToolResult.err(
                'The "key" argument is required and must be a non-empty string.',
                code='missing-argument',
            )
        if not value_raw:
            return ToolResult.err(
                'The "value" argument is required and must be a non-empty string.',
                code='missing-argument',
            )

        # --- persist ----------------------------------------------------------
        from memory.recall import remember as _remember  # pylint: disable=import-outside-toplevel

        fact_id = _remember(kind_raw, key_raw, value_raw)

        return ToolResult.ok(
            f'Remembered [{kind_raw}] {key_raw}.',
            fact_id=fact_id,
            kind=kind_raw,
            key=key_raw,
        )
