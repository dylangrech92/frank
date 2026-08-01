"""Report tool: the terminal evidence gate that submits the final verdict.

A normal, auto-discovered ``Tool`` like every other browser tool -- advertised
to the model the same way -- but its ``run()`` is the reliability keystone of
the whole agent: it manually validates the nested assertion/evidence
structure (the registry only type-checks the four top-level parameters) and
refuses to accept a 'pass' verdict, at any level, that is not backed by
evidence captured this run. ``agent.py`` treats a successful call here as the
run's terminal event -- the model is expected to call this exactly once.
"""

from __future__ import annotations

from typing import Any

from tools.base import Tool
from tools.result import ToolResult

_VERDICTS = ("pass", "fail", "inconclusive")
_EVIDENCE_KINDS = ("aria", "console", "network", "http", "screenshot", "url")

_EVIDENCE_HINT = (
    "Every 'pass' assertion must cite evidence.detail captured THIS run (a "
    "snapshot/console/network/http/url signal or a screenshot). Cite it, or "
    "downgrade the verdict to 'inconclusive'."
)


class Report(Tool):
    """Submit the final verification verdict. Ends the run -- call exactly once."""

    name = "report"
    description = (
        "Submit the final verification result and END the run. Call this "
        "exactly once, when you are done checking every assertion in your "
        "plan -- this is not a progress update, it is the last thing you do. "
        "Every assertion whose verdict is 'pass' MUST cite evidence "
        "(evidence.kind and evidence.detail) captured during THIS run: a "
        "snapshot/console/network/http/url signal you actually observed, or a "
        "screenshot you actually took. If you cannot back a claim with "
        "evidence, its verdict must be 'inconclusive', never 'pass'. The "
        "top-level verdict must not be 'pass' if any assertion failed."
    )
    action = "submit the final report"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": list(_VERDICTS),
                "description": "The overall verdict for this verification run.",
            },
            "plan": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The test plan you generated up front: the concrete "
                    "assertions you set out to check."
                ),
            },
            "assertions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "assertion": {
                            "type": "string",
                            "description": "The specific claim being checked.",
                        },
                        "verdict": {"type": "string", "enum": list(_VERDICTS)},
                        "evidence": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": list(_EVIDENCE_KINDS)},
                                "detail": {
                                    "type": "string",
                                    "description": (
                                        "What you observed, in your own words, "
                                        "citing the concrete signal."
                                    ),
                                },
                                "artifact": {
                                    "type": "string",
                                    "description": (
                                        "Optional path to a saved artifact (e.g. "
                                        "a screenshot) backing this evidence."
                                    ),
                                },
                            },
                            "required": ["kind", "detail"],
                        },
                    },
                    "required": ["assertion", "verdict", "evidence"],
                },
                "description": (
                    "One entry per assertion in your plan, each with its own "
                    "verdict and evidence."
                ),
            },
            "observations": {
                "type": "string",
                "description": (
                    "Narrative of anything unexpected you found, beyond the "
                    "specific assertions."
                ),
            },
        },
        "required": ["verdict", "plan", "assertions", "observations"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Validate the nested report structure and enforce the evidence gate.

        The registry only type-checks the four top-level parameters before
        calling this (``verdict`` is a string, ``plan``/``assertions`` are
        lists, ``observations`` is a string) -- everything nested is validated
        here, in order:

        1. Structural checks on every assertion (dict shape, valid verdict
           enum, evidence is an object, evidence.kind/artifact well-typed when
           present) -- failures return ``code="bad-arguments"``.
        2. The evidence gate: every assertion whose ``verdict`` is ``"pass"``
           or ``"fail"`` must carry a non-empty ``evidence.detail`` and a
           valid ``evidence.kind`` -- failure returns
           ``code="evidence-required"``. ``"inconclusive"`` assertions stay
           lenient (empty detail allowed), matching the honest-verdict
           principle: you cannot always evidence why something is uncertain.
        3. Top-level coherence: ``verdict == "pass"`` is rejected if any
           assertion's verdict is ``"fail"`` -- returns
           ``code="incoherent-verdict"``.

        Returns:
            ``ToolResult.ok("report accepted")`` when every check passes,
            else a ``ToolResult.err`` naming the first problem found, with a
            ``hint`` telling the model how to fix it.
        """
        verdict = kwargs.get("verdict")
        plan = kwargs.get("plan")
        assertions = kwargs.get("assertions")
        observations = kwargs.get("observations")

        if verdict not in _VERDICTS:
            return ToolResult.err(
                f"verdict must be one of {_VERDICTS}, got {verdict!r}.",
                code="bad-arguments",
            )
        if not isinstance(plan, list) or not all(isinstance(p, str) for p in plan):
            return ToolResult.err("plan must be an array of strings.", code="bad-arguments")
        if not isinstance(observations, str):
            return ToolResult.err("observations must be a string.", code="bad-arguments")
        if not isinstance(assertions, list) or not assertions:
            return ToolResult.err(
                "assertions must be a non-empty array -- report at least one "
                "assertion from your plan.",
                code="bad-arguments",
            )

        for i, item in enumerate(assertions):
            if not isinstance(item, dict):
                return ToolResult.err(f"assertions[{i}] must be an object.", code="bad-arguments")

            a_text = item.get("assertion")
            a_verdict = item.get("verdict")
            evidence = item.get("evidence")

            if not isinstance(a_text, str) or not a_text:
                return ToolResult.err(
                    f"assertions[{i}].assertion is required and must be a non-empty string.",
                    code="bad-arguments",
                )
            if a_verdict not in _VERDICTS:
                return ToolResult.err(
                    f"assertions[{i}].verdict must be one of {_VERDICTS}, got {a_verdict!r}.",
                    code="bad-arguments",
                )
            if not isinstance(evidence, dict):
                return ToolResult.err(
                    f"assertions[{i}].evidence is required and must be an object "
                    f"with 'kind' and 'detail'.",
                    code="bad-arguments",
                )

            kind = evidence.get("kind")
            detail = evidence.get("detail")
            artifact = evidence.get("artifact")

            if kind is not None and kind not in _EVIDENCE_KINDS:
                return ToolResult.err(
                    f"assertions[{i}].evidence.kind must be one of "
                    f"{_EVIDENCE_KINDS}, got {kind!r}.",
                    code="bad-arguments",
                )
            if detail is not None and not isinstance(detail, str):
                return ToolResult.err(
                    f"assertions[{i}].evidence.detail must be a string.",
                    code="bad-arguments",
                )
            if artifact is not None and not isinstance(artifact, str):
                return ToolResult.err(
                    f"assertions[{i}].evidence.artifact must be a string when provided.",
                    code="bad-arguments",
                )

            # THE EVIDENCE GATE: a 'pass' or 'fail' assertion with no captured
            # signal is never accepted as-is -- cite it, or the verdict must
            # be honest ('inconclusive') instead. 'inconclusive' stays
            # lenient since the whole point of that verdict is "I could not
            # get evidence."
            if a_verdict in ("pass", "fail") and (not detail or not kind):
                return ToolResult.err(
                    f"assertions[{i}] ('{a_text}') is marked {a_verdict!r} but is "
                    f"missing evidence.detail and/or evidence.kind.",
                    code="evidence-required",
                    hint=_EVIDENCE_HINT,
                )

        if verdict == "pass" and any(item.get("verdict") == "fail" for item in assertions):
            return ToolResult.err(
                "top-level verdict is 'pass' but at least one assertion failed "
                "-- that is incoherent.",
                code="incoherent-verdict",
                hint=(
                    "The overall verdict cannot be 'pass' while any assertion "
                    "failed. Set the top-level verdict to 'fail' (or "
                    "'inconclusive' if the failure itself is uncertain)."
                ),
            )

        return ToolResult.ok("report accepted")
