"""Handle-dialog tool: arm the next dialog's outcome and report dialogs seen."""

from __future__ import annotations

from typing import Any

from runtime.browser import get_session
from tools.base import Tool
from tools.result import ToolResult

_ACTIONS = ("accept", "dismiss")


class HandleDialog(Tool):
    """Arm accept/dismiss (with optional prompt text) for the next JS dialog."""

    name = "handle_dialog"
    description = (
        "Arm *action* ('accept' or 'dismiss') for the NEXT JavaScript dialog "
        "(alert/confirm/prompt) that appears. Without arming, dialogs are "
        "auto-dismissed. Optional *prompt_text* is entered before accepting a "
        "prompt() dialog. The arm is one-shot — it applies only to the next "
        "dialog. Also reports every dialog seen so far (type, message)."
    )
    action = "arm the dialog handler"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Whether to accept or dismiss the next dialog.",
            },
            "prompt_text": {
                "type": "string",
                "description": "Optional text to enter before accepting a prompt() dialog.",
            },
        },
        "required": ["action"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the handle_dialog tool.

        Args:
            action: 'accept' or 'dismiss', applied to the next dialog only.
            prompt_text: Optional text entered before accepting a prompt() dialog.

        Returns:
            A ``ToolResult`` confirming the arm and listing every dialog seen so
            far (full type and message text), or an error on bad arguments.
        """
        action = kwargs.get("action")
        if action not in _ACTIONS:
            return ToolResult.err(
                f"action must be one of {list(_ACTIONS)}, got {action!r}.", code="bad-arguments"
            )

        prompt_text = kwargs.get("prompt_text")
        if prompt_text is not None and not isinstance(prompt_text, str):
            return ToolResult.err("prompt_text must be a string when provided.", code="bad-arguments")

        session = get_session()
        session.page  # ensure the session (and its listeners) has started
        session.arm_dialog(action, prompt_text)

        seen = [f"[{d['type']}] {d['message']}" for d in session.dialogs_seen]
        armed_desc = f"armed to {action} the next dialog"
        if prompt_text is not None:
            armed_desc += f" with prompt_text {prompt_text!r}"
        body = (
            f"{armed_desc}.\n"
            f"dialogs seen so far ({len(seen)}):\n"
            + ("\n".join(seen) if seen else "(none)")
        )

        return ToolResult.ok(body, armed_action=action, dialogs_seen_count=len(seen))
