"""Core agent loop with four named extension hooks for lifecycle phases.

Implements ``handle_user_message`` as the main entry point that drives the
read-eval-print cycle between a Session transcript and an LLMClient, plus three
explicit hook no-op functions and an over-cap handling seam exposed inline in
the loop body.
"""

import json
import sys

from llm import ChatResponse, LLMClient, OverCapError, ToolCall
from session import Session
from tools.registry import dispatch, schemas
from tools.result import ToolResult


# =============================================================================
# Extension hooks — four named no-op seams for later phases
# =============================================================================

def flashback_maybe_seed(session: Session) -> None:
    """Turn-zero memory recall injection seam (phase N).

    Reads long-term episodic memory and optionally seeds the session with
    retrieved context before processing the user's request.

    Currently a no-op extension point.
    """
    pass


def diagnostics_inject_summary(session: Session) -> None:
    """Post-mutation diagnostics summary seam (phase N).

    Inspects file-system or other side-effects produced during this session and
    appends a diagnostics summary as an assistant message so the model remains
    accurate about the world state.

    Currently a no-op extension point.
    """
    pass


def episodic_maybe_extract(session: Session) -> None:
    """End-of-turn episodic memory extraction seam (phase N).

    Analyzes the completed turn for patterns worth persisting to long-term
    episodic memory and stores anything useful.

    Currently a no-op extension point.
    """
    pass


# The fourth hook — over-cap handling — lives directly in the ``handle_user_message``
# loop body as the OverCapError branch described by the contract below.  Its docstring
# lives inline there and is noted under the return path.


# =============================================================================
# Helpers
# =============================================================================

def render_tool_result(name: str, result: ToolResult) -> str:
    """Render a ``ToolResult`` value as plain text for a tool message content slot.

    The rendered string follows this shape:

        [name(STATUS code=CODE)]       # error status with code
        [name(STATUS)]                  # success status, no code
        <body>                          # only when body is non-empty
        hint: <hint_text>               # only when a hint is present

    The bracket header always contains the tool name and status.  When status
    is ``'error'`` and ``result.code`` is set, the kebab-case code is appended
    inside the brackets as ``code=THECODE``.

    Args:
        name: Registered tool name (used in the header).
        result: A ``ToolResult`` instance to render.

    Returns:
        A plain-text string suitable for ``content`` on a ``tool`` role message.
    """
    lines: list[str] = []

    if result.status == "error" and result.code is not None:
        header = f"[{name}({result.status} code={result.code})]"
    else:
        header = f"[{name}({result.status})]"

    lines.append(header)

    body = result.body
    if body:
        if isinstance(body, (dict, list)):
            body_str_val = json.dumps(result.body)
        else:
            body_str_val = str(result.body)
        lines.append(body_str_val)

    hint_text = result.hint
    if hint_text is not None and hint_text:
        lines.append(f"hint: {hint_text}")

    return "\n".join(lines)


# =============================================================================
# Main entry point
# =============================================================================

def handle_user_message(
    text: str,
    session: Session,
    client: LLMClient,
    verbose: bool = False,
) -> str:
    """Execute one agent turn in response to a user message.

    The loop follows this contract exactly:

    1. **setup** — append_user(text), flashback_maybe_seed(session).
    2. **loop** (while True):
       a. assemble_context() from session.
       b. verbose → print full assembled context to stderr prefixed by "assembled context".
       c. client.chat(context, schemas()) inside try … except OverCapError.
       d. **OverCapError path** — append_assistant with a message saying the context
          exceeded the model limit; return that same text (turn ends without retry).
          *A later phase replaces this behaviour with compact-then-retry.*
       e. **normal success** — append_assistant(text, tool_calls), then:
          i.  If no tool calls → episodic_maybe_extract(session) + return response.text.
          ii. For each tool call:
              - echo to stderr the call name and arguments.
              - dispatch via registry (dispatch(name, arguments)).
              - echo rendered result to stderr via render_tool_result().
              - store append_tool_result(call.id, call.name, rendered_result).
          iii. After all calls → diagnostics_inject_summary(session); continue loop.

    Args:
        text: User message string to begin the turn with.
        session: Active ``Session`` holding the conversation transcript.
        client: An ``LLMClient`` instance for sending chat requests.
        verbose: If True, prints the full assembled context lines (prefixed by
            "assembled context") to stderr before each LLM call.

    Returns:
        The final assistant text string — either a normal turn response, an
        episodic extraction pass-through, or the over-cap sentinel message.
    """
    session.append_user(text)
    flashback_maybe_seed(session)

    while True:
        context = session.assemble_context()

        if verbose:
            msg_lines: list[str] = []
            for i, m in enumerate(context):
                role_val = str(m.get("role", ""))
                content_val = str(m.get("content", ""))
                msg_lines.append(
                    f"assembled context\n[{i}] {role_val}: {content_val}"
                )
            print("\n".join(msg_lines), file=sys.stderr)

        try:
            response: ChatResponse = client.chat(context, schemas())
        except OverCapError:
            cap_msg = (
                "Context exceeded the model's token limit for this turn. "
                "Aborting the current retry loop."
            )
            # A later phase replaces this branch with compact-then-retry to
            # trim older messages, rebuild context, and retry the call.
            session.append_assistant(cap_msg)
            return cap_msg

        tool_calls: list[ToolCall] | None = response.tool_calls if response.tool_calls else None
        session.append_assistant(response.text or "", tool_calls=tool_calls)

        if not response.tool_calls:
            episodic_maybe_extract(session)
            return response.text

        for call in response.tool_calls:
            print(
                f"Tool call: {call.name}({json.dumps(call.arguments)})",
                file=sys.stderr,
            )

            result = dispatch(call.name, call.arguments)
            rendered = render_tool_result(call.name, result)

            print(rendered, file=sys.stderr)

            session.append_tool_result(call.id, call.name, rendered)

        diagnostics_inject_summary(session)
