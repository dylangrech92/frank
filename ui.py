"""Minimal ANSI colorization helpers for interactive terminal output.

Pretty-printing is off by default (plain text, current behavior). Call
``enable(True)`` once at startup (wired from the ``--pretty`` CLI flag) to turn
on color; every semantic helper below then wraps its text in the relevant ANSI
codes, and no-ops back to plain text when disabled. Stdlib only, no external
dependencies.
"""

from __future__ import annotations

RESET = "0"
BOLD = "1"
DIM = "2"

FG_RED = "31"
FG_GREEN = "32"
FG_CYAN = "36"
FG_GRAY = "90"

_PRETTY: bool = False


def enable(pretty: bool) -> None:
    """Toggle color output globally.

    Args:
        pretty: When True, subsequent calls to the helpers below emit ANSI
            escape codes; when False, they return their input unchanged.
    """
    global _PRETTY
    _PRETTY = pretty


def is_enabled() -> bool:
    """Return whether color output is currently enabled."""
    return _PRETTY


def style(text: str, *codes: str) -> str:
    """Wrap *text* in the given ANSI SGR *codes* when pretty mode is enabled.

    Args:
        text: The text to colorize.
        *codes: One or more ANSI SGR code strings (e.g. ``BOLD``, ``FG_CYAN``).

    Returns:
        *text* unchanged when pretty mode is off; otherwise *text* wrapped in
        the requested escape sequence and reset back to default afterward.
    """
    if not _PRETTY or not codes:
        return text
    prefix = "\x1b[" + ";".join(codes) + "m"
    return f"{prefix}{text}\x1b[{RESET}m"


def tool_call(text: str) -> str:
    """Render a tool-call echo line in cyan."""
    return style(text, FG_CYAN)


def tool_result(text: str) -> str:
    """Render a tool-result echo block dimmed."""
    return style(text, DIM)


def telemetry(text: str) -> str:
    """Render context/telemetry lines dimmed gray."""
    return style(text, DIM, FG_GRAY)


def error(text: str) -> str:
    """Render an error message in bold red."""
    return style(text, BOLD, FG_RED)


def assistant(text: str) -> str:
    """Prefix an assistant reply with a colored marker so it stands out.

    Args:
        text: The assistant's reply text.

    Returns:
        *text* unchanged when pretty mode is off; otherwise prefixed with a
        bold cyan ``agent:`` marker.
    """
    if not _PRETTY:
        return text
    marker = style("agent:", BOLD, FG_CYAN)
    return f"{marker} {text}"


def prompt_marker(text: str = "> ") -> str:
    """Render the REPL input prompt in bold green."""
    return style(text, BOLD, FG_GREEN)
