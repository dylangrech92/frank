"""Vision tool: ask the optional vision provider about a local image file, on demand."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import session
import vision
from tools.base import Tool
from tools.result import ToolResult
from tools._sandbox import resolve_in_root

# Magic-byte signatures for the image formats the vision providers in this
# project's configs accept. Hand-rolled rather than ``imghdr`` (deprecated in
# 3.12, removed in 3.13) or a new dependency (Pillow) — four fixed-prefix
# checks are all four formats need. WEBP is not prefix-matched here because
# its RIFF size field varies per file; see ``_sniff_image_mime``.
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff_image_mime(data: bytes) -> str | None:
    """Identify PNG / JPEG / GIF / WEBP by magic bytes; return the MIME type or None.

    An unrecognised signature means "not a decodable image" — this is a
    content-type check the vision provider's request depends on, not a
    best-effort guess that degrades gracefully when wrong.
    """
    for signature, mime in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            return mime
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class Vision(Tool):
    """Ask the optional vision provider about a local image file.

    Separate from the automatic screenshot description ``_attach_screenshot``
    performs on every capture: this tool lets the model query any image file
    on disk at will, with an optional *question* to focus the answer. Only
    registered when ``config.json`` carries a ``vision`` block (see
    ``modes.CONDITIONAL_TOOLS``) — the image goes to that provider alone, in
    one no-tools, no-history call, and the main model never receives pixels.
    """

    name = "vision"
    description = (
        "Ask the vision provider about a local image file (e.g. a saved "
        "screenshot or a diagram). Reads the file, sends it to the vision "
        "provider in one isolated call, and returns its text answer. Optionally "
        "focus the answer with a question; without one, a general description is "
        "returned. Only available when a vision provider is configured."
    )
    action = "ask the vision provider about the image"
    oversize_hint = "ask a narrower question about a specific part of the image"
    parallel_safe = True  # read-only network call, no shared state
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to a local image file, relative to the project root.",
            },
            "question": {
                "type": "string",
                "description": (
                    "Optional question to focus the answer on. When omitted, a "
                    "general fact-only description is returned instead."
                ),
            },
        },
        "required": ["image_path"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the vision tool.

        Args:
            image_path: Path to a local image file, relative to the project root.
            question: Optional question to focus the vision provider's answer on.

        Returns:
            A ``ToolResult`` with the provider's text as its body on success, or
            a loud error when the file is missing, unreadable, not a decodable
            image, no vision provider is configured, or the provider call fails.
        """
        # -------------------------------------------------------------------
        # 1. Validate image_path
        # -------------------------------------------------------------------
        raw_path = kwargs.get("image_path") if isinstance(kwargs.get("image_path"), str) else ""
        if not raw_path:
            return ToolResult.err(
                "image_path is required and must be a non-empty string.",
                code="bad-arguments",
            )

        raw_question = kwargs.get("question")
        question = raw_question if isinstance(raw_question, str) and raw_question.strip() else None

        # -------------------------------------------------------------------
        # 2. Resolve to an existing file under the project root
        # -------------------------------------------------------------------
        try:
            resolved = resolve_in_root(Path.cwd(), raw_path)
        except ValueError as exc:
            return ToolResult.err(str(exc), code="path-escapes-root")

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.err(
                f"{resolved} does not exist or is not a regular file.",
                code="not-a-file",
            )

        # -------------------------------------------------------------------
        # 3. Read and identify the image
        # -------------------------------------------------------------------
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            return ToolResult.err(
                f"failed to read {raw_path}: {exc}",
                code="read-error",
            )

        mime = _sniff_image_mime(data)
        if mime is None:
            return ToolResult.err(
                f"{raw_path} is not a recognised image file (checked PNG, JPEG, "
                "GIF, WEBP signatures).",
                code="not-an-image",
            )

        # -------------------------------------------------------------------
        # 4. Vision provider and session — both required to bill and ask.
        #    Defense in depth: the real gate is this tool's conditional
        #    registration (modes.CONDITIONAL_TOOLS), which keeps the tool out
        #    of the active set entirely when no vision block is configured.
        # -------------------------------------------------------------------
        vision_provider = vision.vision_client()
        if vision_provider is None:
            return ToolResult.err(
                "no vision provider is configured (config.json has no 'vision' block).",
                code="vision-not-configured",
            )

        active_session = session.current()
        if active_session is None:
            return ToolResult.err(
                "no active session to bill this call against.",
                code="no-session",
            )

        # -------------------------------------------------------------------
        # 5. Ask — exactly one image, in one isolated call
        # -------------------------------------------------------------------
        data_uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        label = f"Image at {raw_path}"
        description = vision.describe_image(active_session, vision_provider, data_uri, label, question=question)
        if description is None:
            return ToolResult.err(
                f"the vision provider returned no usable description for {raw_path}.",
                code="vision-error",
                hint="Retry once; if it keeps failing, the provider itself may be down.",
            )

        return ToolResult.ok(description, image_path=str(resolved), mime_type=mime)
