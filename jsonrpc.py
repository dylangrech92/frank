"""JSON-RPC message framing using Content-Length headers.

This module provides protocol-agnostic framing for JSON-RPC messages as defined by
the base protocols used by LSP and DAP. It serialises objects to UTF-8, wraps them
in a ``Content-Length`` header block, and reads framed messages back from a binary
stream — with no assumptions about the transport layer (pipes, sockets, etc.).

This module contains NO LSP-specific or DAP-specific logic; it is intentionally
protocol-agnostic so that it can serve every JSON-RPC consumer in this codebase.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, BinaryIO, Callable


def write_message(
    stream: BinaryIO,
    obj: Any,
    lock: threading.Lock | None = None,
) -> None:
    """Serialize *obj* and write one framed JSON-RPC message to *stream*.

    The message is written as a ``Content-Length`` header block followed by the
    UTF-8 encoded JSON body. When *lock* is provided, the entire write+flush is
    wrapped in the lock so concurrent writers cannot interleave frames.

    Args:
        stream: A binary (writable) stream to write the framed message to.
        obj: A JSON-serialisable object (dict, list, str, int, float, bool, None).
        lock: Optional :class:`threading.Lock`. When given, held around the full
            write+flush to prevent interleaving with other writers.
    """
    body_bytes = json.dumps(obj).encode("utf-8")
    header = f"Content-Length: {len(body_bytes)}\r\n\r\n".encode("ascii")

    if lock is not None:
        with lock:
            stream.write(header)
            stream.write(body_bytes)
            stream.flush()
    else:
        stream.write(header)
        stream.write(body_bytes)
        stream.flush()


def _read_header_line(stream: BinaryIO) -> str | None:
    """Read a single ``Content-Length`` header line (terminated by ``\\r\\n``).

    Returns the stripped line content as a string, or ``None`` if EOF is reached
    before a complete line.

    Args:
        stream: A binary (readable) stream to read from.

    Returns:
        The header line text without ``\\r\\n``, or ``None`` on EOF.
    """
    raw_line: list[bytes] = []
    while True:
        byte = stream.read(1)
        if not byte:
            return None  # EOF before complete line
        if byte == b"\n":
            return b"".join(raw_line).decode("utf-8").rstrip("\r")
        raw_line.append(byte)


def read_message(stream: BinaryIO) -> Any | None:
    """Read one framed JSON-RPC message from *stream*.

    Reads header lines terminated by ``\\r\\n`` until a blank line, parses the case-
    insensitive ``Content-Length`` header (ignoring any other headers), then reads
    exactly that many body bytes (looping until complete because ``stream.read`` may
    return short) and json.loads the decoded payload.

    Args:
        stream: A binary (readable) stream to read from.

    Returns:
        The decoded JSON object on success, or ``None`` when EOF is hit before or
        inside a header block (clean close).

    Raises:
        ValueError: If the header block is malformed or ``Content-Length`` is missing.
    """
    # --- Read the header block until blank line ---
    content_length: int | None = None
    while True:
        line = _read_header_line(stream)
        if line is None:
            return None  # EOF before or inside a header

        if line == "":
            break  # blank line = end of headers

        colon_idx = line.find(":")
        if colon_idx < 0:
            raise ValueError(f"Malformed header line: {line!r}")

        name, _sep, val = line[:colon_idx].strip(), "", line[colon_idx + 1 :].strip()
        if name.lower() == "content-length":
            try:
                content_length = int(val)
            except ValueError:
                raise ValueError(f"Content-Length must be integer, got {val!r}") from None

    # --- Content-Length is mandatory for JSON-RPC framing ---
    if content_length is None:
        raise ValueError("Header block missing required 'Content-Length'")

    # --- Read exactly *content_length* body bytes (handle short reads) ---
    parts: list[bytes] = []
    read_so_far = 0
    while read_so_far < content_length:
        chunk = stream.read(content_length - read_so_far)
        if not chunk:
            raise ValueError(
                f"Stream closed after {read_so_far}/{content_length} body bytes"
            )
        parts.append(chunk)
        read_so_far += len(chunk)

    body_str = b"".join(parts).decode("utf-8")
    return json.loads(body_str)


class ReaderThread:
    """Daemon thread that reads framed JSON-RPC messages and dispatches them via a callback.

    Instantiate with a binary *stream*, a ``callback(msg)`` callable, an optional
    thread *name* for debugging, and an optional *on_eof* hook called when the stream
    closes cleanly (read_message returns None). Call :meth:`start` to launch; use
    :meth:`join` to wait for exit.

    Callback exceptions are caught and printed to stderr so a misbehaving handler
    cannot terminate the reader thread.  Exceptions originating in ``read_message``
    (e.g. stream closed mid-frame) also end the loop after one stderr line.
    """

    __slots__ = ("_stream", "_callback", "_on_eof", "_thread")

    def __init__(
        self,
        stream: BinaryIO,
        callback: Callable[[Any], None],
        name: str = "jsonrpc-reader",
        on_eof: Callable[[], None] | None = None,
    ) -> None:
        """Initialise a new *ReaderThread*.

        Args:
            stream: A binary (readable) stream yielding framed messages.
            callback: Invoked with every decoded message object. Must not raise.
            name: Thread name used in stderr diagnostics and thread naming.
            on_eof: Optional callable invoked once ``read_message`` returns ``None``.
        """
        self._stream = stream
        self._callback = callback
        self._on_eof = on_eof

        def _run() -> None:
            while True:
                # read_message may raise on stream errors (closed mid-frame)
                try:
                    msg = read_message(self._stream)
                except Exception as exc:
                    print(f"ReaderThread({name}): {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    break  # loop ends; thread will exit

                if msg is None:
                    # Clean EOF — callback finished normally.
                    if self._on_eof is not None:
                        self._on_eof()
                    break  # loop ends; thread will exit

                try:
                    self._callback(msg)
                except Exception as exc:
                    print(f"ReaderThread({name}): {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

        self._thread = threading.Thread(
            target=_run,
            daemon=True,
            name=name,
        )

    def start(self) -> None:
        """Launch the reader thread."""
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the reader thread to finish.

        Args:
            timeout: Maximum seconds to block (``None`` blocks forever).  Passed through
                to :meth:`threading.Thread.join`.
        """
        self._thread.join(timeout=timeout)
