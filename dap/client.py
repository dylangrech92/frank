"""DAP (Debug Adapter Protocol) client: Content-Length framed DAP over stdio or TCP.

Provides request/response correlation (by ``request_seq``), event subscription and
blocking waits, reverse-request handling with a child-session registry, and two
transport constructors. Framing is delegated to the shared :mod:`jsonrpc` module —
DAP uses the exact same ``Content-Length`` envelope as LSP.
"""

from __future__ import annotations

import itertools
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable

import jsonrpc

TRACE = os.environ.get("CODING_AGENT_DAP_TRACE") == "1"


class _Pending:
    """A single in-flight DAP request awaiting its correlated response."""

    __slots__ = ("event", "response")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.response: dict[str, Any] | None = None


class DAPClient:
    """Bidirectional DAP client over a stdio subprocess or a TCP socket.

    Construct via :meth:`spawn_stdio` (local adapters such as debugpy) or
    :meth:`connect_tcp` (adapters in DAP-server mode such as js-debug). Outbound
    messages carry a monotonically increasing ``seq``; responses are correlated to
    their request by ``request_seq``. Events are buffered so a wait started slightly
    late still observes them. Reverse requests from the adapter (``startDebugging``,
    ``runInTerminal``) are answered by registered handlers, defaulting to success.
    """

    def __init__(
        self,
        readable: Any,
        writable: Any,
        *,
        proc: subprocess.Popen[bytes] | None = None,
        sock: socket.socket | None = None,
        name: str = "dap",
    ) -> None:
        """Initialise the client around already-open binary streams and start reading.

        Args:
            readable: Binary readable stream yielding framed messages from the adapter.
            writable: Binary writable stream for framed messages to the adapter.
            proc: The adapter subprocess (stdio transport), or ``None`` for TCP.
            sock: The connected socket (TCP transport), or ``None`` for stdio.
            name: Human label used in reader-thread naming and trace lines.
        """
        self.name = name
        self._readable = readable
        self._writable = writable
        self._proc = proc
        self._sock = sock
        self._closed = False

        self._write_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = itertools.count(1)

        self._pending: dict[int, _Pending] = {}
        self._event_handlers: dict[str, list[Callable[[dict], None]]] = {}
        self._reverse_handlers: dict[str, Callable[[dict], dict]] = {}
        self._recent_events: list[tuple[str, dict]] = []
        self.child_sessions: dict[str, dict] = {}

        # Default reverse-request handlers.
        self.on_reverse_request("startDebugging", self._default_start_debugging)
        self.on_reverse_request("runInTerminal", lambda args: {})

        self._reader = jsonrpc.ReaderThread(
            readable, self._on_message, name=f"dap-{name}", on_eof=self._on_eof
        )
        self._reader.start()

    # ------------------------------------------------------------------ transports

    @classmethod
    def spawn_stdio(
        cls, command: list[str], cwd: str | None = None, name: str | None = None
    ) -> "DAPClient":
        """Spawn *command* as the adapter subprocess and frame over its stdio.

        Args:
            command: argv to launch the adapter (e.g. ``["python","-m","debugpy.adapter"]``).
            cwd: Working directory for the subprocess.
            name: Optional reader-thread label.

        Returns:
            A started :class:`DAPClient` bound to the subprocess.
        """
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=cwd,
        )
        return cls(proc.stdout, proc.stdin, proc=proc, name=name or f"stdio-{proc.pid}")

    @classmethod
    def connect_tcp(cls, host: str, port: int, name: str | None = None) -> "DAPClient":
        """Open a TCP connection to *host:port* and frame over its socket file objects.

        Args:
            host: Adapter host (e.g. ``"127.0.0.1"``).
            port: Adapter DAP-server port.
            name: Optional reader-thread label.

        Returns:
            A started :class:`DAPClient` bound to the socket.
        """
        sock = socket.create_connection((host, port))
        readable = sock.makefile("rb")
        writable = sock.makefile("wb")
        return cls(readable, writable, sock=sock, name=name or f"tcp-{host}:{port}")

    # ------------------------------------------------------------------ outbound

    def _raw_write(self, obj: dict[str, Any]) -> None:
        """Frame and write a fully-formed message (``seq`` already assigned)."""
        if TRACE:
            print(f"dap-trace {self.name} -> {json.dumps(obj)}", file=sys.stderr, flush=True)
        try:
            jsonrpc.write_message(self._writable, obj, self._write_lock)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"DAP transport is dead: {exc}") from None

    def _next_seq(self) -> int:
        """Return the next outbound sequence number (thread-safe)."""
        with self._lock:
            return next(self._seq)

    def _send_request(self, command: str, arguments: dict | None) -> tuple[int, _Pending]:
        """Register a pending entry then write the request; returns ``(seq, pending)``.

        The pending entry is registered under the lock *before* the write so the reader
        thread can never process the response before the waiter exists.
        """
        pending = _Pending()
        with self._lock:
            seq = next(self._seq)
            self._pending[seq] = pending
        obj: dict[str, Any] = {"seq": seq, "type": "request", "command": command}
        if arguments is not None:
            obj["arguments"] = arguments
        self._raw_write(obj)
        return seq, pending

    def request(self, command: str, arguments: dict | None = None, timeout: float = 10.0) -> dict:
        """Send a request and block until its response arrives.

        Args:
            command: DAP command (e.g. ``"initialize"``, ``"stackTrace"``).
            arguments: Optional arguments object; omitted from the wire when ``None``.
            timeout: Seconds to wait for the correlated response.

        Returns:
            The response ``body`` dict (empty dict if absent).

        Raises:
            TimeoutError: No correlated response within *timeout*.
            RuntimeError: Response carried ``success: false``, or the transport is dead.
        """
        seq, pending = self._send_request(command, arguments)
        if not pending.event.wait(timeout):
            with self._lock:
                self._pending.pop(seq, None)
            raise TimeoutError(f"DAP {command} timed out after {timeout}s")
        with self._lock:
            self._pending.pop(seq, None)
        resp = pending.response
        if resp is None or not resp.get("success", False):
            detail = (resp or {}).get("message", "no/failed response")
            raise RuntimeError(f"DAP {command} failed: {detail}")
        return resp.get("body") or {}

    def send_request_nowait(self, command: str, arguments: dict | None = None) -> int:
        """Fire a request without waiting; returns its ``seq`` for :meth:`get_response`."""
        seq, _ = self._send_request(command, arguments)
        return seq

    def get_response(self, seq: int, timeout: float = 10.0) -> dict:
        """Block for the response to a previously sent *seq* (from :meth:`send_request_nowait`).

        Args:
            seq: The sequence number returned by :meth:`send_request_nowait`.
            timeout: Seconds to wait.

        Returns:
            The response ``body`` dict.

        Raises:
            RuntimeError: Unknown *seq*, or the response carried ``success: false``.
            TimeoutError: No response within *timeout*.
        """
        with self._lock:
            pending = self._pending.get(seq)
        if pending is None:
            raise RuntimeError(f"no pending DAP request with seq={seq}")
        if not pending.event.wait(timeout):
            raise TimeoutError(f"DAP response for seq={seq} timed out after {timeout}s")
        with self._lock:
            self._pending.pop(seq, None)
        resp = pending.response
        if resp is None or not resp.get("success", False):
            detail = (resp or {}).get("message", "no/failed response")
            raise RuntimeError(f"DAP request seq={seq} failed: {detail}")
        return resp.get("body") or {}

    # ------------------------------------------------------------------ events

    @property
    def event_cursor(self) -> int:
        """Current length of the event buffer; pass to :meth:`wait_event` as *after_index*.

        Snapshot this before issuing a request whose result is signalled by an event
        (e.g. ``continue`` → ``stopped``) so the wait observes only the *next* event.
        """
        with self._lock:
            return len(self._recent_events)

    def on_event(self, event: str, callback: Callable[[dict], None]) -> None:
        """Register *callback* to fire (with the event body) on every matching event."""
        with self._lock:
            self._event_handlers.setdefault(event, []).append(callback)

    def wait_event(
        self,
        event: str,
        timeout: float,
        predicate: Callable[[dict], bool] | None = None,
        after_index: int = 0,
    ) -> dict:
        """Block until an *event* (optionally matching *predicate*) is observed.

        Scans the event buffer from *after_index* onward, so an event delivered between
        sending a request and calling this method is not missed. Use ``after_index=0`` for
        one-shot events like ``initialized``; use :attr:`event_cursor` for a "next event".

        Args:
            event: Event name to wait for (e.g. ``"stopped"``, ``"terminated"``).
            timeout: Seconds to wait.
            predicate: Optional filter on the event body.
            after_index: Only consider buffered events at or after this index.

        Returns:
            The matching event's body dict.

        Raises:
            TimeoutError: No matching event within *timeout*.
        """
        end = time.monotonic() + timeout
        with self._cond:
            idx = after_index
            while True:
                while idx < len(self._recent_events):
                    name, body = self._recent_events[idx]
                    idx += 1
                    if name == event and (predicate is None or predicate(body)):
                        return body
                remaining = end - time.monotonic()
                if remaining <= 0 or self._closed:
                    raise TimeoutError(f"DAP event '{event}' not received within {timeout}s")
                self._cond.wait(remaining)

    # ------------------------------------------------------------------ reverse requests

    def on_reverse_request(self, command: str, handler: Callable[[dict], dict]) -> None:
        """Register *handler* to answer the adapter's reverse *command* request.

        The handler receives the reverse request's ``arguments`` and returns the response
        ``body`` dict; the client wraps it as a success response with the correct
        ``request_seq``.
        """
        with self._lock:
            self._reverse_handlers[command] = handler

    def _default_start_debugging(self, args: dict) -> dict:
        """Default ``startDebugging`` handler: record the child config, answer success."""
        config = args.get("configuration") or {}
        req_type = args.get("request", "launch")
        with self._lock:
            key = str(len(self.child_sessions))
            self.child_sessions[key] = {"configuration": config, "request": req_type}
        return {}

    # ------------------------------------------------------------------ router

    def _on_message(self, msg: dict[str, Any]) -> None:
        """Classify and route one inbound DAP message (called on the reader thread)."""
        if TRACE:
            print(f"dap-trace {self.name} <- {json.dumps(msg)}", file=sys.stderr, flush=True)
        mtype = msg.get("type")

        if mtype == "response":
            req_seq = msg.get("request_seq")
            if req_seq is None:
                return
            with self._lock:
                pending = self._pending.get(req_seq)
            if pending is not None:
                pending.response = msg
                pending.event.set()
            return

        if mtype == "event":
            name = msg.get("event", "")
            body = msg.get("body") or {}
            with self._lock:
                self._recent_events.append((name, body))
                handlers = list(self._event_handlers.get(name, []))
                self._cond.notify_all()
            for cb in handlers:
                try:
                    cb(body)
                except Exception:  # noqa: BLE001 — a bad handler must not kill the reader
                    print(f"dap/client {self.name}: event handler for '{name}' raised:", file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)
            return

        if mtype == "request":
            cmd = msg.get("command", "")
            args = msg.get("arguments") or {}
            req_seq = msg.get("seq", 0)
            with self._lock:
                handler = self._reverse_handlers.get(cmd)
            body: dict = {}
            if handler is not None:
                try:
                    body = handler(args) or {}
                except Exception:  # noqa: BLE001
                    print(f"dap/client {self.name}: reverse handler for '{cmd}' raised:", file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)
                    body = {}
            reply = {
                "seq": self._next_seq(),
                "type": "response",
                "request_seq": req_seq,
                "success": True,
                "command": cmd,
                "body": body,
            }
            try:
                self._raw_write(reply)
            except RuntimeError:
                pass
            return

        print(f"dap/client {self.name}: unknown message type {mtype!r}", file=sys.stderr, flush=True)

    def _on_eof(self) -> None:
        """Reader hit EOF: wake every waiter so nothing blocks forever."""
        with self._lock:
            pend = list(self._pending.values())
            self._cond.notify_all()
        for p in pend:
            p.event.set()

    # ------------------------------------------------------------------ lifecycle

    @property
    def alive(self) -> bool:
        """Whether the transport is still usable."""
        if self._closed:
            return False
        if self._proc is not None:
            return self._proc.poll() is None
        if self._sock is not None:
            return self._sock.fileno() != -1
        return True

    def shutdown(self) -> None:
        """Best-effort teardown: disconnect, wake waiters, close transport. Never raises."""
        try:
            if self.alive:
                try:
                    self.request("disconnect", {"terminateDebuggee": True}, timeout=2.0)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._closed = True

        with self._lock:
            pend = list(self._pending.values())
        for p in pend:
            p.event.set()
        with self._cond:
            self._cond.notify_all()

        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2)
                    except Exception:  # noqa: BLE001
                        self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
        for closer in (self._sock, self._readable, self._writable):
            if closer is not None:
                try:
                    closer.close()
                except Exception:  # noqa: BLE001
                    pass


__all__ = ["DAPClient"]
