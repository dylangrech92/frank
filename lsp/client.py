"""Generic JSON-RPC-over-stdio client for one Language Server protocol process."""

from __future__ import annotations

import itertools
import json
import os
import pathlib
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from jsonrpc import ReaderThread, write_message


# Module-level tracing toggle; set CODING_AGENT_LSP_TRACE=1 to enable.
TRACE = os.environ.get("CODING_AGENT_LSP_TRACE") == "1"


@dataclass
class _Pending:
    """A pending LSP request awaiting a server response."""

    event: threading.Event = field(default_factory=threading.Event, repr=False)
    response: dict[str, Any] | None = None


class LSPClient:
    """JSON-RPC over stdio client for a single Language Server protocol process.

    Spawns the server as a subprocess, runs a background reader thread to decode
    framed JSON-RPC messages from stdout, and dispatches them to pending-request
    waiters or notification handlers.

    Attributes:
        name: Human-readable label for this server (default: ``command[0]``).
        server_capabilities: Capabilities returned by the ``initialize`` handshake,
            or ``None`` before/during teardown.
    """

    def __init__(self, command: list[str], root_path: str, name: str | None = None) -> None:
        """Initialise the LSP client and start communicating with the server process.

        Args:
            command: The argv used to spawn the language-server (e.g. ``["pyright-langserver", "--stdio"]``).
            root_path: Absolute path to the project root; passed as the working directory
                for the subprocess and as the ``rootUri`` in the handshake.
            name: Human label for tracing/diagnostics. Defaults to ``command[0]``.
        """
        self.name = name or command[0]
        self._lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._notifications: dict[str, Callable[[Any], None]] = {}
        self._next_id: itertools.count[int] = itertools.count(1)

        # Start the server process; FileNotFoundError propagates to caller.
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,  # type: ignore[arg-type]
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=root_path,
        )

        root_uri = pathlib.Path(root_path).as_uri()

        def _on_eof() -> None:
            """Waken every pending waiter when the server closes its output stream."""
            for pending in self._pending.values():
                pending.event.set()
            self._pending.clear()

        def _on_message(msg: dict[str, Any]) -> None:
            """Route an incoming JSON-RPC message to the correct handler.

            Args:
                msg: The decoded JSON-RPC message dictionary from the server.
            """
            # --- Response (has 'id' plus 'result' or 'error') ---
            if "id" in msg and ("result" in msg or "error" in msg):
                resp_id = int(msg["id"])
                pending = self._pending.pop(resp_id, None)
                if pending is None:
                    print(f"lsp-trace {self.name}: unknown response id={resp_id}", file=sys.stderr, flush=True)
                    return
                pending.response = msg
                pending.event.set()
                return

            # --- Server-to-client request (has 'id' and 'method') MUST be answered ---
            if "id" in msg and "method" in msg:
                method = msg["method"]
                params = msg.get("params")
                result: Any = None  # default response value

                if method == "workspace/configuration":
                    items = params.get("items", []) if params else []
                    result = [{} for _ in items]  # type: ignore[assignment]
                elif method in ("client/registerCapability", "client/unregisterCapability"):
                    pass  # result stays None
                elif method == "window/workDoneProgress/create":
                    pass  # result stays None
                else:
                    pass  # result stays None for anything else

                reply = {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": result,
                }
                try:
                    write_message(self._proc.stdin, reply, self._lock)
                except (BrokenPipeError, OSError):
                    pass  # server likely dying; don't crash readers
                return

            # --- Notification (has 'method' only) ---
            handler = self._notifications.get(msg["method"])
            if handler is not None:
                params = msg.get("params")
                try:
                    handler(params)
                except Exception as exc:
                    print(
                        f"lsp-trace {self.name}: notification handler error: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

        # Tracing envelope wrapping _on_message
        if TRACE:
            original_callback = _on_message

            def tracing_callback(msg: dict[str, Any]) -> None:  # type: ignore[override]
                """Print a receive-trace line then delegate to the real router.

                Args:
                    msg: The decoded JSON-RPC message dictionary.
                """
                print(
                    f"lsp-trace {self.name} <- {json.dumps(msg)}",
                    file=sys.stderr,
                    flush=True,
                )
                original_callback(msg)  # type: ignore[operator]

            reader = ReaderThread(self._proc.stdout, tracing_callback, name=f"lsp-{self.name}", on_eof=_on_eof)
        else:
            reader = ReaderThread(self._proc.stdout, _on_message, name=f"lsp-{self.name}", on_eof=_on_eof)

        self._reader = reader
        self._root_path = root_path
        reader.start()

    # ------------------------------------------------------------------ public

    def _send(
        self, body: dict[str, Any], *, traced: bool = True
    ) -> None:
        """Write a JSON-RPC envelope to the server, dead-process guarded.

        Args:
            body: The message dictionary to send.
            traced: If ``TRACE`` is enabled, emit a trace line before sending.
        """
        if self._proc.poll() is not None:
            raise RuntimeError("server process is dead")

        if TRACE and traced:
            print(
                f"lsp-trace {self.name} -> {json.dumps(body)}",
                file=sys.stderr,
                flush=True,
            )

        try:
            write_message(self._proc.stdin, body, self._lock)
        except (BrokenPipeError, OSError) as exc:
            print(f"lsp-trace {self.name}: write error: {exc}", file=sys.stderr, flush=True)
            raise RuntimeError("server process is dead") from None

    def request(
        self,
        method: str,
        params: Any,
        timeout: float = 10.0,
    ) -> Any:
        """Send a JSON-RPC request and wait for the response.

        Args:
            method: The LSP method name (e.g. ``"textDocument/hover"``).
            params: Parameters to pass with the request.
            timeout: Seconds to wait before raising :exc:`TimeoutError`.

        Returns:
            The ``"result"`` value from the server response dictionary.

        Raises:
            TimeoutError: If no response arrives within *timeout* seconds.
            RuntimeError: If a non-null ``"error"`` key is in the response, or
                the server process has died.
        """
        req_id = next(self._next_id)
        pending = _Pending()
        self._pending[req_id] = pending

        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        try:
            self._send(body)
        except RuntimeError:
            self._pending.pop(req_id, None)
            raise

        fired = pending.event.wait(timeout=timeout)
        if not fired:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"request timed out after {timeout}s: {method} on {self.name}")

        response = pending.response
        if response is None:
            raise RuntimeError(f"No response received for {method} on {self.name}")

        if "error" in response:
            err = response["error"]
            code = err.get("code")
            message = err.get("message", "")
            raise RuntimeError(
                f"LSP error for {method}: code={code}, message={message}"
            )

        return response.get("result")

    def notify(self, method: str, params: Any) -> None:
        """Send a fire-and-forget JSON-RPC notification to the server.

        Args:
            method: The LSP notification method name (e.g. ``"textDocument/didOpen"``).
            params: Parameters for the notification.
        """
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            self._send(body)
        except RuntimeError:
            pass  # server is dead; caller already knows

    def on_notification(self, method: str, callback: Callable[[Any], None]) -> None:
        """Register a handler for a server notification method.

        Args:
            method: The LSP notification verb (e.g. ``"textDocument/publishDiagnostics"``).
            callback: A callable accepting a single *params* argument. Invoked on every
                matching message from the server, inside its own try/except so handler
                errors cannot terminate the reader thread.
        """
        self._notifications[method] = callback

    def initialize(self) -> dict[str, Any]:
        """Perform the LSP handshake: request ``initialize``, then notify ``initialized``.

        Sends an ``initialize`` request with standard capabilities and the project root URI,
        waits 30 seconds for a response (some servers are slow to start), then notifies
        ``initialized`` with an empty params object. Stores the server response as
        :attr:`server_capabilities`.

        Returns:
            The ``result`` dict returned by the server's ``initialize`` method.
        """
        root_uri = pathlib.Path(self._root_path).as_uri()
        base_name = os.path.basename(root_uri.split("://")[1].rstrip("/"))

        params = {  # type: ignore[assignment]
            "processId": os.getpid(),
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": base_name}],
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {},
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "synchronization": {"didSave": True},
                },
                "workspace": {"configuration": True, "workspaceFolders": True},
            },
        }

        result = self.request("initialize", params, timeout=30.0)  # type: ignore[arg-type]
        if result is None:  # type: ignore[unreachable]
            raise RuntimeError("initialize returned null")

        self.server_capabilities = result  # type: ignore[attr-defined]
        self.notify("initialized", {})  # type: ignore[arg-type]
        return result

    def shutdown(self) -> None:
        """Best-effort server teardown.

        Requests ``shutdown``, notifies ``exit``, terminates/wait/kills the process.
        Never raises an exception.
        """
        try:
            self.request("shutdown", None, timeout=5.0)  # type: ignore[arg-type]
        except Exception:  # noqa: E722
            pass

        try:
            self.notify("exit", {})  # type: ignore[arg-type]
        except Exception:  # noqa: E722
            pass

        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pylint: disable=no-member
                self._proc.kill()

    @property
    def alive(self) -> bool:
        """Whether the server process is still running (``poll() is None``)."""
        return self._proc.poll() is None


__all__ = ["LSPClient"]
