"""Debug Adapter Protocol manager - owns breakpoint registry, session lifecycle, and stop state."""

from __future__ import annotations

import os
import queue
from typing import Any

from dap.client import DAPClient


class DebugUnavailableError(Exception):
    """Raised when a debug adapter is not configured or cannot start."""


class DAPManager:
    """The ONLY stateful debug subsystem.

    Owns the breakpoint registry (settable before OR during a session),
    the live DAPClient handle, and the current stop state (thread id, frame id,
    cached stack) which PERSISTS BETWEEN CALLS.
    """

    def __init__(self, config_adapters: dict, root_path: str) -> None:
        """Initialise the debug manager.

        Args:
            config_adapters: The ``debug_adapters`` mapping from config.json.
            root_path: Absolute path to the project root directory.
        """
        self._config_adapters = config_adapters
        self._root = os.path.realpath(root_path)
        self._client: DAPClient | None = None
        self._breakpoints: dict[str, list[dict]] = {}
        self._thread_id: int | None = None
        self._frame_id: int | None = None
        self._frames: list[dict] = []
        self._caps: dict = {}
        self._stop_queue: queue.Queue = queue.Queue()
        self._synthesizers: dict[str, Any] = {}
        self._synthesizers["python"] = self._synthesize_python

    @property
    def active(self) -> bool:
        """Whether there is a live debug session."""
        return self._client is not None and self._client.alive

    def set_breakpoint(self, file: str, line: int, condition: str | None = None) -> dict:
        """Register or replace a breakpoint at *file:*line with optional *condition*.

        Args:
            file: File path (relative to root or absolute).
            line: Line number.
            condition: Optional hit-condition expression string.

        Returns:
            Dict describing the registered breakpoint and verification status.
        """
        if os.path.isabs(file):
            abspath = os.path.realpath(file)
        else:
            abspath = os.path.realpath(os.path.join(self._root, file))

        bp_data: dict[str, Any] = {"line": line, "condition": condition}

        if abspath not in self._breakpoints:
            self._breakpoints[abspath] = [bp_data]
        else:
            found_idx: int | None = None
            for idx, entry in enumerate(self._breakpoints[abspath]):
                if entry["line"] == line:
                    found_idx = idx
                    break
            if found_idx is not None:
                self._breakpoints[abspath][found_idx] = bp_data
            else:
                self._breakpoints[abspath].append(bp_data)

        verified: bool | None = None
        if self.active:
            verified = self._send_breakpoints_for(abspath)

        return {
            "file": abspath,
            "line": line,
            "condition": condition,
            "verified": verified,
            "active": self.active,
        }

    def clear_breakpoint(self, file: str, line: int | None = None) -> int:
        """Remove breakpoint(s) at *file*, optionally restricted to one *line*.

        Args:
            file: File path (relative or absolute).
            line: If provided, clear only this line. If ``None``, clear all lines in file.

        Returns:
            Count of breakpoints removed.
        """
        if os.path.isabs(file):
            abspath = os.path.realpath(file)
        else:
            abspath = os.path.realpath(os.path.join(self._root, file))

        removed = 0

        if line is None:
            removed = len(self._breakpoints.pop(abspath, []))
        else:
            entries = self._breakpoints.get(abspath, [])
            original_count = len(entries)
            new_entries = [e for e in entries if e["line"] != line]
            removed = original_count - len(new_entries)
            if not new_entries:
                self._breakpoints.pop(abspath, None)
            else:
                self._breakpoints[abspath] = new_entries

        if self.active:
            self._send_breakpoints_for(abspath)

        return removed

    def list_breakpoints(self) -> list[dict]:
        """Return a flat list of every registered breakpoint across all files."""
        result: list[dict] = []
        for abspath, entries in self._breakpoints.items():
            for entry in entries:
                result.append({
                    "file": abspath,
                    "line": entry["line"],
                    "condition": entry.get("condition"),
                })
        return result

    def start(self, target: str | None = None, config: dict | None = None, language: str = "python") -> dict:
        """Start a debug adapter session for *language*, optionally debugging *target*.

        Args:
            target: File to run/debug (used for synthesizing a launch config).
            config: Raw launch configuration; if given supersedes synthesis.
            language: Key into ``_config_adapters`` dict.

        Returns:
            Dict with ``state``, ``reason``, and optionally ``location`` and ``stack``.

        Raises:
            DebugUnavailableError: When no adapter is configured for *language* or binary not found.
        """
        cmd_entry = self._config_adapters.get(language)
        if cmd_entry is None:
            raise DebugUnavailableError(f"debug adapter not configured for {language}")

        if self._client is not None:
            self.stop()

        command = cmd_entry["command"]
        try:
            self._client = DAPClient.spawn_stdio(command, cwd=self._root, name=f"debug-{language}")
        except FileNotFoundError as exc:
            raise DebugUnavailableError(f"debug adapter binary not found: {command[0]}") from exc

        self._client.on_event("stopped", lambda body: self._stop_queue.put(("stopped", body)))
        self._client.on_event("terminated", lambda body: self._stop_queue.put(("terminated", body)))
        self._client.on_event("exited", lambda body: None)

        # Pinned sequence - debugpy defers `initialized` until after launch.
        caps = self._client.request(
            "initialize",
            {
                "clientID": "coding-agent",
                "adapterID": language,
                "linesStartAt1": True,
                "columnsStartAt1": True,
                "pathFormat": "path",
                "supportsRunInTerminalRequest": True,
                "supportsStartDebuggingRequest": True,
            },
            timeout=15.0,
        )
        self._caps = caps

        launch_config = config if config is not None else self._synthesize(target, language)
        launch_seq = self._client.send_request_nowait("launch", launch_config)
        self._client.wait_event("initialized", timeout=15.0)
        self._send_all_breakpoints()
        self._client.request("configurationDone", {}, timeout=15.0)
        self._client.get_response(launch_seq, timeout=20.0)

        return self._await_stop(timeout=20.0)

    def control(self, action: str) -> dict:
        """Run a debug control action (continue / step_over / step_into / step_out / pause).

        Args:
            action: One of ``"continue"``, ``"step_over"``, ``"step_into"``, ``"step_out"``, ``"pause"``.

        Returns:
            Dict with updated stop state or terminated state.
        """
        mapping = {
            "continue": "continue",
            "step_over": "next",
            "step_into": "stepIn",
            "step_out": "stepOut",
            "pause": "pause",
        }

        if action not in mapping:
            raise ValueError(f"unknown debug action: {action}")

        if not self.active:
            raise DebugUnavailableError("no active debug session")

        self._client.request(mapping[action], {"threadId": self._thread_id}, timeout=15.0)
        return self._await_stop(timeout=20.0)

    def evaluate(self, expression: str) -> dict:
        """Evaluate *expression* in the frame captured from the last stop event.

        Args:
            expression: The source-expression string to evaluate at this breakpoint.

        Returns:
            Dict with ``result`` and ``type`` keys from DAP evaluate response.
        """
        if not self.active:
            raise DebugUnavailableError("no active debug session")

        body = self._client.request(
            "evaluate",
            {"expression": expression, "frameId": self._frame_id, "context": "repl"},
            timeout=15.0,
        )
        return {
            "result": body.get("result"),
            "type": body.get("type"),
        }

    def variables(self, scope: str | None = None) -> dict:
        """Return variable scopes for the current frame.

        Args:
            scope: If given, only include scopes whose name contains this substring (case-insensitive).

        Returns:
            Dict mapping scope names to their variable dicts (name -> value).
        """
        if not self.active:
            raise DebugUnavailableError("no active debug session")

        scopes_body = self._client.request(
            "scopes", {"frameId": self._frame_id}, timeout=15.0
        )
        out: dict[str, Any] = {}
        for sc in scopes_body.get("scopes", []):
            name = sc.get("name", "?")
            ref = sc.get("variablesReference", 0)
            if scope is not None and scope.lower() not in name.lower():
                continue
            if not ref:
                out[name] = {}
                continue
            vbody = self._client.request(
                "variables", {"variablesReference": ref}, timeout=15.0
            )
            out[name] = {v.get("name"): v.get("value") for v in vbody.get("variables", [])}
        return out

    def stack(self) -> list[dict]:
        """Return the cached frames from the most recent stop event."""
        return [
            {
                "id": f.get("id"),
                "name": f.get("name"),
                "file": (f.get("source") or {}).get("path"),
                "line": f.get("line"),
            }
            for f in self._frames
        ]

    def stop(self) -> dict:
        """Gracefully tear down the debug adapter session."""
        if self._client is not None:
            try:
                self._client.shutdown()
            finally:
                self._client = None

        self._thread_id = None
        self._frame_id = None
        self._frames = []

        while True:
            try:
                self._stop_queue.get_nowait()
            except queue.Empty:
                break

        return {"state": "terminated"}

    # ------------------------------------------------------------------ private helpers

    def _synthesize(self, target: str | None, language: str) -> dict:
        """Use a registered synthesiser function to build a launch config for *language*."""
        syn_fn = self._synthesizers.get(language)
        if syn_fn is None:
            raise DebugUnavailableError(f"no launch-config synthesizer for {language}")
        if target is None:
            raise DebugUnavailableError("debug_start needs a target or a config")
        return syn_fn(target)

    def _synthesize_python(self, target: str) -> dict:
        """Build a debugpy ``launch`` config from a *target* path."""
        program = os.path.realpath(os.path.join(self._root, target)) if not os.path.isabs(target) else target
        return {
            "type": "python",
            "request": "launch",
            "program": program,
            "cwd": self._root,
            "console": "internalConsole",
            "justMyCode": False,
            "stopOnEntry": False,
        }

    def _send_all_breakpoints(self) -> None:
        """Push all registered breakpoints to a live adapter."""
        for abspath in list(self._breakpoints):
            self._send_breakpoints_for(abspath)

    def _send_breakpoints_for(self, abspath: str) -> bool | None:
        """Send ``setBreakpoints`` for *abspath*; return verified status of last entry (or ``None``)."""
        entries = list(self._breakpoints.get(abspath, []))
        bps: list[dict[str, Any]] = []
        for e in entries:
            bp: dict[str, Any] = {"line": e["line"]}
            if e.get("condition"):
                bp["condition"] = e["condition"]
            bps.append(bp)

        body = self._client.request(
            "setBreakpoints",
            {"source": {"path": abspath}, "breakpoints": bps},
            timeout=15.0,
        )
        verified_list = list(body.get("breakpoints", []))
        if not verified_list:
            return None
        last_entry = verified_list[-1]
        return bool(last_entry.get("verified", False))

    def _await_stop(self, timeout: float) -> dict:
        """Block until the debug adapter sends a ``stopped`` or ``terminated`` event."""
        try:
            name, body = self._stop_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("debug adapter produced no stop/termination event")

        if name == "terminated":
            self._thread_id = None
            self._frame_id = None
            self._frames = []
            return {"state": "terminated", "reason": "terminated"}

        # stopped
        self._thread_id = body.get("threadId") or self._thread_id
        st_body = self._client.request(
            "stackTrace",
            {"threadId": self._thread_id, "startFrame": 0, "levels": 20},
            timeout=15.0,
        )
        self._frames = st_body.get("stackFrames", [])
        self._frame_id = self._frames[0].get("id") if self._frames else None

        loc: dict[str, Any] | None = None
        if self._frames:
            top = self._frames[0]
            src = top.get("source") or {}
            loc = {
                "file": src.get("path"),
                "line": top.get("line"),
            }

        return {
            "state": "stopped",
            "reason": body.get("reason"),
            "location": loc,
            "thread_id": self._thread_id,
            "stack": self.stack(),
        }
