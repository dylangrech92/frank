"""Language server manager — discovers, starts, and shares LSP client instances."""

from __future__ import annotations

import os
import pathlib
import urllib.parse
import urllib.request
from typing import Any

from lsp.client import LSPClient


# Module-level mapping from file extension (lowercased, including the dot) to LSP language id.
EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".php": "php",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
}


def path_to_uri(path: str) -> str:
    """Convert *path* to a ``file://`` URI.

    Args:
        path: Absolute file path on disk.

    Returns:
        A ``file://`` URI string for the resolved absolute path.
    """
    return pathlib.Path(path).resolve().as_uri()


def uri_to_path(uri: str) -> str:
    """Convert a ``file://`` URI back to an Absolute path string.

    Args:
        uri: A ``file://`` URI string.

    Returns:
        The resolved absolute file path string.
    """
    parts = urllib.parse.urlparse(uri)
    path_component = parts.path
    return urllib.request.url2pathname(path_component)


class LSPUnavailableError(Exception):
    """Raised when a requested language server is not available."""

    pass


class LSPManager:
    """Manage Language Server Protocol client lifecycle.

    Dispatches per-language LSPClient instances with lazy spawning, caching
    of failures for the session, and shared process deduplication for servers
    that use identical command lists (e.g. css/scss).
    """

    def __init__(self, config_servers: dict[str, Any], root_path: str) -> None:
        """Initialise the manager.

        Args:
            config_servers: The ``language_servers`` mapping loaded from config.json.
            root_path: Absolute path to the project root directory.
        """
        self._config_servers = config_servers
        self._root_path = os.path.realpath(root_path)
        # language -> LSPClient for running servers.
        self._clients: dict[str, LSPClient] = {}
        # language -> reason string for servers that could not start (cached).
        self._failed: dict[str, str] = {}
        # Internal cache keyed by tuple of command args to ensure shared instances.
        self._command_cache: dict[tuple[str, ...], LSPClient] = {}
        # URI -> version number for open documents.
        self._open_docs: dict[str, int] = {}
        # List of callables taking a URI string, invoked when a document must be purged.
        self._purge_callbacks: list[callable] = []
        # Optional callback taking a client, invoked after client.initialize() succeeds.
        self.on_client_start: callable | None = None

        # Wire LSPManager into the file-mutation event bus.
        from tools import _sandbox  # pylint: disable=import-outside-toplevel
        _sandbox.subscribe_mutations(self.handle_mutation)

    # ------------------------------------------------------------------ public

    def language_for_path(self, path: str) -> str | None:
        """Return the LSP language id for *path*, or ``None`` when unknown.

        Args:
            path: File path whose extension (lowercased) is looked up in EXTENSION_LANGUAGES.

        Returns:
            The matching language id string, or ``None`` if the extension is unmapped.
        """
        ext = pathlib.Path(path).suffix.lower()
        return EXTENSION_LANGUAGES.get(ext)

    def get_client(self, language: str, *, spawn: bool = True) -> LSPClient:
        """Ensure a running client for *language*, returning it or raising ``LSPUnavailableError``.

        Args:
            language: The LSP language id (e.g. ``"python"``).
            spawn: When ``True`` (default), spawns a new server process if none is
                already cached or previously failed.  When ``False``, returns the
                cached running client for *language* without spawning — raises
                ``LSPUnavailableError`` with message ``"server not running for <lang>"``
                when no pre-warmed or previously-requested client exists.

        Returns:
            An :class:`LSPClient` instance wired to the configured server process.

        Raises:
            LSPUnavailableError: When no command is configured for *language*, the binary
                is not found, or the server failed during initialization.
        """
        # Already running?
        if language in self._clients:
            return self._clients[language]

        # Failed earlier in this session — do not retry.
        if language in self._failed:
            raise LSPUnavailableError(self._failed[language])

        if not spawn:
            raise LSPUnavailableError(f"server not running for {language}")

        # Is this language even configured?
        config_entry = self._config_servers.get(language)
        if config_entry is None:
            raise LSPUnavailableError(f"no language server configured for {language}")

        command: list[str] = config_entry["command"]
        cmd_key = tuple(command)

        # Reuse an existing client that shares the same command (e.g. css + scss).
        if cmd_key in self._command_cache:
            shared_client = self._command_cache[cmd_key]

            # The shared client might have been created for a different language key;
            # register it under this language too if its key is not already mapped.
            if language not in self._clients:
                self._clients[language] = shared_client
            return shared_client

        try:
            client = LSPClient(command, self._root_path, name=language)
            client.initialize()
        except FileNotFoundError as exc:
            reason = f"server not installed: {command[0]} not found"
            self._failed[language] = reason
            cache_failed_languages: list[str] = []
            for k, v in self._config_servers.items():
                if tuple(v["command"]) == cmd_key:
                    cache_failed_languages.append(k)
            for lang in cache_failed_languages:
                if lang not in self._failed:
                    self._failed[lang] = reason
            raise LSPUnavailableError(reason) from exc
        except Exception as exc:
            reason = f"server failed to start: {exc}"
            self._failed[language] = reason
            cache_failed_languages: list[str] = []
            for k, v in self._config_servers.items():
                if tuple(v["command"]) == cmd_key:
                    cache_failed_languages.append(k)
            for lang in cache_failed_languages:
                if lang not in self._failed:
                    self._failed[lang] = reason
            raise LSPUnavailableError(reason) from exc

        # Success: store under both the command cache and language-specific dict.
        self._clients[language] = client
        self._command_cache[cmd_key] = client

        if self.on_client_start is not None:
            self.on_client_start(client)

        return client

    def detect_languages(self) -> set[str]:
        """Walk *root_path* and collect configured languages whose extensions appear in files.

        Returns:
            A set of language id strings discovered under the project tree. Directories named
            ``.git``, ``node_modules``, ``__pycache__``, ``.venv``, and ``venv`` are skipped.
            Symlinked directories that escape the project root are also skipped.
        """
        scanned_extensions: set[str] = set()

        for dirpath, dirnames, _filenames in os.walk(self._root_path):
            # Skip unwanted directories.
            dirnames[:] = [
                d for d in dirnames
                if d not in (".git", "node_modules", "__pycache__", ".venv", "venv")
            ]

            # Skip symlinked dirs that escape the project root.
            real_dirpath = os.path.realpath(dirpath)
            kept: list[str] = []
            for d in dirnames[:]:
                candidate = os.path.join(dirpath, d)
                if os.path.islink(candidate):
                    real_child = os.path.realpath(candidate)
                    if not real_child.startswith(os.path.realpath(self._root_path)):
                        continue
                kept.append(d)
            dirnames[:] = kept

            for fname in _filenames:
                ext = pathlib.Path(fname).suffix.lower()
                if ext:
                    scanned_extensions.add(ext)

        return {
            lang
            for ext, lang in EXTENSION_LANGUAGES.items()
            if ext in scanned_extensions
        }

    def prewarm(self) -> list[str]:
        """Detect languages in the project and attempt to start each corresponding server.

        Returns:
            A sorted list of human-readable status lines (sorted by language name), one per
            detected language, in the format ``language-server <lang>: <status>``.
        """
        detected = self.detect_languages()
        statuses: list[str] = []

        for lang in sorted(detected):
            try:
                client = self.get_client(lang)
                cmd = ", ".join(self._config_servers[lang]["command"])
                statuses.append(f"language-server {lang}: running ({cmd})")
            except LSPUnavailableError as exc:
                statuses.append(
                    f"language-server {lang}: unavailable — {exc}"
                )

        return statuses

    def shutdown_all(self) -> None:
        """Shutdown every managed client and clear internal state.

        Deduplicates shared instances so each server process is terminated once.
        """
        distinct_clients = list(set(self._clients.values()))
        self._clients.clear()
        for client in distinct_clients:
            try:
                client.shutdown()
            except Exception:  # noqa: E722
                pass

    def on_purge(self, callback: callable) -> None:
        """Register *callback* to be invoked with a URI whenever diagnostics must be discarded.

        Args:
            callback: A callable accepting a single URI string argument.
        """
        self._purge_callbacks.append(callback)

    def _client_for_file(self, path: str) -> LSPClient | None:
        """Return a running client for *path*, or ``None`` when no pre-warmed server exists.

        Does **not** trigger a new server spawn for unmapped languages — uses ``spawn=False``
        so that only pre-warmed and previously-requested servers receive document sync.

        Args:
            path: Absolute file path on disk.

        Returns:
            An :class:`LSPClient` instance, or ``None`` when the extension is unknown or
            no server is running for that language.
        """
        language = self.language_for_path(path)
        if language is None:
            return None

        try:
            return self.get_client(language, spawn=False)
        except LSPUnavailableError:
            return None

    def _did_open(self, path: str) -> None:
        """Send ``textDocument/didOpen`` for *path* to its language server.

        Also records the URI and version in ``_open_docs``.  Reads the file text from disk;
        errors are silently ignored so a dead file never breaks an event branch.

        Args:
            path: Absolute file path on disk.
        """
        try:
            text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        uri = path_to_uri(path)
        language = self.language_for_path(path)  # type: ignore[assignment]
        self._open_docs[uri] = 1

        client = self.get_client(language, spawn=False)
        client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language,
                    "version": 1,
                    "text": text,
                }
            },
        )

    def _did_change(self, path: str) -> None:
        """Send ``textDocument/didChange`` (full-document sync) for *path*.

        Bumps the stored version number.  If the URI is not yet open delegates to ``_did_open``.

        Args:
            path: Absolute file path on disk.
        """
        uri = path_to_uri(path)
        language = self.language_for_path(path)  # type: ignore[assignment]

        if uri not in self._open_docs:
            self._did_open(path)
            return

        self._open_docs[uri] += 1
        n = self._open_docs[uri]

        try:
            text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        client = self.get_client(language, spawn=False)
        client.notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": n},
                "contentChanges": [{"text": text}],
            },
        )

    def _did_close(self, path: str) -> None:
        """Send ``textDocument/didClose`` for *path* and invoke purge callbacks.

        Removes the URI from ``_open_docs`` if present.  **Always** calls every registered
        purge callback with the URI because stale diagnostics may exist regardless of open state.

        Args:
            path: Absolute file path on disk.
        """
        uri = path_to_uri(path)
        language = self.language_for_path(path)  # type: ignore[assignment]

        if uri in self._open_docs:
            del self._open_docs[uri]
            client = self.get_client(language, spawn=False)
            client.notify(
                "textDocument/didClose",
                {"textDocument": {"uri": uri}},
            )

        for callback in self._purge_callbacks:
            try:
                callback(uri)
            except Exception:  # noqa: E722
                pass

    def handle_mutation(self, event: dict) -> None:
        """Bus subscriber — route file-mutation events to the appropriate did* handler.

        Matches the exact ``kind`` strings and ``extra`` payload keys emitted by the file tools:

        * ``created`` -- from ``tools.create_file.CreateFile`` (no extra keys)
        * ``changed`` -- from ``tools.update_file.UpdateFile`` (no extra keys)
        * ``deleted`` -- from ``tools.delete_file.DeleteFile`` (no extra keys)
        * ``renamed`` -- from ``tools.move_file.MoveFile`` with extra key **``old_path``**

        Every branch is wrapped so an exception inside LSP sync never breaks the tool that
        emitted the event; only a single stderr line is printed on failure.

        Args:
            event: A dict with at least ``kind`` (str) and ``path`` (str) keys.
        """
        kind = event["kind"]
        path = event["path"]

        try:
            if kind == "created":
                client = self._client_for_file(path)
                if client is not None:
                    self._did_open(path)

            elif kind == "changed":
                client = self._client_for_file(path)
                if client is not None:
                    self._did_change(path)

            elif kind == "deleted":
                self._did_close(path)

            elif kind == "renamed":
                old_path = event["extra"]["old_path"]

                self._did_close(old_path)

                post_client = self._client_for_file(path)
                if post_client is not None:
                    self._did_open(path)
        except Exception as exc:  # noqa: E722
            import sys
            print(f"lsp-mutation-error: {exc}", file=sys.stderr, flush=True)
