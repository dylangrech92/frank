"""In-memory diagnostics store with publish tracking and settle semantics."""

import time
import threading
from lsp.manager import uri_to_path

# LSP DiagnosticTag.Deprecated (the other defined tag, 1, is Unnecessary and
# is not surfaced here).
_TAG_DEPRECATED = 2


def is_deprecated(diagnostic: dict) -> bool:
    """Whether *diagnostic* carries the LSP ``DiagnosticTag.Deprecated`` tag.

    Args:
        diagnostic: A single LSP diagnostic dict, as stored verbatim from
            ``textDocument/publishDiagnostics``.

    Returns:
        ``True`` when the diagnostic's ``tags`` list contains ``2``
        (``DiagnosticTag.Deprecated``).
    """
    return _TAG_DEPRECATED in (diagnostic.get("tags") or [])


class DiagnosticsStore:
    """Thread-safe store for LSP ``textDocument/publishDiagnostics`` output.

    Maps each URI to its latest diagnostic list, counts how many times
    diagnostics have been delivered per URI, and provides a settle mechanism
    so callers can wait until new diagnostics arrive after a file mutation.
    """

    def __init__(self) -> None:
        """Initialise empty store with its guarding condition."""
        self._diags: dict[str, list] = {}
        self._publish_counts: dict[str, int] = {}
        self.condition = threading.Condition()

    def handle_publish(self, params: dict) -> None:
        """Process a ``textDocument/publishDiagnostics`` notification.

        Stores the diagnostics list (replacing any previous diagnostics for
        that URI), bumps the publish count, and wakes all waiters.

        Args:
            params: Dict with ``uri`` (str) and ``diagnostics`` (list of dict).
        """
        uri = params["uri"]
        diagnostics = params["diagnostics"]

        with self.condition:
            self._diags[uri] = diagnostics
            self._publish_counts[uri] = self._publish_counts.get(uri, 0) + 1
            self.condition.notify_all()

    def purge(self, uri: str) -> None:
        """Remove a URI from the store entirely.

        Args:
            uri: The URI to purge.
        """
        with self.condition:
            self._diags.pop(uri, None)
            self._publish_counts.pop(uri, None)
            self.condition.notify_all()

    def snapshot_counts(self, uris: list[str]) -> dict[str, int]:
        """Return current publish counts for *uris*.

        Missing URIs are assumed to have count 0.

        Args:
            uris: URIs to snapshot counts for.

        Returns:
            Dict mapping each URI to its current publish count.
        """
        with self.condition:
            return {uri: self._publish_counts.get(uri, 0) for uri in uris}

    def wait_for_publish(
        self,
        uris: list[str],
        baseline: dict[str, int],
        deadline_seconds: float = 2.0,
    ) -> bool:
        """Block until every URI has a strictly higher publish count than *baseline*.

        A missing entry in *baseline* counts as ``0``. Waits on the internal
        condition with a monotonic deadline of *deadline_seconds* seconds from
        now. Returns when all URIs have republished or the timeout expires.

        Args:
            uris: URIs to wait for.
            baseline: Baseline publish counts per URI.
            deadline_seconds: How long (seconds) to wait.

        Returns:
            ``True`` if all URIs republished, ``False`` on timeout.
        """
        with self.condition:
            deadline = time.monotonic() + deadline_seconds
            while True:
                ok = True
                for uri in uris:
                    current = self._publish_counts.get(uri, 0)
                    base = baseline.get(uri, 0)
                    if current <= base:
                        ok = False
                        break

                if ok:
                    return True

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False

                self.condition.wait(timeout=remaining)

    def summary(self) -> str | None:
        """Aggregate error/warning counts across all stored diagnostics.

        Severity ``1`` counts as errors, ``2`` as warnings; missing severity
        counts as an error; severities ``3`` and ``4`` are ignored entirely.

        Returns:
            A formatted string like ``"3 errors, 2 warnings in 2 files"``,
            with a trailing ``", N deprecated"`` clause when any counted
            diagnostic carries the LSP ``DiagnosticTag.Deprecated`` tag, or
            ``None`` when there are zero errors and zero warnings.
        """
        with self.condition:
            errors = 0
            warnings = 0
            deprecated = 0
            files_with_issues: set[str] = set()

            for uri, diags in self._diags.items():
                file_path = uri_to_path(uri)

                for diag in diags:
                    severity = diag.get("severity")

                    if severity == 1:
                        errors += 1
                        files_with_issues.add(file_path)
                    elif severity == 2:
                        warnings += 1
                        files_with_issues.add(file_path)
                    elif severity is None:
                        errors += 1
                        files_with_issues.add(file_path)
                    else:
                        continue

                    if is_deprecated(diag):
                        deprecated += 1

            if errors == 0 and warnings == 0:
                return None

            parts: list[str] = []

            if errors > 0:
                error_word = "error" if errors == 1 else "errors"
                parts.append(f"{errors} {error_word}")

            if warnings > 0:
                warning_word = "warning" if warnings == 1 else "warnings"
                parts.append(f"{warnings} {warning_word}")

            file_word = "file" if len(files_with_issues) == 1 else "files"
            result = "⚠ " + ", ".join(parts) + f" in {len(files_with_issues)} {file_word}"

            if deprecated > 0:
                result += f", {deprecated} [deprecated]"

            return result

    def full(self, file_filter: str | None = None) -> list[tuple[str, dict]]:
        """Return every stored (uri, diagnostic) pair sorted by uri then line.

        Args:
            file_filter: When non-empty, only include diagnostics whose URI
                resolves to a path ending with the filter string.

        Returns:
            A flat sorted list of ``(uri, diagnostic)`` tuples.
        """
        result: list[tuple[str, dict]] = []

        for uri, diags in self._diags.items():
            if file_filter:
                try:
                    path = uri_to_path(uri)
                    if not path.endswith(file_filter):
                        continue
                except Exception:  # noqa: E722
                    continue

            for diag in diags:
                result.append((uri, diag))

        result.sort(
            key=lambda item: (
                item[0],
                item[1].get("range", {}).get("start", {}).get("line", 0),
            )
        )
        return result


STORE = DiagnosticsStore()
