"""In-memory diagnostics store with publish tracking and settle semantics."""

import os
import time
import threading
from collections import Counter
from pathlib import Path

from lsp.manager import uri_to_path

# LSP DiagnosticTag.Deprecated (the other defined tag, 1, is Unnecessary and
# is not surfaced here).
_TAG_DEPRECATED = 2

# Files named in a delta summary before it collapses to "+N more", mirroring
# turn/lint_delta.py's _LINT_DELTA_CAP guard against flooding context.
_SUMMARY_FILE_CAP = 3


def _issue_key(uri: str, diagnostic: dict) -> tuple | None:
    """Identity for delta tracking, or ``None`` when *diagnostic* is not counted.

    Severity ``1`` and a missing severity are errors, ``2`` is a warning; ``3``
    (information) and ``4`` (hint) never reach a summary and are keyed ``None``
    so they are skipped by both the snapshot and the diff.

    Deliberately excludes the line and column. An edit that inserts or removes
    lines shifts every diagnostic below it, and a pre-existing error that merely
    moved is not a new one -- keying on position would report the whole tail of
    the file as new after any length-changing edit. Callers count keys as a
    multiset, so a genuinely added second copy of an identical message still
    shows up as one new issue.
    """
    severity = diagnostic.get("severity")
    if severity not in (1, 2, None):
        return None
    return (
        uri,
        severity,
        str(diagnostic.get("code", "")),
        diagnostic.get("message", ""),
    )


def _relative_path(uri: str) -> str:
    """Render *uri* as a project-root-relative path, falling back to *uri* itself.

    Tools resolve every path against ``Path.cwd()`` as the project root (see
    ``tools/_sandbox.py``); this matches that so a summary names files the same
    way the rest of the transcript does.

    A language server may serve files outside the root (a vendored dependency,
    a stdlib stub). Those keep their absolute path -- a ``../../../..`` chain is
    strictly less readable than what it resolves to.
    """
    try:
        path = uri_to_path(uri)
    except Exception:
        return uri
    try:
        relative = os.path.relpath(path, Path.cwd())
    except ValueError:  # different drive on Windows
        return path
    return path if relative.startswith("..") else relative


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

    def _issue_counts(self) -> Counter:
        """Multiset of every counted diagnostic currently stored.

        Caller must hold ``self.condition``.
        """
        counts: Counter = Counter()
        for uri, diags in self._diags.items():
            for diag in diags:
                key = _issue_key(uri, diag)
                if key is not None:
                    counts[key] += 1
        return counts

    def snapshot_issues(self) -> Counter:
        """Capture the pre-dispatch baseline a later ``summary`` diffs against.

        Returns:
            A multiset of ``_issue_key`` values for every counted diagnostic in
            the store right now.
        """
        with self.condition:
            return self._issue_counts()

    def summary(self, baseline: Counter) -> str | None:
        """Report only the diagnostics that appeared since *baseline*.

        A delta, not a full report: a project's pre-existing diagnostic noise is
        never injected, so the counts the model is shown always describe what
        its own edits just did. This is the same contract
        ``turn/lint_delta.py`` holds for linters, and for the same reason -- an
        unattributable project-wide total attached to a successful write reads
        as feedback on that write, and sends the model hunting for errors it did
        not cause.

        Files anywhere in the project are covered, not just the edited ones, so
        an edit that breaks a *different* file still surfaces.

        Args:
            baseline: A snapshot from ``snapshot_issues`` taken before the
                round's tool calls were dispatched.

        Returns:
            A formatted string like ``"⚠ 2 new errors in app/Http/Foo.php"``,
            naming up to ``_SUMMARY_FILE_CAP`` files with a ``"+N more"`` tail,
            and an extra ``", N [deprecated]"`` count when any new diagnostic
            carries the LSP ``DiagnosticTag.Deprecated`` tag. ``None`` when this
            round introduced no new diagnostics.
        """
        with self.condition:
            remaining = self._issue_counts() - baseline
            if not remaining:
                return None

            errors = 0
            warnings = 0
            deprecated = 0
            files: list[str] = []
            seen_files: set[str] = set()

            for uri, diags in self._diags.items():
                for diag in diags:
                    key = _issue_key(uri, diag)
                    if key is None or remaining[key] <= 0:
                        continue
                    remaining[key] -= 1

                    if diag.get("severity") == 2:
                        warnings += 1
                    else:
                        errors += 1

                    if is_deprecated(diag):
                        deprecated += 1

                    path = _relative_path(uri)
                    if path not in seen_files:
                        seen_files.add(path)
                        files.append(path)

            parts: list[str] = []

            if errors > 0:
                error_word = "error" if errors == 1 else "errors"
                parts.append(f"{errors} new {error_word}")

            if warnings > 0:
                warning_word = "warning" if warnings == 1 else "warnings"
                parts.append(f"{warnings} new {warning_word}")

            # The deprecated count rides with the other counts, before "in": after
            # the file list it reads as one more file name.
            if deprecated > 0:
                parts.append(f"{deprecated} [deprecated]")

            shown = files[:_SUMMARY_FILE_CAP]
            where = ", ".join(shown)
            if len(files) > len(shown):
                where += f" +{len(files) - len(shown)} more"

            return "⚠ " + ", ".join(parts) + f" in {where}"

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
