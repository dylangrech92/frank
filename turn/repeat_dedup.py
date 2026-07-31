from __future__ import annotations

import hashlib



# Success-path loop-guard suffix, APPENDED to the full re-rendered result of a
# repeated identical successful call (repeat #2..#cap). Used whenever the body
# must still be shown — a non-dedupable tool, or a changed / possibly-folded
# result — so the model both sees the output and is told the repeat made no
# progress. The dedup stub below replaces the body entirely when it is provably
# redundant.
_REPEAT_STEER_SUFFIX = (
    "\n\n[loop-guard] {name} was just called with these exact arguments "
    "and succeeded. Repeating the identical successful call makes no "
    "progress. Do not re-issue it. If the result was not what you needed, "
    "change your arguments or approach; otherwise move on or tell the "
    "user you are done."
)

# Dedup stub that REPLACES the full render of a repeated identical successful
# read-only call whose body has not changed since its last full render this
# turn. Re-emitting a byte-identical body wastes context and rewards the
# re-issue; the stub points the model back at the earlier result instead. Only
# used when _repeat_render's two safety conditions hold (see there).
_REPEAT_DEDUP_STUB = (
    "[loop-guard] {name} was already called with these exact arguments this "
    "turn and the result has not changed — output omitted; use the earlier "
    "result above. Do not re-issue this call. If you need something different, "
    "change your arguments or approach."
)

# Dedup stub that REPLACES the full render of a repeated identical successful
# VERIFICATION call (run_command/run_tests/verify_scratch) when nothing has been
# modified since the identical earlier run this turn. Verification is exempt from
# the repeat suffix and the hard cap because the edit→retest cycle legitimately
# repeats — but re-running a byte-identical command with ZERO intervening file
# mutations cannot produce a different result, so its output is omitted and the
# model is told to change something before re-verifying. Separate wording from
# _REPEAT_DEDUP_STUB (world fact + one directive) so the message names the true
# reason (no change on disk, not "read-only re-read"). Only used when the
# fingerprint, compaction count, AND mutation count are all unchanged since the
# last full render (see _repeat_render).
_VERIFY_NOCHANGE_STUB = (
    "[no-change] {name} already ran with these exact arguments and nothing has "
    "been modified since — its output is identical to the result shown above. "
    "Make a change before re-running verification."
)


def _render_fingerprint(rendered: str) -> str:
    """Stable content fingerprint of a rendered tool result.

    Lets the repeat-render dedup tell an *unchanged* repeated read (safe to
    replace with a stub) from one whose body actually differs — e.g. a re-read
    of a file the model just edited, which MUST get the fresh body. Hashing
    keeps the per-key bookkeeping O(1) in memory regardless of render size, and
    ``errors="replace"`` guarantees encoding never raises on odd bytes.
    """
    return hashlib.sha256(rendered.encode("utf-8", "replace")).hexdigest()


def _repeat_render(
    name: str,
    key: tuple[str, str],
    rendered: str,
    seen_renders: dict[tuple[str, str], tuple[str, int, int]],
    compactions: int,
    mutations: int,
    *,
    verification: bool,
) -> str:
    """Return a dedup stub for a provably-redundant repeat, else the full render.

    Called only for a repeat (count >= 2) that is a success on a tool eligible
    for dedup — either a read-only tool (``parallel_safe`` and not
    verification-exempt) or a verification tool (run_command/run_tests/
    verify_scratch). The stamp recorded at the last full render is
    ``key -> (fingerprint, compactions, mutations)``; which of those three
    stamps gate the stub depends on the tool class. Full gate table (a matched
    row omits the body; any mismatch re-emits the full body and re-stamps):

    | tool class   | fingerprint | compaction | mutation  | render on match      |
    |--------------|-------------|------------|-----------|----------------------|
    | read-only    | match       | unchanged  | (ignored) | _REPEAT_DEDUP_STUB   |
    | verification | match       | unchanged  | unchanged | _VERIFY_NOCHANGE_STUB|

    The three conditions each guard a distinct hazard:

    (d) Identical arguments do NOT imply an identical result. A re-read of a
        file the model just edited is legitimate and MUST get the fresh body, so
        the render's content *fingerprint* is compared against the one recorded
        at the last full render; on any mismatch the fresh body is returned.
    (e) A compaction may have folded the earlier result out of context. A stub
        that points at a result no longer present strands the model (the known
        amnesia loop), so the stub is withheld unless ``compactions`` is
        unchanged since the last full render.
    (m) For VERIFICATION only: an identical command whose output is stamp-clean
        can still be worth re-running once a file has changed (the edit→retest
        cycle). So the verification stub is withheld unless the turn's mutation
        count is also unchanged since the last full render — nothing modified
        means the output cannot differ. Read-only tools do NOT gate on (m): a
        re-read whose body is byte-identical is redundant regardless of an
        unrelated edit to some OTHER file, so the mutation stamp is carried on
        the record but never consulted for them (the fingerprint already speaks
        for content).

    A verification tool is EXEMPT from the no-progress suffix (a legitimate
    retest is progress, not a no-op loop), so its full-render branch returns the
    body plain; a read-only tool's carries ``_REPEAT_STEER_SUFFIX``.

    ``seen_renders`` is mutated in place: it is re-stamped with the current
    ``(fingerprint, compactions, mutations)`` on every full-render return so the
    NEXT repeat compares against the most recent body/context/mutation-count —
    in particular, after a verification mutation-stamp mismatch this lets a later
    identical run with no further mutations stub again. Kept as a small pure
    function so an eval can drive every branch with a fabricated stamp dict.
    """
    fingerprint = _render_fingerprint(rendered)
    prev = seen_renders.get(key)
    if verification:
        # (d) AND (e) AND (m): body, context, and disk all unchanged -> omit.
        matched = prev is not None and prev == (fingerprint, compactions, mutations)
        stub = _VERIFY_NOCHANGE_STUB
        full = rendered  # verification is exempt from the no-progress suffix
    else:
        # (d) AND (e) only; the mutation stamp is carried but not consulted.
        matched = (
            prev is not None
            and prev[0] == fingerprint
            and prev[1] == compactions
        )
        stub = _REPEAT_DEDUP_STUB
        full = rendered + _REPEAT_STEER_SUFFIX.format(name=name)
    if matched:
        return stub.format(name=name)
    # Mismatch on a gating condition: re-emit the body and re-stamp so the next
    # repeat compares against the fresh/re-materialised result and its stamps.
    seen_renders[key] = (fingerprint, compactions, mutations)
    return full
