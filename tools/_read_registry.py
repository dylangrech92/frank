"""Per-process stale-read registry for the H2 write guard.

Multiple agent processes may operate on the same project directory
concurrently.  This registry lets write tools detect when a file changed on
disk since *this session* last read it, so they can refuse to blindly
overwrite another process's changes.

State is per-process (a module-level dict) -- each agent process is exactly
one session, so there is no need for cross-process persistence here.

Exports
-------
record_read : stamp a resolved absolute path with its current mtime_ns/size
check_fresh : compare a resolved absolute path against its recorded stamp
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

# Maps a resolved absolute path (as str) to the (mtime_ns, size) stamp
# recorded the last time this session read (or wrote) it.
_READS: dict[str, tuple[int, int]] = {}

# record_read runs inside read_file, which is parallel_safe and may execute
# on a ThreadPoolExecutor alongside other reads; guard all registry access.
_LOCK = threading.Lock()


def record_read(path: str | Path) -> None:
    """Stamp *path* with its current on-disk mtime_ns/size.

    Call this after a successful read AND after a successful write (so the
    session's own write does not make the file look stale for its own next
    edit).

    Args:
        path: The already-resolved absolute path that was just read or written.
    """
    key = str(Path(path))
    try:
        st = os.stat(key)
    except OSError:
        # Nothing to stamp if the path vanished (e.g. deleted); no-op.
        return
    with _LOCK:
        _READS[key] = (st.st_mtime_ns, st.st_size)


def check_fresh(path: str | Path) -> str:
    """Check whether *path* still matches this session's recorded read stamp.

    Args:
        path: The already-resolved absolute path to check.

    Returns:
        ``'fresh'`` if a stamp is recorded and the current mtime_ns/size match,
        ``'stale'`` if a stamp is recorded but the file changed since, or
        ``'unread'`` if this session never recorded a read of the path.
    """
    key = str(Path(path))
    with _LOCK:
        stamp = _READS.get(key)
    if stamp is None:
        return 'unread'
    try:
        st = os.stat(key)
    except OSError:
        # The file vanished since it was read -- treat as stale, not fresh.
        return 'stale'
    return 'fresh' if (st.st_mtime_ns, st.st_size) == stamp else 'stale'


def forget(path: str | Path) -> None:
    """Drop any recorded stamp for *path*, e.g. after it is deleted or moved.

    Args:
        path: The already-resolved absolute path to forget.
    """
    with _LOCK:
        _READS.pop(str(Path(path)), None)
