## Diagnosis

Root cause: `queueworks/stats.py`'s `slowest_jobs()` calls
`get_completion_log()` and then sorts the result in place with
`records.sort(key=lambda r: r.duration, reverse=True)`.
`get_completion_log()` (in `store.py`) hands the module-level
`_recent_completions` list back by reference, not a copy -- it is the exact
same list object every other consumer reads. So the moment anything calls
`stats.slowest_jobs()`, the shared completion log is permanently reordered
by duration, and every later call to `chronological_report()` -- which reads
that same list expecting completion order -- renders jobs sorted by
duration instead of the order they actually finished in. There is no
exception anywhere; the report just quietly comes out wrong, two modules
away from where the mutation happened.

## Fix

Have `slowest_jobs()` sort a copy instead of mutating the shared list in
place, e.g. `sorted(get_completion_log(), key=lambda r: r.duration,
reverse=True)`, so generating stats never has a side effect on the
underlying completion history.
