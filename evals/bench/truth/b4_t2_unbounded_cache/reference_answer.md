# grind memory report (workload.py --scale N)

Method: `tracemalloc`, snapshotting immediately before and after the
enrichment stage (`enrich_events`) at scale 1, 5, 10, and 20, plus a
before/after diff to attribute growth to specific allocation sites.

## Root cause: `grind/enrich.py`'s profile cache almost never hits (3,114 calls to `build_profile()` at scale 1)

`enrich_events(events, cache)` in `grind/enrich.py` builds a per-client
profile once per distinct `event.user_agent` and caches it in a plain
`dict` (`cache[event.user_agent] = profile`) so repeat clients in the same
batch skip the rebuild. The cache has **no eviction and no size cap** --
entries grow **20x** alongside the 20x input growth (3,114 at scale 1 to
62,266 at scale 20), because nothing ever removes an entry.

That would be fine if `user_agent` were a low-cardinality key -- but
measured cache hit rate is only 86 of 3,200 calls (~2.7%), constant across
scale (2.69% at scale 1, 2.70% at scale 5, 2.71% at scale 10 and 20)
because it embeds a per-record random build number
(`GrindBrowser/{major}.{minor} (build-{6-digit-random})`), making it
effectively a unique key on almost every record. The only records that
ever hit the cache are the synthetic exact-duplicate records the pipeline
itself injects (same count as `duplicates_flagged` in the report), not
genuine repeat clients. In other words: **~97.3% of records create a
brand-new cache entry**, so the cache grows almost 1:1 with input size
instead of staying bounded by a small set of real clients -- it is a
memoization cache in name only.

## Measured growth (bytes, `tracemalloc.get_traced_memory()` peak during
the enrichment stage only)

| scale | records | cache entries | enrich-stage peak | cache container (dict+keys+values), approx |
|---|---|---|---|---|
| 1  | 3,200  | 3,114  | 1,773,588 B (~1,732 KiB / 1.7 MB) | ~3,674,448 B (~3.5 MB) |
| 5  | 16,000 | 15,568 | 8,751,950 B (~8,548 KiB / 8.3 MB) | ~18,272,220 B (~17.4 MB) |
| 10 | 32,000 | 31,134 | 17,585,784 B (~17,174 KiB / 16.8 MB) | ~36,667,459 B (~35.0 MB) |
| 20 | 64,000 | 62,266 | 35,178,279 B (~34,354 KiB / 33.6 MB) | ~73,339,006 B (~69.9 MB) |

Cache entry count and enrich-stage peak memory both scale **linearly with
input size** -- almost exactly 1x/5x/10x/20x, tracking record count, with
no sign of a ceiling. This is unbounded growth, not a fixed working set:
memory cost is driven by input volume, not by the number of distinct real
clients.

The two largest allocation sites during the enrichment stage (from the
before/after snapshot diff) are both direct consequences of the cache miss
rate: `grind/enrich.py`'s `build_profile()` call to `json.loads()`
(re-decoding each event's metadata JSON to build the cached profile -- at
scale 10 this alone accounts for roughly 6.4 MB of the growth) and the
profile dict construction itself inside `enrich.py` (roughly 8.3 MB at
scale 10). Both are one-time-per-cache-miss costs, so they scale with the
same ~97.3%-miss-rate curve as the entry count.

## What must change

The cache mechanism itself (bounding it, e.g. an LRU with a fixed max
size, or evicting stale entries) would only partially help here: the real
defect is the **key**. Keying the cache by user_agent *without* the random
per-record build number (e.g. by device/browser/plan, or by stripping the
`(build-######)` suffix before lookup) would restore the intended
low-cardinality reuse and the cache would stop growing unboundedly with
input size. This should be fixed before raising ingestion volume, since
peak memory currently grows in direct proportion to record count with no
cap.
