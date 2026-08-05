# grind performance report (workload.py --scale N)

Method: `cProfile` over `workload.py --scale 1` and `workload.py --scale 10`
(default vs. the planned 10x volume), sorted by both cumulative and internal
(self) time. Numbers below are self (tottime) unless stated otherwise.

## Today's cost (scale 1, 3,200 records, 0.523s profiled / 0.27s unprofiled wall)

1. **`canonicalize_path` in `grind/normalize.py` is the dominant hotspot** --
   0.076s self time, ~14.5% of the profiled 0.523s total, called 3,200 times
   (once per record). This is comfortably the #1 cost by self time; the next
   closest function is roughly 40% smaller. It walks every decoded character
   of every path to build a shape fingerprint and a rolling-hash signature,
   so cost scales with path length x record count.

2. **`json.decoder.raw_decode` (stdlib) is the #2 cost** -- 0.044s self time
   (~8.4% of total), 6,314 calls. The profile attributes this time to
   library frames, but the actionable cause is in fixture code:
   `grind/enrich.py`'s `build_profile()` redundantly re-decodes metadata
   JSON -- 3,114 calls at scale 1 (every event except the 86 cache hits) --
   that `grind/parse.py` already decoded once per record and stored on
   `ParsedEvent.metadata`. Fix: have `build_profile` read `event.metadata`
   instead of re-decoding `event.metadata_json`.

3. **`mark_duplicates` in `grind/dedup.py` is a close #3 at this scale** --
   0.040s self time (~7.6% of total) -- cheap enough today to be easy to
   miss, but see the scale-10 section below.

(Also present but not attributable to fixture code as a fix target: `~0.037s`
in `json.encoder.iterencode` and assorted `random`-module cost inside data
generation itself -- generation is a fixed cost of the workload, not
something the pipeline can optimize away.)

## Cost at 10x volume (scale 10, 32,000 records, 8.4s profiled / 5.8s
unprofiled wall)

The ranking flips. **`mark_duplicates` becomes the dominant cost by a wide
margin**: 3.481s self time, ~41.3% of the profiled 8.422s total -- roughly
9x more expensive in absolute terms than at scale 1, even though record
count only grew 10x, because its cost is **O(n^2)**: it deduplicates by
scanning a plain Python list (`seen: list[tuple] = []`, `if key in seen`)
instead of using a set, so every new record is compared against every prior
record in the batch. This is negligible at scale 1 (#3, ~7.6%) and the #1
cost by a wide margin at scale 10 (~41.3%) -- this needs to be fixed
*before* the 10x volume increase, not after.

`canonicalize_path` and `json.decoder.raw_decode` still cost real time at
scale 10 (0.795s / ~9.4%, and 0.593s / ~7.0%, respectively -- both scaling
roughly linearly with record count, as expected for per-record work) but
neither is the top priority once the quadratic dedup cost is in the
picture.

## Cold decoy -- do not spend time here

`grind/legacy.py`'s `reconcile_batch_legacy` looks like the worst offender
in the codebase on paper (a triple-nested loop over all events, genuinely
cubic-shaped), but it never actually runs: it is only called when
`config.ENABLE_LEGACY_RECONCILE` is true, and that flag is `False` in this
pipeline (`grind/pipeline.py`'s `run()` never enters that branch). A
line-level execution trace over the whole pipeline at both scale 1 and
scale 10 shows zero lines of its function body executed -- it is inert, not
a contributor to current or projected cost, and should not be prioritized
ahead of the dedup fix above.

## Priority order

1. `dedup.py:mark_duplicates` -- fix before the 10x volume increase (O(n^2)
   -> O(n) with a set).
2. `normalize.py:canonicalize_path` -- today's #1 cost; worth a look
   independent of the scale change.
3. `enrich.py:build_profile`'s redundant `json.loads` -- smaller win, but a
   one-line fix (reuse `event.metadata`).
4. `legacy.py:reconcile_batch_legacy` -- not a real cost; skip it.
