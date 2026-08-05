"""Per-stage RSS + heap probe for scale 1..16, in isolated processes."""
import resource, sys, gc, tracemalloc

baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def rss_kib_above():
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - baseline_kb) / 1024.0

from grind import config, generator, parse, dedup, enrich, aggregate

# Parse sys.argv[1] as --scale, default to 4
scale = int(sys.argv[1]) if len(sys.argv) > 1 else 4
legacy_on = sys.argv[2].lower() == "1" if len(sys.argv) > 2 else False
config.ENABLE_LEGACY_RECONCILE = legacy_on

tracemalloc.start(5)
seed = config.SEED_BASE ^ (scale * 0x1000003)
raw = generator.generate_events(scale, seed)
rss_gen = rss_kib_above()

gc.collect()
parsed, rejected = parse.parse_events(raw)
rss_parse = rss_kib_above()

if config.ENABLE_LEGACY_RECONCILE:
    legacy_res = aggregate  # placeholder just to exercise the import
    from grind import legacy
    triples = legacy.reconcile_batch_legacy(parsed)
    rss_legacy = rss_kib_above()
else:
    dc = dedup.mark_duplicates(parsed)
    rss_dedup = rss_kib_above()

if not config.ENABLE_LEGACY_RECONCILE:
    dc = dedup.mark_duplicates(parsed)
    cache = {}
    enrich.enrich_events(parsed, cache)
    rss_enrich = rss_kib_above()
    del cache
    aggregate.build_report(parsed, rejected, dc, 0)
    rss_aggregate = rss_kib_above()
    print(f"{scale},{len(raw)},{dc},{rejected:.0f},{rss_gen:.0f},{rss_parse:.0f},{rss_dedup:.0f},{rss_enrich:.0f},{rss_aggregate:.0f},0,,0")
else:
    dc = dedup.mark_duplicates(parsed)
    print(f"{scale},{len(raw)},{dc},{rejected:.0f},{rss_gen:.0f},{rss_parse:.0f},{rss_kib_above():.0f},0,0,{rss_legacy:.0f},1,{len(triples)}")

