"""acceptance — runs truth/<id>/acceptance/probe_<band>_<slug>.py against a
fresh temp copy of the post-run tree, band-weighted (core/edge/adversarial).

Spec fields:

    band_weights     dict[str, float], default {"core": 100} — must sum to
                      100. Every band present under truth/<id>/acceptance/
                      must be declared here, and every declared band must
                      have at least one probe on disk — a mismatch is a
                      truth-authoring bug and fails loudly rather than
                      silently rescaling.
    probe_timeout_s  int, default 120 — per-probe subprocess timeout.

Each probe runs as ``python3 probe_x.py <tree_copy>``; exit 0 = pass. A
fresh ``shutil.copytree`` of the post-run tree is made per probe (contract:
"probes can't dirty grading"), so a probe that writes into the tree it was
given cannot affect any other probe or grader.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PROBE_RE = re.compile(r"^probe_(core|edge|adversarial)_.+\.py$")


def grade(spec: dict, ctx) -> dict:
    if ctx.truth_dir is None:
        raise ValueError("acceptance grader requires a truth dir")
    probes_dir = ctx.truth_dir / "acceptance"
    if not probes_dir.is_dir():
        raise ValueError(f"missing acceptance/ dir under {ctx.truth_dir}")

    band_weights = spec.get("band_weights", {"core": 100.0})
    if abs(sum(band_weights.values()) - 100.0) > 1e-6:
        raise ValueError(f"acceptance band_weights {band_weights} must sum to 100")

    probes_by_band: dict[str, list[Path]] = {}
    for path in sorted(probes_dir.glob("probe_*.py")):
        m = _PROBE_RE.match(path.name)
        if not m:
            raise ValueError(f"probe {path.name} does not match 'probe_<band>_<slug>.py'")
        probes_by_band.setdefault(m.group(1), []).append(path)

    if not probes_by_band:
        raise ValueError(f"no acceptance probes found under {probes_dir}")

    unknown_bands = sorted(set(probes_by_band) - set(band_weights))
    if unknown_bands:
        raise ValueError(f"probes exist for band(s) {unknown_bands} not present in band_weights {band_weights}")
    for band in band_weights:
        if band not in probes_by_band:
            raise ValueError(f"band_weights declares {band!r} but no probes found for it under {probes_dir}")

    timeout_s = spec.get("probe_timeout_s", 120)
    probe_results = []
    band_scores: dict[str, float] = {}

    for band, probes in probes_by_band.items():
        passed = 0
        for probe in probes:
            probe_scratch = Path(tempfile.mkdtemp(prefix="bench-probe-"))
            probe_tree = probe_scratch / "tree"
            try:
                shutil.copytree(ctx.tree, probe_tree)
                proc = subprocess.run(
                    [sys.executable, str(probe), str(probe_tree)],
                    cwd=str(probe.parent),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                ok = proc.returncode == 0
                if ok:
                    passed += 1
                probe_results.append({
                    "band": band,
                    "probe": probe.name,
                    "passed": ok,
                    "returncode": proc.returncode,
                    "stderr_excerpt": "" if ok else proc.stderr[-500:],
                })
            except subprocess.TimeoutExpired:
                probe_results.append({
                    "band": band,
                    "probe": probe.name,
                    "passed": False,
                    "returncode": None,
                    "stderr_excerpt": f"timed out after {timeout_s}s",
                })
            finally:
                shutil.rmtree(probe_scratch, ignore_errors=True)
        band_scores[band] = passed / len(probes)

    score = sum(band_scores[band] * weight for band, weight in band_weights.items())
    score = max(0.0, min(100.0, score))

    return {
        "score": score,
        "details": {
            "band_scores": band_scores,
            "pass_fraction": score / 100.0,
            "probes": probe_results,
        },
    }
