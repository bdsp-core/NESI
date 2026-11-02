#!/usr/bin/env python3
"""SPECTRA mapping sensitivity (Supplementary Tables 6 and 7).

Reads retrain/mapping_M<n>_withhold_overlap_seed2026/test_scores.csv from
retrain_split_fix.py --mapping n (n = 1..24, numbered as in spectra_mappings.py), all
trained on the corrected split with seed 2026. Every mapping is scored on the same test
segments, so one set of patient resamples is shared across mappings: CIs and the paired
differences from M1 come from the same draws. Two-sided bootstrap p-values are Holm-adjusted
across the comparisons within each scale. "M1 repeat" (the universal retrain on the same split
and seed) measures run-to-run training variation and is not in the Holm family.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
RET = ROOT.parent / "retrain"
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
BOOT = 1000
SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
KEY = ["Dataset", "MorgothOutputFilename"]


def rho(a, b):
    return abs(stats.spearmanr(a, b)[0])


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]).clip(max=1)
    out = np.empty_like(adj)
    out[o] = adj
    return out


runs = {}
for n in range(1, 25):
    f = RET / f"mapping_M{n}_withhold_overlap_seed2026" / "test_scores.csv"
    if f.exists():
        runs[n] = pd.read_csv(f, low_memory=False)[KEY + ["BDSPPatientID", "TrueScore", "NewNESI"]]
print(f"{len(runs)} of 24 mappings found: {sorted(runs)}")
assert 1 in runs, "M1 is the reference"
# The universal retrain is the same configuration as M1 (same data, mapping and seed); GPU training is not
# bit-reproducible, so the two differ only by run-to-run variation. Shown as "M1 repeat", outside the Holm family.
rep = pd.read_csv(RET / "withhold_overlap_seed2026" / "test_scores.csv", low_memory=False)[KEY + ["NewNESI"]]

d = runs[1].rename(columns={"NewNESI": "M1"})
for n, r in runs.items():
    if n > 1:
        d = d.merge(r[KEY + ["NewNESI"]].rename(columns={"NewNESI": f"M{n}"}), on=KEY, validate="1:1")
assert len(d) == len(runs[1]), "mappings were scored on different test segments"
d = d.merge(rep.rename(columns={"NewNESI": "M1 repeat"}), on=KEY, validate="1:1")
assert len(d) == len(runs[1])
cols = [f"M{n}" for n in sorted(runs)] + ["M1 repeat"]

rows = []
for s in SCALES:
    g = d[d.Dataset == s]
    rng = np.random.default_rng(2026)
    ids = [np.asarray(v) for v in g.groupby("BDSPPatientID").indices.values()]
    t = g.TrueScore.values
    x = {c: g[c].values for c in cols}
    b = []
    for _ in range(BOOT):
        i = np.concatenate([ids[j] for j in rng.integers(0, len(ids), len(ids))])
        if len(np.unique(t[i])) > 1:
            b.append([rho(t[i], x[c][i]) for c in cols])
    b = pd.DataFrame(b, columns=cols)
    point = {c: rho(t, x[c]) for c in cols}
    p = {}
    for c in cols[1:]:
        dl = b[c] - b["M1"]
        p[c] = min(1.0, 2 * min((dl <= 0).mean(), (dl >= 0).mean()))
    fam = [c for c in p if c != "M1 repeat"]
    adj = dict(zip(fam, holm([p[c] for c in fam]))) if fam else {}
    for c in cols:
        rows.append(dict(
            Mapping=c, Scale=s, n=len(g), patients=len(ids),
            rho=point[c], lo=b[c].quantile(0.025), hi=b[c].quantile(0.975),
            delta=point[c] - point["M1"],
            delta_lo=(b[c] - b["M1"]).quantile(0.025), delta_hi=(b[c] - b["M1"]).quantile(0.975),
            p=p.get(c, np.nan), p_holm=adj.get(c, np.nan),
            better=("" if c == "M1" else (c if point[c] > point["M1"] else "M1")),
        ))

r = pd.DataFrame(rows)
r.round(4).to_csv(OUT / "mapping_summary.csv", index=False)
w = r.pivot(index="Mapping", columns="Scale", values="rho").reindex(cols)[SCALES]
print(w.round(3).to_string())
sig = r[r.p_holm < 0.05]
print("\nHolm p < 0.05:")
print(sig[["Mapping", "Scale", "rho", "delta", "p_holm", "better"]].round(3).to_string(index=False) if len(sig) else "none")
