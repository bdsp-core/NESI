#!/usr/bin/env python3
"""Supplementary Fig. 31 and its text: NESI vs conventional qEEG features on each scale.

Reads intermediates/qeeg_per_segment.csv, written by qeeg_baseline_check.py: all CAM-S and
ICANS test segments and a random sample of RASS and GCS test segments. Features per segment
(19 channels, Welch 4-s windows, 0.5-30 Hz, median over channels): relative delta power
(0.5-4 Hz), delta/alpha ratio (8-13 Hz) and SEF95. Scales are oriented so that higher means
more severe (-RASS, 15 - GCS, CAM-S, ICANS); rho keeps its sign. CIs resample patients.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output"
(BUILD / "tables").mkdir(parents=True, exist_ok=True)
(BUILD / "figures").mkdir(parents=True, exist_ok=True)
BOOT = 1000
SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
LABEL = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}
FEATS = [("delta_alpha", "Delta/alpha power ratio"), ("rel_delta", "Relative delta power"),
         ("sef95", "SEF95 (Hz)"), ("NESI", "NESI")]
SEVERITY = {"RASS": lambda s: -s, "GCS": lambda s: 15 - s, "CAMS": lambda s: s, "ICANS": lambda s: s}

d = pd.read_csv(ROOT / "intermediates" / "qeeg_per_segment.csv")
rows = []
for s in SCALES:
    g = d[d.Dataset == s].reset_index(drop=True)
    y = SEVERITY[s](g.RawScore).values
    x = {c: g[c].values for c, _ in FEATS}
    ids = [np.asarray(v) for v in g.groupby("BDSPPatientID").indices.values()]
    rng = np.random.default_rng(2026)
    b = []
    for _ in range(BOOT):
        i = np.concatenate([ids[j] for j in rng.integers(0, len(ids), len(ids))])
        if len(np.unique(y[i])) > 1:
            b.append([stats.spearmanr(y[i], x[c][i])[0] for c, _ in FEATS])
    b = np.array(b)
    for k, (c, _) in enumerate(FEATS):
        rows.append(dict(Scale=s, Feature=c, Segments=len(g), Patients=len(ids), rho=stats.spearmanr(y, x[c])[0],
                         lo=np.percentile(b[:, k], 2.5), hi=np.percentile(b[:, k], 97.5)))
r = pd.DataFrame(rows)
r.round(3).to_csv(BUILD / "tables" / "qeeg_supp_fig31.csv", index=False)
print(r.round(3).to_string(index=False))

fig, ax = plt.subplots(len(FEATS), len(SCALES), figsize=(17, 15))
rr = r.set_index(["Scale", "Feature"])
for i, (c, name) in enumerate(FEATS):
    for j, s in enumerate(SCALES):
        g = d[d.Dataset == s]
        lv = sorted(g.RawScore.unique())
        a = ax[i, j]
        a.boxplot([g.loc[g.RawScore == v, c] for v in lv], flierprops=dict(markersize=1.5, alpha=0.4))
        a.set_xticks(range(1, len(lv) + 1), [f"{v:g}\nn={(g.RawScore == v).sum()}" for v in lv], fontsize=6 if len(lv) > 8 else 7)
        if c in ("delta_alpha",):
            a.set_yscale("log")
        q = rr.loc[(s, c)]
        a.set_title(f"{'ABCD'[i]}. {name}, {LABEL[s]}\nρ = {q.rho:.2f} ({q.lo:.2f} to {q.hi:.2f})", fontsize=10)
        a.set_xlabel(LABEL[s]); a.set_ylabel(name)
fig.tight_layout()
fig.savefig(BUILD / "figures" / "SuppFig31_qeeg.png", dpi=200)
