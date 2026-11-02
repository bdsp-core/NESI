#!/usr/bin/env python3
"""Leave-one-cohort-out retrains (Supplementary Fig. 35 and its text).

Reads retrain/loco_<scale>_withhold_overlap_seed2026/{heldout,test}_scores.csv from
retrain_split_fix.py --leave-out, and compares each with the universal retrain on the
same split and seed (retrain/withhold_overlap_seed2026). CIs resample patients.
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
RET = ROOT.parent / "retrain"
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output"
(BUILD / "tables").mkdir(parents=True, exist_ok=True)
(BUILD / "figures").mkdir(parents=True, exist_ok=True)
BOOT = 1000
SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
LABEL = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}


def rho(a, b):
    return abs(stats.spearmanr(a, b)[0])


def boot(g, cols):
    """Patient bootstrap of rho(TrueScore, col) for each col; returns (B, len(cols))."""
    rng = np.random.default_rng(2026)
    ids = [np.asarray(v) for v in g.groupby("BDSPPatientID").indices.values()]
    t = g.TrueScore.values
    x = [g[c].values for c in cols]
    out = []
    for _ in range(BOOT):
        i = np.concatenate([ids[j] for j in rng.integers(0, len(ids), len(ids))])
        if len(np.unique(t[i])) > 1:
            out.append([rho(t[i], v[i]) for v in x])
    return np.array(out)


uni = pd.read_csv(RET / "withhold_overlap_seed2026" / "test_scores.csv", low_memory=False)
key = ["Dataset", "MorgothOutputFilename"]
rows, panels = [], {}
for s in SCALES:
    d = RET / f"loco_{s}_withhold_overlap_seed2026"
    h = pd.read_csv(d / "heldout_scores.csv", low_memory=False)
    b = boot(h, ["NewNESI"])[:, 0]
    full = dict(rho=rho(h.TrueScore, h.NewNESI), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                n=len(h), patients=h.BDSPPatientID.nunique())
    t = pd.read_csv(d / "test_scores.csv", low_memory=False)
    t = t[t.Dataset == s].merge(uni[key + ["NewNESI"]].rename(columns={"NewNESI": "UniNESI"}), on=key)
    bb = boot(t, ["NewNESI", "UniNESI"])
    dl = bb[:, 1] - bb[:, 0]
    rows.append(dict(
        Scale=s, full_n=full["n"], full_patients=full["patients"],
        full_rho=full["rho"], full_lo=full["lo"], full_hi=full["hi"],
        test_n=len(t), test_patients=t.BDSPPatientID.nunique(),
        loco_rho=rho(t.TrueScore, t.NewNESI), loco_lo=np.percentile(bb[:, 0], 2.5), loco_hi=np.percentile(bb[:, 0], 97.5),
        uni_rho=rho(t.TrueScore, t.UniNESI), uni_lo=np.percentile(bb[:, 1], 2.5), uni_hi=np.percentile(bb[:, 1], 97.5),
        delta=rho(t.TrueScore, t.UniNESI) - rho(t.TrueScore, t.NewNESI),
        delta_lo=np.percentile(dl, 2.5), delta_hi=np.percentile(dl, 97.5),
    ))
    panels[s] = (h, t)

r = pd.DataFrame(rows)
r.round(4).to_csv(BUILD / "tables" / "loco_summary.csv", index=False)
print(r.round(3).to_string(index=False))

# Supplementary Fig. 33 (submitted as 35): NESI by clinical score, full excluded cohort (A-D) and matched test set (E-H).
COLOR = {"RASS": "#7BAFD4", "GCS": "#F5B85C", "CAMS": "#3DB8A4", "ICANS": "#C97BB0"}
fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
for j, s in enumerate(SCALES):
    rr = r.set_index("Scale").loc[s]
    for i, g in enumerate(panels[s]):
        a = ax[i, j]
        lv = sorted(g.TrueScore.unique())
        bp = a.boxplot([g.NewNESI[g.TrueScore == v] for v in lv], patch_artist=True, widths=0.6,
                       tick_labels=[f"{v:g}\nn={(g.TrueScore == v).sum()}" for v in lv],
                       medianprops=dict(color="black"), flierprops=dict(marker=".", markersize=2, markerfacecolor="gray", markeredgecolor="gray", alpha=0.4))
        for b in bp["boxes"]:
            b.set_facecolor(COLOR[s])
        a.tick_params(axis="x", labelsize=7, rotation=45 if s == "GCS" else 0)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", ls="--", alpha=0.3)
        a.set_xlabel(f"{LABEL[s]} score"); a.set_ylabel("NESI")
        rv, lo, hi = (rr.full_rho, rr.full_lo, rr.full_hi) if i == 0 else (rr.loco_rho, rr.loco_lo, rr.loco_hi)
        a.set_title(f"{'ABCDEFGH'[4 * i + j]}. {LABEL[s]}\nρ = {rv:.2f} ({lo:.2f}–{hi:.2f})", fontsize=11, fontweight="bold")
fig.text(0.5, 0.005, "Row 1 (A–D): each model trained without that scale, evaluated on the full excluded cohort. "
         "Row 2 (E–H): the same models on the held-out test set.\nBoxes: median and IQR; whiskers: 1.5×IQR; n: EEG observations per score. "
         "Spearman's ρ with 95% CI from 1,000 bootstrap resamples of patients.", ha="center", va="bottom", fontsize=10, style="italic")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(BUILD / "figures" / "SuppFig33_leave_one_cohort_out.png", dpi=200)
