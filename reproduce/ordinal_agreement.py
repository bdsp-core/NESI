#!/usr/bin/env python3
"""Supp Fig 26: three-class agreement between NESI and each clinical scale.

NESI cutoffs for each scale are midpoints between the median NESI of adjacent
severity classes in that scale's training and validation data. Evaluated on the
manuscript's test set; CIs resample patients.

    python reproduce/ordinal_agreement.py --recompute   # seconds; writes class labels and bootstrap draws
    python reproduce/ordinal_agreement.py               # table and figure from those intermediates
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output"
INTER = ROOT / "intermediates"
SCORES = ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv"
SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
LABEL = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}
SEVERITY = {
    "RASS": lambda s: np.select([s >= -1, s >= -3], [0, 1], 2),
    "GCS": lambda s: np.select([s >= 13, s >= 9], [0, 1], 2),
    "CAMS": lambda s: np.select([s <= 1, s <= 5], [0, 1], 2),
    "ICANS": lambda s: np.select([s <= 0, s <= 2], [0, 1], 2),
}
BOOT = 1000


def metrics(t, p):
    return np.abs(t - p).mean(), cohen_kappa_score(t, p, weights="quadratic", labels=[0, 1, 2])


def recompute():
    INTER.mkdir(exist_ok=True)
    f = pd.read_csv(SCORES, low_memory=False)
    seg, draws, cuts = [], [], []
    for ds in SCALES:
        d = f[f.Dataset == ds]
        dev, te = d[d.Split != "Test"], d[d.Split == "Test"]
        g = SEVERITY[ds](dev.RawScore.values)
        med = [np.median(dev.NESI.values[g == k]) for k in range(3)]
        c = [(med[0] + med[1]) / 2, (med[1] + med[2]) / 2]
        cuts.append(dict(Scale=ds, median_good=med[0], median_moderate=med[1], median_bad=med[2], cut_good_moderate=c[0], cut_moderate_bad=c[1]))
        s = te[["Dataset", "BDSPPatientID", "MorgothOutputFilename", "RawScore", "NESI"]].copy()
        s["TrueClass"], s["NESIClass"] = SEVERITY[ds](te.RawScore.values), np.digitize(te.NESI.values, c)
        seg.append(s)
        ids = {k: np.asarray(v) for k, v in s.groupby("BDSPPatientID").indices.items()}
        keys = list(ids)
        t, p = s.TrueClass.values, s.NESIClass.values
        rng = np.random.default_rng(2026)
        for b in range(BOOT):
            i = np.concatenate([ids[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
            m, q = metrics(t[i], p[i])
            draws.append(dict(Scale=ds, draw=b, maoe=m, qwk=q))
    pd.concat(seg, ignore_index=True).to_csv(INTER / "ordinal_test_classes.csv", index=False)
    pd.DataFrame(draws).to_csv(INTER / "ordinal_bootstrap_draws.csv", index=False)
    pd.DataFrame(cuts).to_csv(INTER / "ordinal_cuts.csv", index=False)


def build():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = pd.read_csv(INTER / "ordinal_test_classes.csv", low_memory=False)
    d = pd.read_csv(INTER / "ordinal_bootstrap_draws.csv")
    rows, cms = [], {}
    for ds in SCALES:
        g, b = s[s.Dataset == ds], d[d.Scale == ds]
        m, q = metrics(g.TrueClass.values, g.NESIClass.values)
        cms[ds] = confusion_matrix(g.TrueClass, g.NESIClass, labels=[0, 1, 2])
        rows.append(dict(Scale=ds, Segments=len(g), Patients=g.BDSPPatientID.nunique(), MAOE=m, MAOE_lo=b.maoe.quantile(0.025), MAOE_hi=b.maoe.quantile(0.975), QWK=q, QWK_lo=b.qwk.quantile(0.025), QWK_hi=b.qwk.quantile(0.975)))
    r = pd.DataFrame(rows)
    r.round(3).to_csv(BUILD / "tables/ordinal_agreement.csv", index=False)
    pd.concat([pd.DataFrame(cms[k], index=["Good", "Moderate", "Bad"], columns=["Good", "Moderate", "Bad"]).assign(Scale=k) for k in SCALES]).to_csv(BUILD / "tables/ordinal_confusion_counts.csv")

    fig = plt.figure(figsize=(10.2, 4.8), dpi=200)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.62, left=0.075, right=0.985, top=0.95, bottom=0.07)
    top = gs[0].subgridspec(1, 4, wspace=0.55)
    names = ["Good", "Moderate", "Bad"]
    for i, ds in enumerate(SCALES):
        ax = fig.add_subplot(top[i])
        n = cms[ds] / cms[ds].sum(1, keepdims=True) * 100
        ax.imshow(n, cmap="Blues", vmin=0, vmax=100)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{n[a, b]:.1f}%", ha="center", va="center", fontsize=9, color="white" if n[a, b] > 60 else "black")
        ax.set_xticks(range(3), names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(3), names, fontsize=9)
        ax.set_title(LABEL[ds], fontweight="bold", fontsize=10)
        ax.set_xlabel("NESI category", fontsize=9)
        if i == 0:
            ax.set_ylabel("True category", fontsize=9)
    bot = gs[1].subgridspec(1, 2, wspace=0.14)
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
    for j, (k, title, ymax) in enumerate([("MAOE", "Mean Absolute Ordinal Error", 0.65), ("QWK", "Quadratic-Weighted Kappa", 1.1)]):
        ax = fig.add_subplot(bot[j])
        v, lo, hi = r[k].values, r[k + "_lo"].values, r[k + "_hi"].values
        ax.bar(range(4), v, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6, width=0.65, yerr=[v - lo, hi - v], capsize=3, error_kw={"linewidth": 1})
        for x in range(4):
            ax.text(x, hi[x] + ymax * 0.02, f"{v[x]:.3f}\n({lo[x]:.3f}–{hi[x]:.3f})", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(4), [LABEL[x] for x in SCALES], fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(0, ymax)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(BUILD / "figures/supp_fig26.png", bbox_inches="tight")
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    (BUILD / "tables").mkdir(parents=True, exist_ok=True), (BUILD / "figures").mkdir(parents=True, exist_ok=True)
    if a.recompute or not (INTER / "ordinal_test_classes.csv").exists():
        recompute()
    build()
