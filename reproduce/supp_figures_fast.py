#!/usr/bin/env python3
"""Supplementary Figs. 2, 24, 25 and 36, from the per-segment score file."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(os.environ["NESI_DATA_ROOT"])
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
f = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
f["pid"] = f.BDSPPatientID.astype(str).str.replace(r"\.0$", "", regex=True)
SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
LABEL = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}
SEVERE = {"RASS": lambda s: s <= -4, "GCS": lambda s: s <= 8, "CAMS": lambda s: s >= 6, "ICANS": lambda s: s >= 3}

# Supp Fig 2: class-wise counts
fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
for a, ds in zip(ax, SCALES):
    v = f[f.Dataset == ds].RawScore.value_counts().sort_index()
    a.bar(v.index.astype(int).astype(str), v.values, color="#4C78A8")
    a.set(title=f"{LABEL[ds]} (n = {v.sum():,})", ylabel="Segments", xlabel="Score")
fig.tight_layout()
fig.savefig(OUT / "SuppFig2_class_distributions.png", dpi=200)

# Supp Figs 24-25: logistic calibration fitted on train and validation, assessed on test
fig24, ax24 = plt.subplots(1, 4, figsize=(14, 3.4))
fig25, ax25 = plt.subplots(1, 4, figsize=(14, 3.4))
rows = []
for i, ds in enumerate(SCALES):
    d = f[f.Dataset == ds]
    tr, te = d[d.Split != "Test"], d[d.Split == "Test"]
    y0, y1 = SEVERE[ds](tr.RawScore).astype(int), SEVERE[ds](te.RawScore).astype(int)
    m = LogisticRegression().fit(tr[["NESI"]], y0)
    p = m.predict_proba(te[["NESI"]])[:, 1]
    fp, tp, th = roc_curve(y0, tr.NESI)
    cut = th[np.argmax(tp - fp)]
    hit = te.NESI >= cut
    obs, pred = calibration_curve(y1, p, n_bins=10 if ds in ("RASS", "GCS") else 5, strategy="quantile")
    ax24[i].plot(pred, obs, "o-")
    ax24[i].plot([0, 1], [0, 1], "k:")
    ax24[i].set(title=LABEL[ds], xlabel="Predicted probability", ylabel="Observed proportion")
    fp1, tp1, _ = roc_curve(y1, te.NESI)
    auc = roc_auc_score(y1, te.NESI)
    ax25[i].plot(fp1, tp1, label=f"AUROC {auc:.3f}")
    ax25[i].plot(hit[y1 == 0].mean(), hit[y1 == 1].mean(), "ro", label=f"NESI ≥ {cut:.3f}")
    ax25[i].plot([0, 1], [0, 1], "k:")
    ax25[i].set(title=LABEL[ds], xlabel="1 − specificity", ylabel="Sensitivity")
    ax25[i].legend(loc="lower right", fontsize=8)
    rows.append(dict(Scale=ds, threshold=round(cut, 3), sensitivity=round(hit[y1 == 1].mean(), 3), specificity=round(1 - hit[y1 == 0].mean(), 3), auroc=round(auc, 3)))
fig24.tight_layout()
fig24.savefig(OUT / "SuppFig24_calibration.png", dpi=200)
fig25.tight_layout()
fig25.savefig(OUT / "SuppFig25_roc.png", dpi=200)
print(pd.DataFrame(rows).to_string(index=False))

# Supp Fig 36: AUROC at each ordinal cut-point
fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
for a, ds in zip(ax, SCALES):
    te = f[(f.Dataset == ds) & (f.Split == "Test")]
    levels = np.sort(te.RawScore.unique())
    xs, ys = [], []
    for k in levels[:-1]:
        y = (te.RawScore <= k).astype(int) if ds in ("RASS", "GCS") else (te.RawScore > k).astype(int)
        if 0 < y.mean() < 1:
            xs.append(str(int(k)))
            ys.append(roc_auc_score(y, te.NESI))
    a.plot(xs, ys, "o-", color="#4C78A8")
    a.set(title=LABEL[ds], xlabel="Cut-point k", ylabel="AUROC", ylim=(0.5, 1))
fig.tight_layout()
fig.savefig(OUT / "SuppFig36_threshold_auroc.png", dpi=200)
print("written to", OUT)
