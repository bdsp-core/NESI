#!/usr/bin/env python3
"""Paired Wilcoxon signed-rank tests for the supplement's 5-fold comparison (Supp Fig 6).

Compares each deep model with each classical baseline, per scale and metric, using
the per-fold results the figure plots. With 5 pairs the smallest possible p is
0.0625 two-sided and 0.03125 one-sided.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "github_moved/cohort_models"
NAMES = {
    "RASS": ("{}_5fld_{}_Results_6classRASS.pkl", {"ResNet-GAP": "ResNetGAP_5fldResults_6classRASS.pkl", "ResNet-BiLSTM": "ResNetBiLSTM_NoShuffle_5fldResults_6classRASS.pkl", "ResNet-BiLSTM-shuffle": "ReSNetBiLSTM_WithShuffle_5fldResults_6classRASS.pkl"}),
    "GCS": ("{}_5fld_{}_Results_3classGCS.pkl", {"ResNet-GAP": "ResNetGAP_5fldResults_3classGCS.pkl", "ResNet-BiLSTM": "ResNetBiLSTM_NoShuffle_5fldResults_3classGCS.pkl", "ResNet-BiLSTM-shuffle": "ReSNetBiLSTM_WithShuffle_5fldResults_3classGCS.pkl"}),
    "CAMS": ("{}_5fld_{}_Results_3class.pkl", {"ResNet-GAP": "RESNET_5foldOnlyGap_Results_3class.pkl", "ResNet-BiLSTM": "RESNET_5foldNOSHUFFLE_Results_3class.pkl", "ResNet-BiLSTM-shuffle": "RESNET_5foldTIMESHUFFLE_Results_3class.pkl"}),
    "ICANS": ("{}_5fld_{}_Results.pkl", {"ResNet-GAP": "RESNET_5foldOnlyGap_Results.pkl", "ResNet-BiLSTM": "RESNET_5foldNOSHUFFLE_Results.pkl", "ResNet-BiLSTM-shuffle": "RESNET_5foldTIMESHUFFLE_Results.pkl"}),
}
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def load(scale, name):
    for d in [BASE / scale / "Results5Fld", Path(__file__).resolve().parent.parent / "cohort_models" / scale / "Results5Fld"]:
        if (d / name).exists():
            return pickle.load(open(d / name, "rb"))
    raise FileNotFoundError(name)


rows = []
for scale, (tmpl, deep) in NAMES.items():
    base = {f"{m}-{f}": load(scale, tmpl.format(m, f)) for m in ["LR", "SVM", "KNN"] for f in ["median", "mean", "covar"]}
    for dn, df in deep.items():
        d = load(scale, df)
        for bn, b in base.items():
            for metric in METRICS:
                x, y = np.array([f[metric] for f in d], float), np.array([f[metric] for f in b], float)
                diff = x - y
                two = stats.wilcoxon(x, y, method="exact").pvalue if np.any(diff) else 1.0
                one = stats.wilcoxon(x, y, alternative="greater", method="exact").pvalue if np.any(diff) else 1.0
                rows.append(dict(Scale=scale, Deep=dn, Baseline=bn, Metric=metric, mean_deep=x.mean(), mean_base=y.mean(), folds_better=int((diff > 0).sum()), p_two_sided=two, p_one_sided=one))
r = pd.DataFrame(rows)
r["holm_one_sided"] = np.nan
for _, idx in r.groupby(["Scale", "Metric"]).groups.items():
    p = r.loc[idx, "p_one_sided"].values
    o = np.argsort(p)
    h = np.maximum.accumulate(np.minimum(1, p[o] * (len(p) - np.arange(len(p)))))
    r.loc[idx[o], "holm_one_sided"] = h
r.round(4).to_csv(OUT / "fivefold_wilcoxon.csv", index=False)
s = r.groupby(["Scale", "Metric"]).agg(comparisons=("p_two_sided", "size"), deep_better_all_folds=("folds_better", lambda v: int((v == 5).sum())), min_p_two=("p_two_sided", "min"), n_two_lt05=("p_two_sided", lambda v: int((v < 0.05).sum())), n_one_lt05=("p_one_sided", lambda v: int((v < 0.05).sum())), n_holm_lt05=("holm_one_sided", lambda v: int((v < 0.05).sum())))
s.to_csv(OUT / "fivefold_wilcoxon_summary.csv")
print(s.to_string())
print("overall comparisons", len(r), "deep better in all 5 folds", int((r.folds_better == 5).sum()), "not", int((r.folds_better < 5).sum()))
print(r[r.folds_better < 5].sort_values("folds_better").head(12)[["Scale", "Deep", "Baseline", "Metric", "mean_deep", "mean_base", "folds_better"]].round(3).to_string(index=False))
