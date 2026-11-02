#!/usr/bin/env python3
"""Supplementary Fig. 10: per-patient MAE of the RASS ResNet-GAP model by diagnosis category.

Pooled five-fold cross-validation test predictions (cohort_models/RASS/Results5Fld/
ResNetGAP_5fldResults_6classRASS.pkl), with diagnosis categories from File1-3.xlsx grouped by the
map_dx keyword rules (as in by_diagnosis.py). The published figure came from RASS_Results_Compare.py,
which read a combined diagnosis CSV that was not kept; patients without a diagnosis are Unknown/Other.
The original script ran no test; this one adds Kruskal-Wallis and Holm-adjusted pairwise Mann-Whitney tests.
"""
import os
import pickle
from itertools import combinations
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


def group(x):
    """Same keyword rules as map_dx in cohort_models/RASS/Cohort/Table1_RASS.py."""
    c = str(x).strip().lower()
    if any(k in c for k in ["intracerebral", "ich", "hemorrhage", "ischemic", "stroke", "subdural", "sdh", "traumatic", "tbi", "brain tumor", "tumor", "seizure", "sz", "dement", "neuroinfection", "neuroinflamm", "neuro"]):
        return "Neurological"
    if any(k in c for k in ["cardiac arrest", "arrest", "endocarditis", "cardiac", "circulatory"]):
        return "Cardiac/Circulatory"
    if any(k in c for k in ["sepsis", "toxic", "metabolic", "tme", "encephalopathy", "renal failure", "renal", "liver failure", "hepatic", "liver"]):
        return "Systemic/Metabolic"
    if any(k in c for k in ["respiratory", "pneumonia"]):
        return "Respiratory"
    return "Unknown/Other"


with open(ROOT / "github_moved/cohort_models/RASS/Results5Fld/ResNetGAP_5fldResults_6classRASS.pkl", "rb") as f:
    folds = pickle.load(f)
df = pd.DataFrame(dict(
    f=[x for fo in folds for x in fo["Test_filenames"]],
    t=np.concatenate([np.asarray(fo["Y_true"]).ravel() for fo in folds]),
    p=np.concatenate([np.asarray(fo["Y_pred"]).ravel() for fo in folds]),
))
df["pid"] = df.f.astype(str).str.split("_").str[0].str.split("-").str[1].str[5:]

frames = []
for name in ["File1", "File2", "File3"]:
    d = pd.read_excel(ROOT / "s3" / f"{name}.xlsx")
    d.columns = d.columns.astype(str).str.strip()
    cat = [c for c in d.columns if "diagnosis" in c.lower() and "category" in c.lower()][0]
    frames.append(d[["BDSPPatientID", cat]].set_axis(["pid", "cat"], axis=1))
dx = pd.concat(frames).dropna()
dx["pid"] = dx.pid.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
dx = dx.assign(Dx=dx.cat.map(group)).drop_duplicates("pid").set_index("pid").Dx

s = df.assign(ae=(df.t - df.p).abs()).groupby("pid").agg(MAE=("ae", "mean"))
s["Dx"] = s.index.map(dx)
matched = int(s.Dx.notna().sum())
s["Dx"] = s.Dx.fillna("Unknown/Other")
cats = sorted(s.Dx.unique())

rows = [dict(Dx=c, patients=int((s.Dx == c).sum()), median=s.MAE[s.Dx == c].median(),
             q1=s.MAE[s.Dx == c].quantile(0.25), q3=s.MAE[s.Dx == c].quantile(0.75)) for c in cats]
kw = stats.kruskal(*[s.MAE[s.Dx == c] for c in cats]).pvalue
pairs = list(combinations(cats, 2))
raw = np.array([stats.mannwhitneyu(s.MAE[s.Dx == a], s.MAE[s.Dx == b]).pvalue for a, b in pairs])
o = np.argsort(raw)
adj = np.empty_like(raw)
adj[o] = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw[o]).clip(max=1)
pd.DataFrame(rows).round(4).to_csv(BUILD / "tables" / "supp_fig10_mae_by_dx.csv", index=False)
pw = pd.DataFrame([dict(a=a, b=b, p=p, p_holm=q) for (a, b), p, q in zip(pairs, raw, adj)])
pw.to_csv(BUILD / "tables" / "supp_fig10_pairwise.csv", index=False)
print(f"{len(df):,} segments, {len(s):,} patients, {matched:,} with a diagnosis; Kruskal-Wallis p = {kw:.2g}")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(pw.round(4).to_string(index=False))

# Same styling as the published panel
colors = plt.cm.Pastel1(np.linspace(0, 1, len(cats)))
fig, ax = plt.subplots(figsize=(6, 5), dpi=210)
for i, (c, col) in enumerate(zip(cats, colors)):
    bp = ax.boxplot(s.MAE[s.Dx == c].values, positions=[i], widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.2), boxprops=dict(linewidth=0.8, color="#333333"),
                    whiskerprops=dict(linewidth=0.8, color="#333333"), capprops=dict(linewidth=0.8, color="#333333"),
                    flierprops=dict(marker="o", markerfacecolor="gray", markersize=2, alpha=0.4, markeredgecolor="none"))
    bp["boxes"][0].set_facecolor(col)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([f"{c}\n(n={int((s.Dx == c).sum())})" for c in cats], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("MAE (subject-wise)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(BUILD / "figures" / "SuppFig10_mae_by_diagnosis.png")
