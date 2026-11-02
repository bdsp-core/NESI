#!/usr/bin/env python3
"""External validation at Northwestern (Fig 7, Supp Fig 38 and their numbers).

The first run builds intermediates/external/recordings.csv from Erika's files in
NESI_EXTERNAL_ROOT: one row per recording, with its mean and median NESI, clinical
scores and hours to each assessment, and no calendar dates. Later runs read
only that file. NESI is computed in 10-minute windows advanced by 1 s, then averaged
over each recording.
"""
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output"
INTER = ROOT / "intermediates" / "external"
INTER.mkdir(parents=True, exist_ok=True)
TABLE = INTER / "recordings.csv"
BOOT, SEED = 1000, 2026
WINDOWS = [("30 min", 0.5), ("60 min", 1.0), ("120 min", 2.0), ("All", np.inf)]


def build():
    ext = Path(os.environ["NESI_EXTERNAL_ROOT"])
    x = pd.read_excel(ext / "nesi-erika.xlsx")
    rows = []
    for f in (ext / "NESIPredictions/NESIPredictions").glob("*.csv"):
        m = re.match(r"(.+?)_ErikaSegment_(\d+)_NESI", f.name)
        v = pd.read_csv(f).NESI.values
        rows.append(dict(Filename=m[1], windows=len(v), nesi_median=np.median(v), nesi_mean=v.mean()))
    d = x.merge(pd.DataFrame(rows), on="Filename")
    t0 = d.EEG_segment_StartdateTime
    t1 = t0 + pd.to_timedelta(d["Segment duration_s"], unit="s")
    for name, col in [("rass", "RASS_closest_Datetime"), ("gcs", "GCS_closest_DateTime"), ("cam", "AnyCAM_closest_Datetime")]:
        a = d[col]
        d[f"{name}_side"] = np.where(a < t0, "before", np.where(a > t1, "after", "inside"))
        d[f"{name}_gap_h"] = np.where(a < t0, (t0 - a).dt.total_seconds(), np.where(a > t1, (a - t1).dt.total_seconds(), 0)) / 3600
    d = d.rename(columns={"Segment duration_s": "duration_s", "RASS_numeric_closest": "rass", "GCS_closest": "gcs", "AnyCAM_closestScore": "cam", "Segmentduration_sec_AfterAutoreject": "duration_after_autoreject_s"})
    keep = ["Filename", "duration_s", "duration_after_autoreject_s", "windows", "nesi_median", "nesi_mean", "rass", "gcs", "cam"]
    keep += [f"{n}_{k}" for n in ["rass", "gcs", "cam"] for k in ["side", "gap_h"]]
    d[keep].to_csv(TABLE, index=False)


def rho_ci(a, b, rng):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    draws = [abs(stats.spearmanr(a[i], b[i])[0]) for i in (rng.integers(0, n, n) for _ in range(BOOT))]
    return abs(stats.spearmanr(a, b)[0]), *np.percentile(draws, [2.5, 97.5])


def boxes(ax, d, col, title):
    vals = sorted(d[col].unique())
    groups = [d[d[col] == v].nesi_mean for v in vals]
    b = ax.boxplot(groups, patch_artist=True, widths=0.6, flierprops=dict(marker=".", markersize=3, alpha=0.4), medianprops=dict(color="black", linewidth=1.5))
    for p in b["boxes"]:
        p.set_facecolor("#2b83ba" if col == "rass" else "#e6ab02")
    ax.set_xticklabels([f"{int(v)}\nn={len(g)}" for v, g in zip(vals, groups)], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def main():
    if not TABLE.exists():
        build()
    d = pd.read_csv(TABLE)
    rng = np.random.default_rng(SEED)
    num, rows = {}, []
    num["EXT.recordings"] = len(d)
    num["EXT.duration_median_s"] = d.duration_s.median()
    num["EXT.autoreject_recordings"] = d.duration_after_autoreject_s.notna().sum()
    num["EXT.windows_equal_duration_minus_599"] = (d.windows == d.duration_s - 599).mean()
    for scale in ["rass", "gcs"]:
        for k, v in d[f"{scale}_side"].value_counts().items():
            num[f"EXT.{scale}.{k}"] = v
        for lab, h in WINDOWS:
            s = d[d[f"{scale}_gap_h"] <= h]
            r, lo, hi = rho_ci(s[scale], s.nesi_mean, rng)
            rm = abs(stats.spearmanr(s[scale], s.nesi_median)[0])
            key = lab.split()[0]
            num[f"EXT.{scale}.{key}.n"], num[f"EXT.{scale}.{key}.rho"], num[f"EXT.{scale}.{key}.lo"], num[f"EXT.{scale}.{key}.hi"] = len(s), r, lo, hi
            rows.append(dict(Scale=scale.upper(), Window=lab, n=len(s), rho=r, lo=lo, hi=hi, rho_if_median_summary=rm))
    # CAM: positive vs negative, among those assessable; UTA means RASS -4 or -5
    c = d[d.cam.isin(["Positive", "Negative"])]
    y, x = (c.cam == "Positive").astype(int).values, c.nesi_mean.values
    auc = roc_auc_score(y, x)
    draws = []
    for _ in range(BOOT):
        i = rng.integers(0, len(c), len(c))
        draws.append(roc_auc_score(y[i], x[i]))
    num["EXT.cam.auc"], (num["EXT.cam.auc_lo"], num["EXT.cam.auc_hi"]) = auc, np.percentile(draws, [2.5, 97.5])
    for lab, g in d.groupby("cam"):
        k = lab.split(",")[0]
        num[f"EXT.cam.{k}.n"], num[f"EXT.cam.{k}.median"] = len(g), g.nesi_mean.median()
    near = c[c.cam_gap_h <= 1]
    yn, xn = (near.cam == "Positive").astype(int).values, near.nesi_mean.values
    num["EXT.cam.auc_60min"], num["EXT.cam.n_60min"] = roc_auc_score(yn, xn), len(near)
    (BUILD / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(BUILD / "tables" / "external_validation.csv", index=False)
    json.dump({k: float(v) for k, v in num.items()}, open(INTER / "numbers.json", "w"), indent=1)

    out = BUILD / "figures" / "final"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for a, sc, name in zip(ax, ["rass", "gcs"], ["RASS", "GCS"]):
        r = [x for x in rows if x["Scale"] == name and x["Window"] == "All"][0]
        boxes(a, d, sc, f"{name}\nρ = {r['rho']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]")
        a.set_xlabel(f"{name} score")
    ax[0].set_ylabel("NESI")
    fig.tight_layout()
    fig.savefig(out / "Fig7.png", dpi=300)
    fig, ax = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    for i, (sc, name) in enumerate([("rass", "RASS"), ("gcs", "GCS")]):
        for j, (lab, h) in enumerate(WINDOWS):
            s = d[d[f"{sc}_gap_h"] <= h]
            r = [x for x in rows if x["Scale"] == name and x["Window"] == lab][0]
            t = "All assessments" if lab == "All" else f"Within {lab}"
            boxes(ax[i, j], s, sc, f"{'ABCDEFGH'[i * 4 + j]}. {name}, {t}\nρ = {r['rho']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}], n = {r['n']}")
        ax[i, 0].set_ylabel("NESI")
    fig.tight_layout()
    fig.savefig(out / "SuppFig38.png", dpi=250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(json.dumps({k: round(float(v), 3) for k, v in num.items() if "cam" in k or k.startswith(("EXT.rec", "EXT.dur", "EXT.auto", "EXT.win")) or k.endswith(("before", "after", "inside"))}, indent=0))


if __name__ == "__main__":
    main()
