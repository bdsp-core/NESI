#!/usr/bin/env python3
"""Supplementary Figs 14-17: cross-score blind evaluation of the scale-specific score models.

Each scale's ResNet-GAP CORN model (the released "Best" weights in
cohort_models/<scale>/<scale>_Best_DL_model/ResNetGAP/; these are fold 4 for CAM-S,
fold 3 for ICANS and fold 4 for RASS of the 5-fold runs, and the GCS fold is not
recorded) is applied to the held-out segments of the other three cohorts. Following
cohort_models/{CAMS,GCS,ICANS}/model/Testing/Logitsof*Model_against_*.py:

- held-out segments: a per-cohort 80/20 patient split, train_test_split(test_size=0.2,
  random_state=42) on the cohort's unique patients (not the NESI split);
- outputs: CORN logits -> class probabilities -> class log-odds, probabilities clamped
  to [1e-7, 1 - 1e-7] (so log-odds lie within +-16.1). The published figures read the
  CORN outputs as unconditional (see corn_class_logits); --exact draws the chained
  version as SuppFigNN_exact.png. Trends are the same either way;
- before plotting, a segment is dropped when any class log-odds lies more than 3 IQR
  outside the quartiles of its true-score group (clean_logit_outliers). --no-clean
  draws every segment.

No script for the RASS model (Supp Fig 17) was kept; it is done the same way with six
classes. MORGOTH activations come from the NESI feature pickles in NESI_DATA_ROOT/s3
(train rows from the float32 memmap train_f32.npy.mem when present), falling back to
the credentialed bucket for any segment not found there.

Per-segment outputs are cached in NESI_DATA_ROOT/intermediates/cross_score/; later runs
only draw. --refresh recomputes.
"""
import argparse
import concurrent.futures as cf
import io
import os
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(os.environ["NESI_DATA_ROOT"])
REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "figures" / "final"
CACHE = ROOT / "intermediates" / "cross_score"
META = ROOT / "github_moved" / "NESI" / "model" / "Training"
BUCKET = "bdsp-opendata-credentialed"
HEADS = ["SLEEP", "NM", "BS", "FOCGEN", "SLOWING", "IIIC"]
SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
LABEL = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}
NCLASS = {"RASS": 6, "GCS": 3, "CAMS": 3, "ICANS": 3}
# Figure number, model, columns (other cohorts) in published order
FIGS = {
    14: ("CAMS", ["RASS", "GCS", "ICANS"]),
    15: ("ICANS", ["RASS", "GCS", "CAMS"]),
    16: ("GCS", ["RASS", "CAMS", "ICANS"]),
    17: ("RASS", ["CAMS", "ICANS", "GCS"]),
}
# Rows of the 3-class figures, top to bottom: (model class index, label). The GCS model's
# class 0 is GCS 3-8 (GCS_DL_ResNets_5Fld.py), so its rows are put in severity order.
ROWS = {
    "CAMS": [(0, "Mild\nCAM-S (CAM-S 0-1)"), (1, "Moderate\nCAM-S (CAM-S 2-5)"), (2, "Severe\nCAM-S (CAM-S 6-7)")],
    "ICANS": [(0, "Mild\nICANS (ICANS 0)"), (1, "Moderate\nICANS (ICANS 1-2)"), (2, "Severe\nICANS (ICANS 3-4)")],
    "GCS": [(2, "Mild\nGCS (GCS 13-15)"), (1, "Moderate\nGCS (GCS 9-12)"), (0, "Severe\nGCS (GCS 3-8)")],
}
PASTEL = ["#A8E6CF", "#FFD3B6", "#FFAAA5"]
RASS_COLORS = ["#F4A9A8", "#C9B3DC", "#FBE3B4", "#F0E08A", "#B4E3D0", "#C5F5C5"]


def pid(f):
    return f.split("_")[0].split("-")[1][5:]


def cohort(ds):
    """Segment table for one cohort as in the Testing scripts: patient, file, raw score."""
    if ds == "CAMS":
        d = pd.read_csv(META / "CAMSTraining_Final_Metadata.csv")
        d = d.assign(RawScore=d.CAMS_SF)
    elif ds == "GCS":
        d = pd.read_csv(META / "GCSTraining_Final_Metadata.csv")
        d["MorgothOutputFilename"] = d.Filename.astype(str) + ".csv"
        d = d.assign(BDSPPatientID=d.MorgothOutputFilename.map(pid), RawScore=d.GCS_value)
    elif ds == "RASS":
        d = pd.read_csv(META / "RASSTraining_Final_Metadata.csv")
        d["MorgothOutputFilename"] = d.Filename.str.rstrip("'") + ".csv"
        d = d[~d.RASS_value.isin([1, 2, 3, 4])]
        d = d.assign(BDSPPatientID=d.MorgothOutputFilename.map(pid), RawScore=d.RASS_value)
    else:
        d = pd.read_csv(META / "ICANSTraining_Final_Metadata.csv")
        d = d.assign(BDSPPatientID=d.StudyID, RawScore=d.ICANS_raw)
    return d.assign(Dataset=ds)[["BDSPPatientID", "Dataset", "MorgothOutputFilename", "RawScore"]].reset_index(drop=True)


def held_out(d):
    """get_test_metedata_cohort: the 20% test patients of a 70/10/20 split, seed 42."""
    from sklearn.model_selection import train_test_split

    s = d[["BDSPPatientID", "Dataset"]].drop_duplicates()
    _, te = train_test_split(s, test_size=0.2, stratify=s["Dataset"], random_state=42)
    return d[d.BDSPPatientID.isin(set(te.BDSPPatientID))].reset_index(drop=True)


def s3_matrix(ds, fn, s3):
    parts = []
    for h in HEADS:
        for k in range(5):
            try:
                body = s3.get_object(Bucket=BUCKET, Key=f"yama/cohort_models/{ds}/MorgothActivations/{h}/{fn}")["Body"].read()
                break
            except Exception:
                if k == 4:
                    raise
                time.sleep(2**k)
        parts.append(pd.read_csv(io.BytesIO(body)).drop(columns=["pred_class"], errors="ignore").values)
    return np.concatenate(parts, axis=1).astype("float32")


def features(seg):
    """591 x 17 MORGOTH matrices for the rows of `seg`, in order."""
    X = np.full((len(seg), 591, 17), np.nan, "float32")
    key = pd.Series(np.arange(len(seg)), index=pd.MultiIndex.from_frame(seg[["Dataset", "MorgothOutputFilename"]]))
    for name in ["test", "val", "train"]:
        r = pd.read_csv(ROOT / "s3" / f"NESI_{name}_results.csv", usecols=["Filename", "Dataset"], low_memory=False)
        r["row"] = np.arange(len(r))
        hit = r.merge(seg[["Dataset", "MorgothOutputFilename"]], left_on=["Dataset", "Filename"], right_on=["Dataset", "MorgothOutputFilename"])
        if hit.empty:
            continue
        dst = key.loc[list(zip(hit.Dataset, hit.Filename))].values
        mem = ROOT.parent / "train_f32.npy.mem"
        if name == "train" and mem.exists():
            src = np.memmap(mem, dtype="float32", mode="r", shape=(len(r), 591, 17))
        else:
            print(f"loading NESI {name} features", flush=True)
            src = np.asarray(pickle.load(open(ROOT / "s3" / f"NESITripletIP_Morgoth_{name}_data.pkl", "rb")))
        order = np.argsort(hit.row.values)
        for a in range(0, len(order), 4000):
            o = order[a : a + 4000]
            X[dst[o]] = src[hit.row.values[o]]
        del src
        print(f"  {name}: {len(hit)} segments", flush=True)
    miss = np.where(np.isnan(X[:, 0, 0]))[0]
    if len(miss):
        import boto3

        print(f"{len(miss)} segments not in the NESI pickles; fetching from S3", flush=True)
        s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "opendata")).client("s3")
        part = CACHE / "s3_segments"
        part.mkdir(parents=True, exist_ok=True)

        def one(i):
            f = part / f"{seg.Dataset[i]}__{seg.MorgothOutputFilename[i]}.npy"
            if not f.exists():
                np.save(f, s3_matrix(seg.Dataset[i], seg.MorgothOutputFilename[i], s3))
            return i, np.load(f)

        with cf.ThreadPoolExecutor(8) as ex:
            for i, a in ex.map(one, miss):
                X[i] = a
    return X


def corn_class_logits(z, exact=False):
    """CORN outputs (N, K-1) -> clamped class probabilities and class log-odds (N, K).

    As published (corn_to_class_probs in the Testing scripts), sigmoid(z_k) is taken as
    P(y > k), the CORAL reading. CORN's outputs are conditional, P(y > k | y > k-1), so
    exact=True chains them by a cumulative product. The published reading leaves the
    first class unchanged, makes the last class P(y > K-2 | y > K-3), and lets middle
    classes go negative before the clamp (the floor at -16.1 in the Moderate rows).
    """
    p = 1 / (1 + np.exp(-z.astype("float64")))
    if exact:
        p = np.cumprod(p, axis=1)  # P(y > k)
    k = z.shape[1] + 1
    cp = np.empty((len(z), k))
    cp[:, 0] = 1 - p[:, 0]
    cp[:, 1:-1] = p[:, :-1] - p[:, 1:]
    cp[:, -1] = p[:, -1]
    cp = np.clip(cp, 1e-7, 1 - 1e-7)
    return cp, np.log(cp / (1 - cp))


def compute():
    import ast

    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(os.cpu_count() or 4)
    src = (REPO / "cohort_models/CAMS/model/Testing/LogitsofCAMSModel_against_RASS_GCS_ICANS_dataset.py").read_text()
    ns = {"torch": torch, "nn": nn, "F": F, "np": np}
    for n in ast.parse(src).body:
        if isinstance(n, ast.ClassDef) and n.name in ("ResidualBlock1D", "MORGOTH_ResNet1D_onlyGAP_CORAL"):
            exec(compile(ast.Module([n], []), "m", "exec"), ns)
    seg = pd.concat([held_out(cohort(ds)) for ds in SCALES], ignore_index=True)
    for ds, v in seg.groupby("Dataset", sort=False):
        print(f"{ds}: {len(v)} held-out segments, {v.BDSPPatientID.nunique()} patients", flush=True)
    X = features(seg)
    rows = []
    for model_ds, cols in [(m, c) for m, c in FIGS.values()]:
        k = NCLASS[model_ds]
        net = ns["MORGOTH_ResNet1D_onlyGAP_CORAL"](num_features=17, num_classes=k)
        w = REPO / f"cohort_models/{model_ds}/{model_ds}_Best_DL_model/ResNetGAP/RESNETGAP_Best_{model_ds}.pth"
        net.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        net.eval()
        for ds in cols:
            idx = np.where(seg.Dataset == ds)[0]
            with torch.no_grad():
                z = np.concatenate([net(torch.from_numpy(X[idx[a : a + 512]])).numpy() for a in range(0, len(idx), 512)])
            t = seg.iloc[idx].reset_index(drop=True).assign(Model=model_ds)
            for j in range(k - 1):
                t[f"corn{j}"] = z[:, j]
            rows.append(t)
            print(f"{model_ds} model on {ds}: {len(idx)} segments", flush=True)
    out = pd.concat(rows, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE / "cross_score_corn_outputs.csv.gz", index=False)
    return out


def with_logits(df, exact=False):
    """Add class log-odds columns logit0.. from the cached CORN outputs corn0.."""
    parts = []
    for m, t in df.groupby("Model", sort=False):
        k = NCLASS[m]
        _, lo = corn_class_logits(t[[f"corn{j}" for j in range(k - 1)]].values, exact=exact)
        t = t.copy()
        for j in range(k):
            t[f"logit{j}"] = lo[:, j]
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def clean(t, k, mult=3.0):
    """clean_logit_outliers: drop a segment if any class log-odds is > mult IQR outside its group's quartiles."""
    L = t[[f"logit{j}" for j in range(k)]].values
    keep = np.ones(len(t), bool)
    for c in np.unique(t.RawScore):
        m = (t.RawScore == c).values
        for j in range(k):
            q1, q3 = np.percentile(L[m, j], [25, 75])
            lo, hi = q1 - mult * (q3 - q1), q3 + mult * (q3 - q1)
            keep[m & ((L[:, j] < lo) | (L[:, j] > hi))] = False
    return t[keep]


def boxes(ax, t, j, color, lw=0.8, rotate=False):
    lv = np.sort(t.RawScore.unique())
    bp = ax.boxplot(
        [t.loc[t.RawScore == v, f"logit{j}"].values for v in lv],
        tick_labels=[str(int(v)) for v in lv], patch_artist=True, widths=0.7,
        flierprops=dict(marker="o", markersize=2, markeredgecolor="gray", alpha=0.3, linestyle="none"),
        medianprops=dict(color="black", linewidth=1.2),
    )
    for b in bp["boxes"]:
        b.set(facecolor=color, edgecolor="black", linewidth=lw)
    for x in bp["whiskers"] + bp["caps"]:
        x.set(color="#333333", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)
    if rotate:
        ax.tick_params(axis="x", labelrotation=45)


def letter(ax, s, x=0.88, y=0.06):
    ax.text(x, y, f"({s})", transform=ax.transAxes, ha="right", va="bottom", fontsize=11)


def draw(df, do_clean=True, suffix=""):
    OUT.mkdir(parents=True, exist_ok=True)
    stats = []
    for num, (m, cols) in FIGS.items():
        k = NCLASS[m]
        data = {}
        for ds in cols:
            t = df[(df.Model == m) & (df.Dataset == ds)]
            c = clean(t, k) if do_clean else t
            data[ds] = c
            stats.append(dict(Figure=f"SuppFig{num}", Model=LABEL[m], Cohort=LABEL[ds], Segments=len(t), Patients=t.BDSPPatientID.nunique(), Dropped=len(t) - len(c)))
        if m == "RASS":
            fig, axes = plt.subplots(6, 3, figsize=(7.2, 8.2), dpi=300)
            for ci, ds in enumerate(cols):
                for r in range(6):
                    ax = axes[r, ci]
                    boxes(ax, data[ds], r, RASS_COLORS[r])
                    letter(ax, "abcdefghijklmnopqr"[ci * 6 + r], x=0.9, y=0.08)
                    if ci == 0:
                        ax.set_ylabel(f"Logits of RASS {r - 5}", fontsize=8)
                    if r == 5:
                        ax.set_xlabel(f"True {LABEL[ds]} scale", fontsize=8)
        else:
            fig, axes = plt.subplots(3, 3, figsize=(7, 5), dpi=300)
            for ci, ds in enumerate(cols):
                for r, (j, lab) in enumerate(ROWS[m]):
                    ax = axes[r, ci]
                    boxes(ax, data[ds], j, PASTEL[r], rotate=num in (14, 15))
                    letter(ax, "abcdefghi"[ci * 3 + r])
                    if ci == 0:
                        ax.set_ylabel("Logits for " + lab, fontsize=9)
                    if r == 2:
                        ax.set_xlabel(f"True {LABEL[ds]} scale", fontsize=10, labelpad=6)
        fig.tight_layout()
        fig.savefig(OUT / f"SuppFig{num}{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("wrote", OUT / f"SuppFig{num}{suffix}.png")
    s = pd.DataFrame(stats)
    print(s.to_string(index=False))
    return s


def medians(df, do_clean=True):
    """Median class log-odds by true score: the numbers behind each box."""
    out = []
    for m, cols in FIGS.values():
        k = NCLASS[m]
        for ds in cols:
            t = df[(df.Model == m) & (df.Dataset == ds)]
            t = clean(t, k) if do_clean else t
            g = t.groupby("RawScore")[[f"logit{j}" for j in range(k)]].median().round(2)
            g.insert(0, "n", t.groupby("RawScore").size())
            out.append(g.reset_index().assign(Model=LABEL[m], Cohort=LABEL[ds]))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute the per-segment outputs")
    ap.add_argument("--no-clean", action="store_true", help="plot every segment (no 3-IQR outlier removal)")
    ap.add_argument("--exact", action="store_true", help="chain CORN's conditional outputs (corn_class_logits); "
                    "figures are written as SuppFigNN_exact.png")
    a = ap.parse_args()
    f = CACHE / "cross_score_corn_outputs.csv.gz"
    df = pd.read_csv(f) if f.exists() and not a.refresh else compute()
    df = with_logits(df, exact=a.exact)
    tag = ("_exact" if a.exact else "") + ("_noclean" if a.no_clean else "")
    draw(df, do_clean=not a.no_clean, suffix=tag)
    medians(df, do_clean=not a.no_clean).to_csv(CACHE / f"cross_score_medians{tag}.csv", index=False)
