#!/usr/bin/env python3
"""Spearman of conventional qEEG features with each scale in a test-set sample (Supp Fig 32 check).

The manuscript's baselines put each feature through the NESI head. For one input feature
that head is close to a monotonic transform, so the feature's own Spearman with the scale
approximates the head's. Per-segment features are saved and reused; segments are read and
deleted.

    python reproduce/qeeg_baseline_check.py [N per large cohort]
"""
import concurrent.futures as cf
import os
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import signal, stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
INTER = ROOT / "intermediates"
INTER.mkdir(exist_ok=True)
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
BUCKET = "bdsp-opendata-credentialed"
FOLDER = {"RASS": "RASS_EEG10minSegments", "GCS": "GCS_EEG10minSegments", "CAMS": "CAMS_10minEEGSegments", "ICANS": "ICANS_10minEEGSegments"}
CACHE = INTER / "qeeg_per_segment.csv"
s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "opendata")).client("s3")


def features(x, fs):
    if x.shape[0] > x.shape[1]:
        x = x.T
    x = np.nan_to_num(np.asarray(x[:19], dtype=np.float64))
    f, p = signal.welch(x, fs=fs, nperseg=int(4 * fs), axis=-1)
    keep = (f >= 0.5) & (f <= 30)
    f, p = f[keep], p[:, keep]
    df = f[1] - f[0]
    total = p.sum(1) * df + 1e-12
    delta = p[:, f < 4].sum(1) * df
    alpha = p[:, (f >= 8) & (f < 13)].sum(1) * df + 1e-12
    sef = f[np.minimum((np.cumsum(p, 1) * df / total[:, None] < 0.95).sum(1), len(f) - 1)]
    return np.median(delta / total), np.median(delta / alpha), np.median(sef)


def main():
    f = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
    rng = np.random.default_rng(32)
    parts = []
    for ds in FOLDER:
        t = f[(f.Dataset == ds) & (f.Split == "Test")]
        parts.append(t.iloc[rng.choice(len(t), min(N, len(t)), replace=False)])
    sample = pd.concat(parts, ignore_index=True)
    done = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["Dataset", "MorgothOutputFilename"])
    seen = set(zip(done.Dataset, done.MorgothOutputFilename))
    todo = sample[[(a, b) not in seen for a, b in zip(sample.Dataset, sample.MorgothOutputFilename)]]
    print(f"{len(sample)} segments, {len(todo)} to compute", flush=True)

    def load_mat(path):
        """CAM-S and ICANS segments are MATLAB v7.3 (HDF5), which scipy can't read."""
        try:
            return sio.loadmat(path, variable_names=["data", "Fs"])
        except NotImplementedError:
            import h5py

            with h5py.File(path, "r") as f:
                x = np.asarray(f["data"])
                fs = np.asarray(f["Fs"]).squeeze()
            return {"data": x.T if x.shape[0] > x.shape[1] else x, "Fs": fs}

    def one(r):
        key = f"yama/cohort_models/{r.Dataset}/{FOLDER[r.Dataset]}/{Path(r.MorgothOutputFilename).stem}.mat"
        with tempfile.NamedTemporaryFile(suffix=".mat", dir=ROOT.parent) as tmp:
            s3.download_file(BUCKET, key, tmp.name)
            d = load_mat(tmp.name)
            return dict(Dataset=r.Dataset, MorgothOutputFilename=r.MorgothOutputFilename, BDSPPatientID=r.BDSPPatientID, RawScore=r.RawScore, NESI=r.NESI,
                        **dict(zip(["rel_delta", "delta_alpha", "sef95"], features(d["data"], int(np.asarray(d["Fs"]).squeeze())))))

    new, miss = [], 0
    with cf.ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(one, r) for r in todo.itertuples()]
        for k, fu in enumerate(cf.as_completed(futs)):
            try:
                new.append(fu.result())
            except Exception as e:
                miss += 1
                if miss < 6:
                    print("miss", type(e).__name__, str(e)[:100], flush=True)
            if k % 50 == 49:
                pd.concat([done, pd.DataFrame(new)]).to_csv(CACHE, index=False)
                print(k + 1, flush=True)
    d = pd.concat([done, pd.DataFrame(new)])
    d.to_csv(CACHE, index=False)
    summarize(d, miss)


def summarize(d, miss=0):
    rows = []
    for ds, g in d.groupby("Dataset"):
        y = {"RASS": -g.RawScore, "GCS": 15 - g.RawScore}.get(ds, g.RawScore)
        row = dict(Scale=ds, Segments=len(g), NESI=stats.spearmanr(y, g.NESI)[0])
        for c in ["rel_delta", "delta_alpha", "sef95"]:
            s = stats.spearmanr(y, g[c])[0]
            row[c] = -s if c == "sef95" else s
        rows.append(row)
    r = pd.DataFrame(rows).round(3)
    r.to_csv(OUT / "qeeg_baseline_check.csv", index=False)
    print(r.to_string(index=False), "\nmissing", miss)


if __name__ == "__main__":
    main()
