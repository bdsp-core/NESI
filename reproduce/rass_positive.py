#!/usr/bin/env python3
"""Exploratory: NESI for held-out patients' segments with RASS +1 to +4 (R1.3, R2.3, R3.5).

These segments were excluded from training and evaluation. Uses the published
weights on MORGOTH activations from the credentialed bucket.
"""
import ast
import concurrent.futures as cf
import io
import os
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.nn import functional as F

ROOT = Path(os.environ["NESI_DATA_ROOT"])
REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT.parent / "rass_positive_features.npz"
BUCKET, PREFIX = "bdsp-opendata-credentialed", "yama/cohort_models/RASS/MorgothActivations"
HEADS = ["SLEEP", "NM", "BS", "FOCGEN", "SLOWING", "IIIC"]

scores = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
r = scores[(scores.Dataset == "RASS") & (scores.Split == "Test")]
test_ids = set(r.BDSPPatientID.astype(str))
meta = pd.read_csv(ROOT / "github_moved/cohort_models/RASS/model/Training/RASSTraining_Final_Metadata.csv")
meta["fn"] = meta.Filename.str.rstrip("'") + ".csv"
meta["pid"] = meta.fn.map(lambda f: f.split("_")[0].split("-")[1][5:])
pos = meta[(meta.RASS_value > 0) & meta.pid.isin(test_ids)].reset_index(drop=True)
print("held-out RASS>0 segments", len(pos), "patients", pos.pid.nunique(), pos.RASS_value.value_counts().sort_index().to_dict(), flush=True)

s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "opendata")).client("s3")


def load(fn):
    parts = []
    for h in HEADS:
        body = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{h}/{fn}")["Body"].read()
        d = pd.read_csv(io.BytesIO(body))
        parts.append(d.drop(columns=["pred_class"], errors="ignore").values)
    return np.concatenate(parts, axis=1).astype("float32")


if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    X, ok = z["X"], z["ok"]
else:
    X, ok = [], []
    with cf.ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(load, f): i for i, f in enumerate(pos.fn)}
        res = {}
        for k, fu in enumerate(cf.as_completed(futs)):
            i = futs[fu]
            try:
                a = fu.result()
                if a.shape == (591, 17):
                    res[i] = a
            except Exception as e:
                print("miss", type(e).__name__, flush=True)
            if k % 200 == 0:
                print(k, flush=True)
    ok = np.array(sorted(res))
    X = np.stack([res[i] for i in ok])
    np.savez(CACHE, X=X, ok=ok)
print("features", X.shape, flush=True)
pos = pos.iloc[ok].reset_index(drop=True)

src = (REPO / "NESI/model/Testing/NESI_Test_Universal_Fulldataset.py").read_text()
ns = {"torch": torch, "nn": nn, "F": F, "np": np}
for n in ast.parse(src).body:
    if isinstance(n, ast.ClassDef) and n.name in ("ResidualBlock1D", "MORGOTH_ResNet1D_onlyGAP", "EEGScoringModel"):
        exec(compile(ast.Module([n], []), "m", "exec"), ns)
ck = REPO / "NESI/model/ModelCheckpoints"
enc = ns["MORGOTH_ResNet1D_onlyGAP"](num_features=17)
enc.load_state_dict(torch.load(ck / "ResNetGAP_BestModel.pth", map_location="cpu", weights_only=False))
head = ns["EEGScoringModel"]()
head.load_state_dict(torch.load(ck / "NESI_best_model.pth", map_location="cpu", weights_only=False)["model_state_dict"])
m = nn.Sequential(enc, head).eval()
with torch.no_grad():
    pos["NESI"] = np.concatenate([np.ravel(m(torch.from_numpy(X[i : i + 128])).numpy()) for i in range(0, len(X), 128)])

allr = pd.concat([
    r.assign(pid=r.BDSPPatientID.astype(str), RASS=r.RawScore)[["pid", "RASS", "NESI"]],
    pos.assign(RASS=pos.RASS_value)[["pid", "RASS", "NESI"]],
], ignore_index=True)
rows = [dict(RASS=int(k), Segments=len(v), Patients=v.pid.nunique(), Median=v.NESI.median(), Q1=v.NESI.quantile(0.25), Q3=v.NESI.quantile(0.75)) for k, v in allr.groupby("RASS")]
t = pd.DataFrame(rows).round(3)
t.to_csv(OUT / "rass_positive_by_level.csv", index=False)
print(t.to_string(index=False))
g = np.random.default_rng(13)


def boot(d):
    d = d.reset_index(drop=True)
    ids = {k: v.index.values for k, v in d.groupby("pid")}
    K = list(ids)
    out = []
    for _ in range(1000):
        x = d.loc[np.concatenate([ids[K[j]] for j in g.integers(0, len(K), len(K))])]
        if x.RASS.nunique() > 1:
            out.append(stats.spearmanr(x.RASS, x.NESI)[0])
    return stats.spearmanr(d.RASS, d.NESI)[0], *np.percentile(out, [2.5, 97.5])


for nm, d in [("RASS -5 to 0", allr[allr.RASS <= 0]), ("RASS 0 to +4", allr[allr.RASS >= 0]), ("RASS -5 to +4", allr)]:
    print(nm, len(d), d.pid.nunique(), "rho (signed) %.3f (%.3f, %.3f)" % boot(d), flush=True)
