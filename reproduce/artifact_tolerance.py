#!/usr/bin/env python3
"""NESI performance by artifact burden in a random sample of test-set segments (R3.17).

Applies compute_nonusable_percentage from EEG_SQA_based_segment_discard.py to the
released 10-minute EEG segments. Segments are read and deleted; nothing is kept
except per-segment percentages, in intermediates/artifact_per_segment.csv. Once that
file exists, this script only recomputes the summary; --refresh rebuilds it from the EEG.
"""
import ast
import concurrent.futures as cf
import os
import sys
import tempfile
from pathlib import Path

import boto3
from botocore.config import Config
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
INTER = ROOT / "intermediates"
INTER.mkdir(exist_ok=True)
N = 1000  # per cohort
BUCKET = "bdsp-opendata-credentialed"
s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "opendata")).client(
    "s3", config=Config(connect_timeout=20, read_timeout=60, retries={"max_attempts": 5})
)

src = (REPO / "cohort_models/RASS/EEGPreprocessingDownloadSQA/EEG_SQA_based_segment_discard.py").read_text()
ns = {"np": np}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name == "compute_nonusable_percentage":
        exec(compile(ast.Module([node], []), "sqa", "exec"), ns)
pct = ns["compute_nonusable_percentage"]

f = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
rng = np.random.default_rng(317)
parts = []
for ds in ["RASS", "GCS"]:
    t = f[(f.Dataset == ds) & (f.Split == "Test")]
    parts.append(t.iloc[rng.choice(len(t), min(N, len(t)), replace=False)])
sample = pd.concat(parts, ignore_index=True)


def one(i):
    r = sample.iloc[i]
    key = f"yama/cohort_models/{r.Dataset}/{r.Dataset}_EEG10minSegments/{Path(r.MorgothOutputFilename).stem}.mat"
    with tempfile.NamedTemporaryFile(suffix=".mat", dir=ROOT.parent) as tmp:
        s3.download_file(BUCKET, key, tmp.name)
        d = sio.loadmat(tmp.name, variable_names=["data", "Fs"])
        return i, pct(np.asarray(d["data"]), int(np.asarray(d["Fs"]).squeeze()))


PER_SEGMENT = INTER / "artifact_per_segment.csv"
if PER_SEGMENT.exists() and "--refresh" not in sys.argv:
    sample = pd.read_csv(PER_SEGMENT, low_memory=False)
else:
    vals = {}
    with cf.ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(one, i) for i in range(len(sample))]
        for k, fu in enumerate(cf.as_completed(futs)):
            try:
                i, v = fu.result()
                vals[i] = v
            except Exception as e:
                print("miss", type(e).__name__, str(e)[:80], flush=True)
            if k % 100 == 0:
                print(k, flush=True)
    sample["pct_unusable"] = sample.index.map(vals)
    sample = sample.dropna(subset=["pct_unusable"])
    sample[["Dataset", "BDSPPatientID", "MorgothOutputFilename", "RawScore", "NESI", "pct_unusable"]].to_csv(PER_SEGMENT, index=False)
sample[["Dataset", "pct_unusable"]].value_counts().sort_index().to_csv(OUT / "artifact_counts.csv")
rows = []
for ds, d in sample.groupby("Dataset"):
    for lab, v in [("0%", d[d.pct_unusable == 0]), (">0% to <40%", d[d.pct_unusable > 0]), ("All", d)]:
        if len(v) > 10 and v.RawScore.nunique() > 1:
            ids = {k: x.index.values for k, x in v.groupby("BDSPPatientID")}
            K = list(ids)
            b = []
            for _ in range(1000):
                x = v.loc[np.concatenate([ids[K[j]] for j in rng.integers(0, len(K), len(K))])]
                if x.RawScore.nunique() > 1:
                    b.append(abs(stats.spearmanr(x.RawScore, x.NESI)[0]))
            lo, hi = np.percentile(b, [2.5, 97.5])
            rows.append(dict(Scale=ds, Artifact=lab, Segments=len(v), Patients=len(K), rho=round(abs(stats.spearmanr(v.RawScore, v.NESI)[0]), 3), lo=round(lo, 3), hi=round(hi, 3)))
r = pd.DataFrame(rows)
r.to_csv(OUT / "artifact_tolerance.csv", index=False)
print(sample.groupby("Dataset").pct_unusable.value_counts().sort_index().to_string())
print(r.to_string(index=False))
