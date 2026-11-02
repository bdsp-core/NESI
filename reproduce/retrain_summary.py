#!/usr/bin/env python3
"""Compare each sensitivity retrain with the published model on the manuscript's test set.

Reads the test_scores.csv files written by retrain_split_fix.py. CIs resample patients.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
BOOT = 1000


def rho(a, b):
    return abs(stats.spearmanr(a, b)[0])


rows = []
for f in sorted((ROOT.parent / "retrain").glob("*/test_scores.csv")):
    if "smoke" in f.parent.name or f.parent.name.startswith("mapping_"):  # mappings: mapping_summary.py
        continue
    d = pd.read_csv(f, low_memory=False)
    for ds, g in d.groupby("Dataset"):
        rng = np.random.default_rng(2026)
        ids = {k: np.asarray(v) for k, v in g.groupby("BDSPPatientID").indices.items()}
        keys = list(ids)
        t, n, p = g.TrueScore.values, g.NewNESI.values, g.PredictedNESI.values
        new, delta = [], []
        for _ in range(BOOT):
            i = np.concatenate([ids[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
            if len(np.unique(t[i])) > 1:
                a, b = rho(t[i], n[i]), rho(t[i], p[i])
                new.append(a), delta.append(a - b)
        rows.append(dict(
            Run=f.parent.name, Scale=ds, Segments=len(g), Patients=len(keys),
            rho_published=rho(t, p), rho_new=rho(t, n), lo=np.percentile(new, 2.5), hi=np.percentile(new, 97.5),
            delta=rho(t, n) - rho(t, p), delta_lo=np.percentile(delta, 2.5), delta_hi=np.percentile(delta, 97.5),
            r_new_vs_published=np.corrcoef(n, p)[0, 1],
        ))
r = pd.DataFrame(rows).round(3)
r.to_csv(OUT / "retrain_summary.csv", index=False)
print(r.to_string(index=False))
