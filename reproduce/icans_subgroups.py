#!/usr/bin/env python3
"""ICANS Spearman on the test set with and without patients who never had ICANS (Supplement)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
f = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
t = f[(f.Dataset == "ICANS") & (f.Split == "Test")].copy()
peak_all = f[f.Dataset == "ICANS"].groupby("BDSPPatientID").RawScore.max()
t["peak"] = t.BDSPPatientID.map(peak_all)


def row(lab, d, boot=1000):
    rng = np.random.default_rng(2026)
    ids = {k: np.asarray(v) for k, v in d.groupby("BDSPPatientID").indices.items()}
    keys, b = list(ids), []
    x, y = d.RawScore.values, d.NESI.values
    for _ in range(boot):
        i = np.concatenate([ids[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        if len(np.unique(x[i])) > 1:
            b.append(stats.spearmanr(x[i], y[i])[0])
    return dict(Group=lab, Segments=len(d), Patients=len(keys), rho=stats.spearmanr(x, y)[0], boot_mean=np.mean(b), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5))


r = pd.DataFrame([
    row("All test", t),
    row("Patients with any grade above 0", t[t.peak > 0]),
    row("Patients with grade above 0 in the test set", t[t.groupby("BDSPPatientID").RawScore.transform("max") > 0]),
    row("Segments with grade above 0", t[t.RawScore > 0]),
]).round(3)
r.to_csv(OUT / "icans_subgroups.csv", index=False)
print(r.to_string(index=False))
