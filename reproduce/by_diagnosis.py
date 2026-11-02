#!/usr/bin/env python3
"""Held-out NESI performance by major diagnostic category (R3.27)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


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


frames = []
for f in ["File1", "File2", "File3"]:
    d = pd.read_excel(ROOT / "s3" / f"{f}.xlsx")
    d.columns = d.columns.astype(str).str.strip()
    cat = [c for c in d.columns if "diagnosis" in c.lower() and "category" in c.lower()][0]
    frames.append(d[["BDSPPatientID", cat]].rename(columns={cat: "cat"}))
dx = pd.concat(frames).dropna(subset=["BDSPPatientID"])
dx["BDSPPatientID"] = dx.BDSPPatientID.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
dx["Dx"] = dx.cat.map(group)
dx = dx.drop_duplicates("BDSPPatientID").set_index("BDSPPatientID").Dx

s = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
t = s[s.Split == "Test"].copy()
t["Dx"] = t.BDSPPatientID.astype(str).map(dx).fillna("Unknown/Other")
rng = np.random.default_rng(27)
rows = []
for (ds, g), v in t.groupby(["Dataset", "Dx"]):
    v = v.reset_index(drop=True)
    p = v.BDSPPatientID.nunique()
    rho = abs(stats.spearmanr(v.RawScore, v.NESI)[0]) if v.RawScore.nunique() > 1 else np.nan
    lo = hi = np.nan
    if p >= 5 and v.RawScore.nunique() > 1:
        ids = {k: x.index.values for k, x in v.groupby("BDSPPatientID")}
        K = list(ids)
        b = []
        for _ in range(1000):
            ix = np.concatenate([ids[K[j]] for j in rng.integers(0, len(K), len(K))])
            x = v.loc[ix]
            if x.RawScore.nunique() > 1:
                b.append(abs(stats.spearmanr(x.RawScore, x.NESI)[0]))
        lo, hi = np.percentile(b, [2.5, 97.5])
    rows.append(dict(Scale=ds, Dx=g, Patients=p, Segments=len(v), rho=round(rho, 3), lo=round(lo, 3), hi=round(hi, 3)))
r = pd.DataFrame(rows)
r.to_csv(OUT / "performance_by_diagnosis.csv", index=False)
print(r.to_string(index=False))
