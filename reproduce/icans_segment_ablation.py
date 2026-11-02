#!/usr/bin/env python3
"""ICANS segment-selection sensitivity (Supplementary Fig. 32 and its text).

Compares ICANS test-set correlations for three ways of choosing EEG segments: the proposed
maximum-wakefulness segment, a random awake segment, and all segments of each patient-day
(each segment carrying that day's ICANS grade). The random-awake and all-segment scores were
produced by Arka Roy with the published models from MORGOTH activations that exist only on his
machine; they are saved under intermediates/icans_segment_ablation/. CIs resample patients;
the segment-level CIs of the original analysis are also reported for comparison.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
I = ROOT / "intermediates" / "icans_segment_ablation"
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
BOOT = 1000


def prep(f, col):
    d = pd.read_csv(f, low_memory=False)
    d = d[(d.Dataset == "ICANS") & (d.Split.str.lower() == "test")][["BDSPPatientID", "RawScore", col]].dropna()
    d["BDSPPatientID"] = d.BDSPPatientID.astype(str).str.replace(r"\.0$", "", regex=True)
    return d.rename(columns={col: "NESI"}).reset_index(drop=True)


SETS = {
    "max_awake": prep(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", "NESI"),
    "random_awake": prep(I / "random_awake_scores.csv", "NESI_RandomAwake"),
    "all_segments": prep(I / "all_segments_scores.csv", "NESI_RandomAwake"),
}


def seg_boot(x, y):
    """Segment-level bootstrap, as in the original analysis (seed 42)."""
    rng = np.random.default_rng(42)
    b = [stats.spearmanr(x[i], y[i])[0] for i in (rng.choice(len(x), len(x)) for _ in range(BOOT))]
    return np.percentile(b, [2.5, 97.5])


def pat_boot(d):
    rng = np.random.default_rng(2026)
    ids = [np.asarray(v) for v in d.groupby("BDSPPatientID").indices.values()]
    x, y, b = d.RawScore.values, d.NESI.values, []
    for _ in range(BOOT):
        i = np.concatenate([ids[j] for j in rng.integers(0, len(ids), len(ids))])
        if len(np.unique(x[i])) > 1:
            b.append(stats.spearmanr(x[i], y[i])[0])
    return np.percentile(b, [2.5, 97.5])


rows = []
for k, d in SETS.items():
    x, y = d.RawScore.values.astype(float), d.NESI.values.astype(float)
    sl, sh = seg_boot(x, y)
    pl, ph = pat_boot(d)
    per = d.groupby("BDSPPatientID").size()
    rows.append(dict(Set=k, segments=len(d), patients=len(per), max_segments_per_patient=per.max(),
                     rho=stats.spearmanr(x, y)[0], lo=pl, hi=ph, seg_lo=sl, seg_hi=sh))
r = pd.DataFrame(rows)
r.round(4).to_csv(OUT / "icans_segment_ablation.csv", index=False)
print(r.round(3).to_string(index=False))
