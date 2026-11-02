#!/usr/bin/env python3
"""Which score produced each released mortality result file?

Applies the coefficients stored in results_NESI_Death.pkl and results_GCS_Death.pkl to the
real GCS and NESI sequences of the first test split, and compares with the stored AUROC.
Needs intermediates/mortality/mortality_segments.csv from mortality.py features.
"""
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.argv = sys.argv[:1]
import mortality as M  # noqa: E402

p, ids, died = M.assemble("lexical")
_, te = train_test_split(np.array(ids), test_size=0.15, random_state=M.SEEDS[0], stratify=died)
t = p[p.pid.isin(set(te)) & (p.n >= 2)]
rows = []
for name in ["NESI", "GCS"]:
    r = pickle.load(open(M.REPO / f"NESI/DeathPrediction_NESIvsGCS/Results/results_{name}_Death.pkl", "rb"))[0][2]
    for score in ["GCS", "NESI"]:
        x = np.vstack([v[:2] for v in t[score]])
        rows.append(dict(file=f"results_{name}_Death.pkl", applied_to=score, auroc=round(roc_auc_score(t.died, x @ r["coef"].ravel() + r["intercept"][0]), 4), stored_auroc=round(r["auroc"], 4), patients=len(t), stored_patients=r["n_test"], coef=np.round(r["coef"].ravel(), 3).tolist(), intercept=round(float(r["intercept"][0]), 3)))
d = pd.DataFrame(rows)
M.TABLES.mkdir(parents=True, exist_ok=True)
d.to_csv(M.TABLES / "mortality_label_check.csv", index=False)
print(d.to_string(index=False))
