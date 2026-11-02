#!/usr/bin/env python3
"""Per-sheet patient overlap with MORGOTH's own train/test list (R3.13).

datasets_deidentified_list.xlsx: s3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/.
Saves a compact patient table to intermediates so the numbers regenerate without the workbook.
"""
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ["NESI_DATA_ROOT"])
LISTS = ROOT / "intermediates" / "morgoth_lists"
CACHE = LISTS / "morgoth_patient_roles.csv"
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def norm(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


if CACHE.exists():
    roles = pd.read_csv(CACHE, dtype=str)
else:
    x = pd.ExcelFile(LISTS / "datasets_deidentified_list.xlsx")
    parts = []
    for s in x.sheet_names:
        d = x.parse(s, dtype=str)
        id_col = next((c for c in d.columns if c.lower() in ("bdsp_mrn", "bdsppatientid", "bdspid", "patientid", "pid", "bdsp_patient_id", "mrn")), None)
        role = next((c for c in d.columns if c.lower() == "morgoth"), None)
        print(f"{s}: rows {len(d)}, id {id_col}, role {role}", flush=True)
        if id_col:
            parts.append(pd.DataFrame(dict(sheet=s, pid=norm(d[id_col]), site=d.get("SiteID", d.get("siteid")), role=d[role].str.lower().str.strip() if role else "unlisted", setting=d.get("eeg_setting"))))
    roles = pd.concat(parts).drop_duplicates()
    roles.to_csv(CACHE, index=False)

f = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
f["pid"] = norm(f.BDSPPatientID)
rows = []
for s, g in f.groupby("Split"):
    u = set(g.pid)
    for (sheet, role), r in roles.groupby(["sheet", "role"]):
        rows.append(dict(Split=s, sheet=sheet, role=role, sheet_patients=r.pid.nunique(), overlap=len(u & set(r.pid))))
o = pd.DataFrame(rows)
o.to_csv(OUT / "morgoth_overlap_by_sheet.csv", index=False)
print(o[o.Split == "Test"].to_string(index=False))
for role in sorted(roles.role.dropna().unique()):
    r = set(roles[roles.role == role].pid)
    print(role, "patients", len(r), {s: len(set(g.pid) & r) for s, g in f.groupby("Split")})

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

heedb = roles[roles.sheet == "HEEDB_All"]
PRE = set(heedb[heedb.role == "pretrain"].pid)
HTRAIN = set(heedb[heedb.role == "train"].pid)
# Supervised training for the IIIC and spike heads came from further sheets
ALLTRAIN = set(roles[roles.role == "train"].pid) | set(roles[roles.sheet == "SN2_Train_Demographic"].pid)


def rho_ci(d, boot=1000):
    rng = np.random.default_rng(2026)
    gr = {k: np.asarray(v) for k, v in d.groupby("pid").indices.items()}
    keys, y, x, b = list(gr), d.TransformedScore.values, d.NESI.values, []
    for _ in range(boot):
        i = np.concatenate([gr[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        if len(np.unique(y[i])) > 1:
            b.append(abs(stats.spearmanr(y[i], x[i])[0]))
    return abs(stats.spearmanr(y, x)[0]), np.percentile(b, 2.5), np.percentile(b, 97.5)


sets = {"Pretraining": PRE, "Training (HEEDB_All)": HTRAIN, "Training (all heads)": ALLTRAIN, "Pretraining or training (HEEDB_All)": PRE | HTRAIN, "Pretraining or training (all heads)": PRE | ALLTRAIN}
c = pd.DataFrame([{"Set": k, "MORGOTH patients": len(v), **{s: len(set(g.pid) & v) for s, g in f.groupby("Split")}} for k, v in sets.items()])
c.to_csv(OUT / "morgoth_overlap_counts.csv", index=False)
print(c.to_string(index=False))
t = f[f.Split == "Test"]
rows = []
for ds, g in t.groupby("Dataset"):
    a = rho_ci(g)
    row = dict(Scale=ds, Patients=g.pid.nunique(), rho=a[0], lo=a[1], hi=a[2])
    for lab, ex in [("heedb", PRE | HTRAIN), ("all", PRE | ALLTRAIN)]:
        k = g[~g.pid.isin(ex)]
        b = rho_ci(k)
        row.update({f"patients_excl_{lab}": k.pid.nunique(), f"segments_excl_{lab}": len(k), f"rho_excl_{lab}": b[0], f"lo_{lab}": b[1], f"hi_{lab}": b[2]})
    rows.append(row)
e = pd.DataFrame(rows).round(3)
e.to_csv(OUT / "morgoth_overlap_rho.csv", index=False)
print(e.T.to_string())
sites = heedb[heedb.role.isin(["pretrain", "train"])].groupby(["role", "site"]).pid.nunique().to_dict()
print("sites", sites)
meta_sites = roles[roles.role == "train"].groupby("site").pid.nunique().to_dict()
print("train sites all sheets", meta_sites)
