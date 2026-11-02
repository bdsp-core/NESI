#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path


current = Path(__file__).resolve()
NESI_ROOT = None

for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")

CSV_PATH = NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "Final_NESIscores_full_dataset_result.csv"

df = pd.read_csv(CSV_PATH)
df["Split"] = df["Split"].astype(str).str.strip().str.lower()
df["Dataset"] = df["Dataset"].astype(str).str.strip().str.upper()
df["RawScore"] = pd.to_numeric(df["RawScore"], errors="coerce")
df["NESI"] = pd.to_numeric(df["NESI"], errors="coerce")
df = df[(df["Split"] == "test") & df["Dataset"].isin(["RASS", "GCS", "CAMS", "ICANS"])]
df = df.dropna(subset=["RawScore", "NESI"])

results = []

for dataset in ["RASS", "GCS", "CAMS", "ICANS"]:
    d = df[df["Dataset"] == dataset]

    for score, g in d.groupby("RawScore", sort=True):
        q1, median, q3 = np.percentile(g["NESI"], [25, 50, 75])

        results.append({
            "Dataset": dataset,
            "Clinical Score": int(score),
            "N EEGs": len(g),
            "N Patients": g["BDSPPatientID"].nunique(),
            "NESI Median (Q1–Q3)": f"{median:.3f} ({q1:.3f}–{q3:.3f})"
        })

df_table = pd.DataFrame(results)

for dataset in ["RASS", "GCS", "CAMS", "ICANS"]:
    print(f"\n{'=' * 65}")
    print(f"{dataset} — TEST SET: NESI MEDIAN (IQR) BY CLINICAL SCORE")
    print(f"{'=' * 65}")
    print(
        df_table[df_table["Dataset"] == dataset]
        .drop(columns="Dataset")
        .to_string(index=False)
    )


# In[ ]:




