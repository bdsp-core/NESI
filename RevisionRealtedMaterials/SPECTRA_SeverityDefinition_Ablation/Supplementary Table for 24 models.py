#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


folder = Path(NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" )


files = {
    "RASS": "RASS24_model_Result.csv",
    "GCS": "GC24_model_Result.csv",
    "CAM-S": "CAMS24_model_Result.csv",
    "ICANS": "ICANS24_model_Result.csv"
}

table_rho = pd.DataFrame({"Model": [f"M{i}" for i in range(1, 25)]})
table_p = table_rho.copy()

for scale, filename in files.items():
    df = pd.read_csv(folder / filename)
    df["Model"] = df["Model"].replace({"Proposed NESI": "M1"})
    df["Model"] = df["Model"].str.replace(r"^Model (\d+)$", r"M\1", regex=True)
    df = df.set_index("Model").reindex([f"M{i}" for i in range(1, 25)])

    table_rho[f"{scale} ρ (95% CI)"] = [
        f"{r.Mean_Rho:.2f} ({r.CI_Lower:.2f}–{r.CI_Upper:.2f})"
        for r in df.itertuples()
    ]
    table_p[f"{scale} p (Holm)"] = [
        "—" if pd.isna(p) else f"{p:.2f}"
        for p in df["p_Holm"]
    ]

table_rho.loc[0, "Model"] = "M1 (Proposed)"
table_p.loc[0, "Model"] = "M1 (Proposed)"

table_rho.to_csv(folder / "Supplementary_Table_Spearman.csv", index=False)

print(table_rho.to_string(index=False))


# In[11]:


import pandas as pd
from pathlib import Path

folder = Path(NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" )

files = {
    "RASS": "RASS24_model_Result.csv",
    "GCS": "GC24_model_Result.csv",
    "CAM-S": "CAMS24_model_Result.csv",
    "ICANS": "ICANS24_model_Result.csv"
}

def create_holm_table(folder, files):
    table = pd.DataFrame({"Model": [f"M{i}" for i in range(1, 25)]})

    for scale, filename in files.items():
        df = pd.read_csv(Path(folder) / filename)
        df["Model"] = df["Model"].replace({"Proposed NESI": "M1"})
        df["Model"] = df["Model"].str.replace(r"^Model (\d+)$", r"M\1", regex=True)
        df = df.set_index("Model").reindex([f"M{i}" for i in range(1, 25)])

        proposed_rho = df.loc["M1", "Mean_Rho"]

        table[f"{scale} p (Holm)"] = [
            "—" if model == "M1" or pd.isna(row["p_Holm"]) else
            (f"p < 0.05* ({'M1' if proposed_rho > row['Mean_Rho'] else model})"
             if row["p_Holm"] < 0.05
             else f"{row['p_Holm']:.3f}")
            for model, row in df.iterrows()
        ]

    table.loc[0, "Model"] = "M1 (Proposed)"
    output = Path(folder) / "Supplementary_Table_Holm_With_Winner.csv"
    table.to_csv(output, index=False)

    print(table.to_string(index=False))
    print(f"\nSaved: {output}")
    return table

table_holm = create_holm_table(folder, files)