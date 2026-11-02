#!/usr/bin/env python
# coding: utf-8

# # **Load NESI's Split CSV Metadata file**

# In[1]:


import pandas as pd
from pathlib import Path


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

nesi_path = NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "Final_NESIscores_full_dataset_result.csv"
df_NESI = pd.read_csv(nesi_path)

df_NESI_test_original = df_NESI[
    df_NESI["Split"].astype("string").str.strip().str.lower() == "test"
].copy()

print("*"*50)
print("ORIGINAL NESI TEST SET - DATASET-WISE STATISTICS")
print("*"*50)
print("Total Test Rows:", len(df_NESI_test_original))
print("Total Unique Subjects:", df_NESI_test_original["BDSPPatientID"].nunique())

stats_original = df_NESI_test_original.groupby("Dataset").agg(
    EEG_Rows=("BDSPPatientID", "size"),
    Unique_Subjects=("BDSPPatientID", "nunique")
).reindex(["RASS", "GCS", "CAMS", "ICANS"], fill_value=0)

print(stats_original)


# # **Load MORGOTH Split CSV Metadata file**

# In[2]:


excel_path = NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "00_datasets-scx.xlsx"
df_HEEDB = pd.read_excel(excel_path, sheet_name="HEEDB_All")

print('*'*40)
print('HEEDB Original File')
print('*'*40)
print("Shape:", df_HEEDB.shape)
print("Columns:", df_HEEDB.columns.tolist())

df_HEEDB_filter = df_HEEDB[["Morgoth", "BDSPPatientID", "SiteID", "SexDSC", "AgeAtVisitAvg"]].copy()

print('*'*40)
print('HEEDB Filtered File')
print('*'*40)
print("Shape:", df_HEEDB_filter.shape)

df_HEEDB_filter2 = df_HEEDB_filter[df_HEEDB_filter["Morgoth"].isin(["test", "train"])].copy() #, "pretrain"

print('*'*40)
print("Shape:", df_HEEDB_filter2.shape)
print("Morgoth counts:")
print(df_HEEDB_filter2["Morgoth"].value_counts())


# # **Statistics of HEEDB's Patient Split**

# In[5]:


print('*'*40)
print('HEEDB Unique Patient Statistics')
print('*'*40)

print("Total rows:", len(df_HEEDB_filter2))
print("Total unique patients:", df_HEEDB_filter2["BDSPPatientID"].nunique())

print("\nUnique patients by Morgoth subset:")
print(df_HEEDB_filter2.groupby("Morgoth")["BDSPPatientID"].nunique())

pretrain_ids = set(df_HEEDB_filter2.loc[df_HEEDB_filter2["Morgoth"] == "pretrain", "BDSPPatientID"].dropna())
train_ids = set(df_HEEDB_filter2.loc[df_HEEDB_filter2["Morgoth"] == "train", "BDSPPatientID"].dropna())
test_ids = set(df_HEEDB_filter2.loc[df_HEEDB_filter2["Morgoth"] == "test", "BDSPPatientID"].dropna())

print("\nPatients appearing in both Train and PreTrain:", len(pretrain_ids & test_ids))
print("\nPatients appearing in both Train and Test:", len(train_ids & test_ids))
print("\nPatients appearing in both PreTrain and Test:", len(pretrain_ids & test_ids))


# # **Overlap Statistics of NESI's Test and MORGOTH's Train set**

# In[4]:

print('*'*50)
print('NESI Dataset')
print('*'*50)
print("Shape:", df_NESI.shape)
print("Columns:", df_NESI.columns.tolist())

df_MORGOTH = df_HEEDB[["Morgoth", "BDSPPatientID"]].copy()

df_NESI["BDSPPatientID"] = df_NESI["BDSPPatientID"].astype("string").str.strip()
df_MORGOTH["BDSPPatientID"] = df_MORGOTH["BDSPPatientID"].astype("string").str.strip()

nesi_splits = ["Train", "Val", "Test"]
morgoth_splits = ["pretrain", "train", "test"]

df_NESI["Split"] = df_NESI["Split"].astype("string").str.strip().str.lower()
df_MORGOTH["Morgoth"] = df_MORGOTH["Morgoth"].astype("string").str.strip().str.lower()

results = []
for nesi_split in nesi_splits:
    nesi_ids = set(df_NESI.loc[df_NESI["Split"] == nesi_split.lower(), "BDSPPatientID"].dropna())
    row = {"NESI Split": nesi_split, "NESI Unique Patients": len(nesi_ids)}
    for morgoth_split in morgoth_splits:
        morgoth_ids = set(df_MORGOTH.loc[df_MORGOTH["Morgoth"] == morgoth_split, "BDSPPatientID"].dropna())
        row[f"MORGOTH {morgoth_split.capitalize()}"] = len(nesi_ids & morgoth_ids)
    results.append(row)

df_overlap = pd.DataFrame(results)

print('*'*50)
print('NESI vs MORGOTH - Unique Patient Overlap')
print('*'*50)
print(df_overlap)


# # **Sensitivity Analysis**
# - **Analysis 1: Exclude MORGOTH	Pretrain + Train**
# - **Analysis 2: Exclude MORGOTH Train**

# In[5]:


import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr


def run_sensitivity_analysis(analysis="both", n_boot=1000, seed=42):
    if analysis not in [1, 2, "both"]:
        raise ValueError("analysis must be 1, 2, or 'both'")

    df_NESI = pd.read_csv(nesi_path)
    df_HEEDB = pd.read_excel(excel_path, sheet_name="HEEDB_All")

    for df in [df_NESI, df_HEEDB]:
        df["BDSPPatientID"] = df["BDSPPatientID"].astype("string").str.strip()

    df_NESI["Split"] = df_NESI["Split"].astype("string").str.strip().str.lower()
    df_NESI["Dataset"] = df_NESI["Dataset"].astype("string").str.strip().str.upper()
    df_HEEDB["Morgoth"] = df_HEEDB["Morgoth"].astype("string").str.strip().str.lower()

    df_test = df_NESI[df_NESI["Split"] == "test"].copy()

    pretrain_ids = set(df_HEEDB.loc[df_HEEDB["Morgoth"] == "pretrain", "BDSPPatientID"].dropna())
    train_ids = set(df_HEEDB.loc[df_HEEDB["Morgoth"] == "train", "BDSPPatientID"].dropna())

    cohorts = {"Original": df_test}
    if analysis in [1, "both"]:
        cohorts["Approach 1"] = df_test[~df_test["BDSPPatientID"].isin(pretrain_ids | train_ids)].copy()
    if analysis in [2, "both"]:
        cohorts["Approach 2"] = df_test[~df_test["BDSPPatientID"].isin(train_ids)].copy()

    print("*"*65)
    print("NESI TEST SET - MORGOTH OVERLAP SENSITIVITY ANALYSIS")
    print("*"*65)
    print("Approach 1: Exclude MORGOTH Pretrain + Train")
    print("Approach 2: Exclude MORGOTH Train only")
    print(f"Bootstrap: {n_boot} iterations | Seed: {seed}")
    print(f"Original NESI Test: {len(df_test)} EEG rows | {df_test['BDSPPatientID'].nunique()} patients")

    results = []
    for dataset in ["RASS", "GCS", "CAMS", "ICANS"]:
        print(f"\n{'='*55}\n{dataset}\n{'='*55}")
        for name, df in cohorts.items():
            d = df[df["Dataset"] == dataset].copy()
            d["TransformedScore"] = pd.to_numeric(d["TransformedScore"], errors="coerce")
            d["NESI"] = pd.to_numeric(d["NESI"], errors="coerce")
            d = d.dropna(subset=["TransformedScore", "NESI"])

            y_true = d["TransformedScore"].to_numpy()
            y_nesi = d["NESI"].to_numpy()
            n = len(d)
            n_subjects = d["BDSPPatientID"].nunique()

            if n < 2 or np.unique(y_true).size < 2 or np.unique(y_nesi).size < 2:
                print(f"{name}: Insufficient data | EEGs={n}, Subjects={n_subjects}")
                continue

            rng = np.random.default_rng(seed)
            corrs = np.full(n_boot, np.nan)

            for i in range(n_boot):
                idx = rng.choice(n, n, replace=True)
                corrs[i] = spearmanr(y_true[idx], y_nesi[idx]).statistic

            corrs = corrs[np.isfinite(corrs)]
            if len(corrs) == 0:
                print(f"{name}: No valid bootstrap correlations")
                continue

            mean_rho = np.mean(corrs)
            ci = np.percentile(corrs, [2.5, 97.5])

            results.append({
                "Dataset": dataset,
                "Cohort": name,
                "EEG_Rows": n,
                "Unique_Subjects": n_subjects,
                "Mean_Spearman": mean_rho,
                "CI_Lower": ci[0],
                "CI_Upper": ci[1]
            })

            print(f"{name}: {mean_rho:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | EEGs={n}, Subjects={n_subjects}")

        r = [x for x in results if x["Dataset"] == dataset]
        original = next((x for x in r if x["Cohort"] == "Original"), None)
        if original:
            for x in r:
                if x["Cohort"] != "Original":
                    delta = x["Mean_Spearman"] - original["Mean_Spearman"]
                    print(f"Δρ ({x['Cohort']} - Original): {delta:+.4f}")

    df_results = pd.DataFrame(results)
    print(f"\n{'*'*65}\nFINAL SUMMARY\n{'*'*65}")
    print(df_results.round(4))
    return df_results

df_sensitivity_results = run_sensitivity_analysis(analysis="both")


# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr


def run_sensitivity_analysis(analysis="both", n_boot=1000, seed=42, fig_save_path=None):
    if analysis not in [1, 2, "both"]:
        raise ValueError("analysis must be 1, 2, or 'both'")

    df_NESI = pd.read_csv(nesi_path)
    df_HEEDB = pd.read_excel(excel_path, sheet_name="HEEDB_All")

    for df in [df_NESI, df_HEEDB]:
        df["BDSPPatientID"] = df["BDSPPatientID"].astype("string").str.strip()

    df_NESI["Split"] = df_NESI["Split"].astype("string").str.strip().str.lower()
    df_NESI["Dataset"] = df_NESI["Dataset"].astype("string").str.strip().str.upper()
    df_HEEDB["Morgoth"] = df_HEEDB["Morgoth"].astype("string").str.strip().str.lower()

    df_test = df_NESI[df_NESI["Split"] == "test"].copy()
    pretrain_ids = set(df_HEEDB.loc[df_HEEDB["Morgoth"] == "pretrain", "BDSPPatientID"].dropna())
    train_ids = set(df_HEEDB.loc[df_HEEDB["Morgoth"] == "train", "BDSPPatientID"].dropna())

    cohorts = {}
    if analysis in [1, "both"]:
        cohorts["Approach 1"] = df_test[~df_test["BDSPPatientID"].isin(pretrain_ids | train_ids)].copy()
    if analysis in [2, "both"]:
        cohorts["Approach 2"] = df_test[~df_test["BDSPPatientID"].isin(train_ids)].copy()
    cohorts["Original"] = df_test

    datasets = ["RASS", "GCS", "CAMS", "ICANS"]
    colors = {"RASS": "#2589BD", "GCS": "#F2B51B", "CAMS": "#20A98C", "ICANS": "#CA8DB0"}
    score_ranges = {"RASS": list(range(-5, 1)), "GCS": list(range(3, 16)),
                    "CAMS": list(range(8)), "ICANS": list(range(4))}

    row_labels = {
        "Approach 1": "Excluding MORGOTH-\noverlapping patients",
        "Approach 2": "Excluding MORGOTH\ntraining patients",
        "Original": "Original NESI test cohort\n(including overlap)"
    }

    print("*"*65)
    print("NESI TEST SET - MORGOTH OVERLAP SENSITIVITY ANALYSIS")
    print("*"*65)
    print("Approach 1: Exclude MORGOTH Pretrain + Train")
    print("Approach 2: Exclude MORGOTH Train only")
    print(f"Bootstrap: {n_boot} iterations | Seed: {seed}")
    print(f"Original NESI Test: {len(df_test)} EEG rows | {df_test['BDSPPatientID'].nunique()} patients")

    fig, axes = plt.subplots(len(cohorts), 4, figsize=(20, 5.2*len(cohorts)), squeeze=False)
    results = []

    for row, (name, df) in enumerate(cohorts.items()):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            d = df[df["Dataset"] == dataset].copy()
            d["RawScore"] = pd.to_numeric(d["RawScore"], errors="coerce")
            d["TransformedScore"] = pd.to_numeric(d["TransformedScore"], errors="coerce")
            d["NESI"] = pd.to_numeric(d["NESI"], errors="coerce")
            d = d.dropna(subset=["RawScore", "TransformedScore", "NESI"])

            y_true = d["TransformedScore"].to_numpy()
            y_nesi = d["NESI"].to_numpy()
            n = len(d)
            n_subjects = d["BDSPPatientID"].nunique()
            mean_rho, ci = np.nan, [np.nan, np.nan]

            if n >= 2 and np.unique(y_true).size >= 2 and np.unique(y_nesi).size >= 2:
                rng = np.random.default_rng(seed)
                corrs = np.full(n_boot, np.nan)
                for i in range(n_boot):
                    idx = rng.choice(n, n, replace=True)
                    corrs[i] = spearmanr(y_true[idx], y_nesi[idx]).statistic
                corrs = corrs[np.isfinite(corrs)]
                if len(corrs):
                    mean_rho = np.mean(corrs)
                    ci = np.percentile(corrs, [2.5, 97.5])

            results.append({
                "Dataset": dataset, "Cohort": name, "EEG_Rows": n,
                "Unique_Subjects": n_subjects, "Mean_Spearman": mean_rho,
                "CI_Lower": ci[0], "CI_Upper": ci[1]
            })

            groups = score_ranges[dataset]
            box_data = [d.loc[d["RawScore"] == s, "NESI"].to_numpy() for s in groups]
            valid = [(i+1, x) for i, x in enumerate(box_data) if len(x)]

            if valid:
                bp = ax.boxplot(
                    [x for _, x in valid],
                    positions=[p for p, _ in valid],
                    widths=0.62, patch_artist=True, whis=1.5,
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(edgecolor="black", linewidth=1),
                    whiskerprops=dict(color="black", linewidth=1),
                    capprops=dict(color="black", linewidth=1),
                    flierprops=dict(marker="o", markersize=2, markerfacecolor="gray",
                                    markeredgecolor="none", alpha=0.45)
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(colors[dataset])

            ax.set_xticks(np.arange(1, len(groups)+1))
            ax.set_xticklabels(groups, fontsize=9)
            ax.set_xlim(0.4, len(groups)+0.6)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if np.isfinite(mean_rho):
                ax.set_title(f"{dataset}\nρ = {mean_rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
                             fontsize=12, fontweight="bold", pad=8)
            else:
                ax.set_title(f"{dataset}\nρ = N/A", fontsize=12, fontweight="bold", pad=8)

            ax.set_xlabel(f"{dataset} score" if row == len(cohorts)-1 else "", fontsize=12, labelpad=30)
            if col == 0:
                ax.set_ylabel("Universal model's NESI", fontsize=12)

            for i, x in enumerate(box_data):
                ax.text(i+1, -0.075, f"n={len(x)}",
                        transform=ax.get_xaxis_transform(),
                        ha="right" if dataset == "GCS" else "center",
                        va="top", fontsize=10,
                        rotation=45 if dataset == "GCS" else 0,
                        clip_on=False)

            if dataset == "GCS":
                ax.tick_params(axis="x", labelsize=8)

        print(f"\n{'='*55}\n{name}\n{'='*55}")
        for r in results[-4:]:
            print(f"{r['Dataset']}: {r['Mean_Spearman']:.2f} "
                  f"[{r['CI_Lower']:.2f}, {r['CI_Upper']:.2f}] | "
                  f"EEGs={r['EEG_Rows']}, Subjects={r['Unique_Subjects']}")

    fig.subplots_adjust(left=0.13, right=0.99, top=0.94, bottom=0.12,
                        hspace=0.65, wspace=0.23)

    for row, name in enumerate(cohorts):
        pos = axes[row, 0].get_position()
        fig.text(0.075, (pos.y0+pos.y1)/2, row_labels[name], rotation=90,
                 ha="center", va="center", fontsize=13, fontweight="bold")

    fig.text(0.5, 0.025, "Boxes: median & IQR; whiskers: 1.5×IQR; points: outliers.",
             ha="center", fontsize=10, style="italic")

    if fig_save_path is not None:
        fig_save_path = Path(fig_save_path)
        fig_save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved: {fig_save_path}")

    plt.show()

    df_results = pd.DataFrame(results)
    print(f"\n{'*'*65}\nFINAL SUMMARY\n{'*'*65}")
    print(df_results.round(4))

    for dataset in datasets:
        r = df_results[df_results["Dataset"] == dataset]
        original = r[r["Cohort"] == "Original"]["Mean_Spearman"].iloc[0]
        for _, x in r.iterrows():
            if x["Cohort"] != "Original":
                print(f"{dataset} | Δρ ({x['Cohort']} - Original): {x['Mean_Spearman']-original:+.2f}")

    return df_results


# In[22]:


df_results = run_sensitivity_analysis(
    analysis=1,
    n_boot=1000,
    seed=42,
    fig_save_path=NESI_ROOT / "RevisionRelatedMaterials" / "Morgoth_vs_NESI_datasplit_Realted" / "MORGOTH_Overlap_Sensitivity_Analysis.png"
)


# In[ ]:




