#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---------------- Standard Libraries ----------------
import os
import pickle
from pathlib import Path
from datetime import datetime
import time
# ---------------- Data Handling ----------------
import numpy as np
import pandas as pd

# ---------------- Visualization ----------------
import matplotlib.pyplot as plt
import seaborn as sns
import mat73
from tqdm.auto import tqdm
# ---------------- Machine Learning (sklearn) ----------------
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score, average_precision_score
)
from scipy.stats import spearmanr, wilcoxon

# ---------------- Deep Learning (PyTorch) ----------------
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torchsummary import summary


# # **NESI model result for Individual Cohort**

current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")

NESI_result_path = NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "Final_NESIscores_full_dataset_result.csv"


FeatureCompare_Root = NESI_ROOT / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI"


#------------------ NESI model's results ---------------------
df_NESI_result = pd.read_csv(NESI_result_path)
df_NESI_test = df_NESI_result[df_NESI_result["Split"]=="Test"]

df_RASS_from_NESI = df_NESI_test[(df_NESI_test['Dataset']=='RASS')].reset_index(drop=True)
df_GCS_from_NESI = df_NESI_test[(df_NESI_test['Dataset']=='GCS')].reset_index(drop=True)
df_CAMS_from_NESI = df_NESI_test[(df_NESI_test['Dataset']=='CAMS')].reset_index(drop=True)
df_ICANS_from_NESI = df_NESI_test[(df_NESI_test['Dataset']=='ICANS')].reset_index(drop=True)


# # **RASS cohort NESI vs Traditional Features**


FeatureData_Root = FeatureCompare_Root / "CohortFeatureData"

df_NESI_result = pd.read_csv(NESI_result_path)
df_NESI_test = df_NESI_result[df_NESI_result["Split"] == "Test"].copy()
df_RASS_from_NESI = df_NESI_test[df_NESI_test["Dataset"] == "RASS"].reset_index(drop=True)

features = {
    "SEF95": "SEF95Median",
    "RelativeDelta": "RelativeDeltaMedian",
    "DeltaAlpha": "DeltaAlphaMedian"
}

def clean_filename(x):
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != 1:
            raise ValueError(f"Unexpected filename structure: {x}")
        x = x[0]
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return str(x).strip()

for feature, output_col in tqdm(features.items(), desc="Processing traditional EEG features", unit="feature"):
    mat_path = FeatureData_Root / f"RASS_All_{feature}.mat"
    mat = mat73.loadmat(str(mat_path))

    if mat["Filename"] is None:
        raise ValueError(f"{feature}: Filename could not be decoded by mat73")

    filenames = np.asarray([
        clean_filename(x) for x in tqdm(mat["Filename"], desc=f"{feature}: Filenames", leave=False)
    ])
    values = np.asarray(mat[feature], dtype=float)

    if values.shape == (19, len(filenames)):
        values = values.T

    if values.shape != (len(filenames), 19):
        raise ValueError(
            f"{feature}: Filename count={len(filenames)}, Feature shape={values.shape}"
        )

    if pd.Index(filenames).has_duplicates:
        raise ValueError(f"{feature}: Duplicate filenames found in MAT file")

    feature_map = pd.Series(np.nanmedian(values, axis=1), index=filenames)

    df_RASS_from_NESI[output_col] = (
        df_RASS_from_NESI["MorgothOutputFilename"]
        .astype(str)
        .str.strip()
        .map(feature_map)
    )

    matched = df_RASS_from_NESI[output_col].notna().sum()
    unmatched = len(df_RASS_from_NESI) - matched

    tqdm.write(f"{feature}: Total MAT EEGs={len(filenames)}, Matched={matched}, Unmatched={unmatched}")

print(f"\nFinal RASS Test DataFrame: {df_RASS_from_NESI.shape}")
print("Columns:", df_RASS_from_NESI.columns.tolist())

print(df_RASS_from_NESI.head())


# In[4]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from tqdm.auto import tqdm

y_true = df_RASS_from_NESI["TransformedScore"].values
features = ["NESI", "SEF95Median", "RelativeDeltaMedian", "DeltaAlphaMedian"]
feature_values = {f: df_RASS_from_NESI[f].values for f in features}

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000
rng = np.random.default_rng(42)

boot_corrs = {f: np.zeros(n_boot) for f in features}

for i in tqdm(range(n_boot), desc="Bootstrapping"):
    idx = rng.choice(n, n, replace=True)
    for f in features:
        boot_corrs[f][i], _ = spearmanr(y_true[idx], feature_values[f][idx])

# ---------------- stats ----------------
results = []

for f in features:
    mean_corr = np.mean(boot_corrs[f])
    ci = np.percentile(boot_corrs[f], [2.5, 97.5])

    results.append({
        "Feature": f,
        "Mean Spearman": mean_corr,
        "95% CI Lower": ci[0],
        "95% CI Upper": ci[1]
    })

df_corr_results = pd.DataFrame(results)

# ---------------- WILCOXON TEST ----------------
comparison_results = []

for f in features[1:]:
    stat, p_value = wilcoxon(
        boot_corrs["NESI"],
        boot_corrs[f],
        alternative="greater"
    )

    if p_value < 0.05:
        winner = "NESI significantly better"
    else:
        winner = "No significant difference (or traditional feature not worse)"

    comparison_results.append({
        "Comparison": f"NESI vs {f}",
        "NESI Mean Spearman": np.mean(boot_corrs["NESI"]),
        "Feature Mean Spearman": np.mean(boot_corrs[f]),
        "p-value": p_value,
        "Result": winner
    })

df_comparison_results = pd.DataFrame(comparison_results)

# ---------------- output ----------------
print(f"RASS Test EEGs: {n}")
print(f"Bootstrap iterations: {n_boot}")

print("\nSPEARMAN CORRELATION RESULTS")
print(df_corr_results)

print("\nSTATISTICAL TEST (Wilcoxon Rank Test, paired)")
print(df_comparison_results)


# # **CAMS cohort NESI vs Traditional Features**

# In[5]:


from pathlib import Path
import numpy as np
import pandas as pd
import mat73
from tqdm.auto import tqdm

FeatureData_Root = FeatureCompare_Root / "CohortFeatureData"

df_NESI_result = pd.read_csv(NESI_result_path)

df_NESI_test = df_NESI_result[df_NESI_result["Split"] == "Test"].copy()
df_CAMS_from_NESI = df_NESI_test[df_NESI_test["Dataset"] == "CAMS"].reset_index(drop=True)

features = {
    "SEF95": "SEF95Median",
    "RelativeDelta": "RelativeDeltaMedian",
    "DeltaAlpha": "DeltaAlphaMedian"
}

def clean_filename(x):
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != 1:
            raise ValueError(f"Unexpected filename structure: {x}")
        x = x[0]
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return str(x).strip()

for feature, output_col in tqdm(features.items(), desc="Processing traditional EEG features", unit="feature"):
    mat_path = FeatureData_Root / f"CAMS_All_{feature}.mat"
    mat = mat73.loadmat(str(mat_path))

    if mat["Filename"] is None:
        raise ValueError(f"{feature}: Filename could not be decoded by mat73")

    filenames = np.asarray([
        clean_filename(x) for x in tqdm(mat["Filename"], desc=f"{feature}: Filenames", leave=False)
    ])
    values = np.asarray(mat[feature], dtype=float)

    if values.shape == (19, len(filenames)):
        values = values.T

    if values.shape != (len(filenames), 19):
        raise ValueError(
            f"{feature}: Filename count={len(filenames)}, Feature shape={values.shape}"
        )

    if pd.Index(filenames).has_duplicates:
        raise ValueError(f"{feature}: Duplicate filenames found in MAT file")

    feature_map = pd.Series(np.nanmedian(values, axis=1), index=filenames)

    df_CAMS_from_NESI[output_col] = (
        df_CAMS_from_NESI["MorgothOutputFilename"]
        .astype(str)
        .str.strip()
        .map(feature_map)
    )

    matched = df_CAMS_from_NESI[output_col].notna().sum()
    unmatched = len(df_CAMS_from_NESI) - matched

    tqdm.write(f"{feature}: Total MAT EEGs={len(filenames)}, Matched={matched}, Unmatched={unmatched}")

print(f"\nFinal CAMS Test DataFrame: {df_CAMS_from_NESI.shape}")
print("Columns:", df_CAMS_from_NESI.columns.tolist())

print(df_CAMS_from_NESI)


# In[6]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from tqdm.auto import tqdm

y_true = df_CAMS_from_NESI["TransformedScore"].values
features = ["NESI", "SEF95Median", "RelativeDeltaMedian", "DeltaAlphaMedian"]
feature_values = {f: df_CAMS_from_NESI[f].values for f in features}

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000
rng = np.random.default_rng(42)

boot_corrs = {f: np.zeros(n_boot) for f in features}

for i in tqdm(range(n_boot), desc="Bootstrapping"):
    idx = rng.choice(n, n, replace=True)
    for f in features:
        boot_corrs[f][i], _ = spearmanr(y_true[idx], feature_values[f][idx])

# ---------------- stats ----------------
results = []

for f in features:
    mean_corr = np.mean(boot_corrs[f])
    ci = np.percentile(boot_corrs[f], [2.5, 97.5])

    results.append({
        "Feature": f,
        "Mean Spearman": mean_corr,
        "95% CI Lower": ci[0],
        "95% CI Upper": ci[1]
    })

df_corr_results = pd.DataFrame(results)

# ---------------- WILCOXON TEST ----------------
comparison_results = []

for f in features[1:]:
    stat, p_value = wilcoxon(
        boot_corrs["NESI"],
        boot_corrs[f],
        alternative="greater"
    )

    if p_value < 0.05:
        winner = "NESI significantly better"
    else:
        winner = "No significant difference (or traditional feature not worse)"

    comparison_results.append({
        "Comparison": f"NESI vs {f}",
        "NESI Mean Spearman": np.mean(boot_corrs["NESI"]),
        "Feature Mean Spearman": np.mean(boot_corrs[f]),
        "p-value": p_value,
        "Result": winner
    })

df_comparison_results = pd.DataFrame(comparison_results)

# ---------------- output ----------------
print(f"CAMS Test EEGs: {n}")
print(f"Bootstrap iterations: {n_boot}")

print("\nSPEARMAN CORRELATION RESULTS")
print(df_corr_results.round(4))

print("\nSTATISTICAL TEST (Wilcoxon Rank Test, paired)")
print(df_comparison_results)


# # **ICANS cohort NESI vs Traditional Features**

# In[7]:


from pathlib import Path
import numpy as np
import pandas as pd
import mat73
from tqdm.auto import tqdm

FeatureData_Root = FeatureCompare_Root / "CohortFeatureData"

df_NESI_result = pd.read_csv(NESI_result_path)

df_NESI_test = df_NESI_result[df_NESI_result["Split"] == "Test"].copy()
df_ICANS_from_NESI = df_NESI_test[df_NESI_test["Dataset"] == "ICANS"].reset_index(drop=True)

features = {
    "SEF95": "SEF95Median",
    "RelativeDelta": "RelativeDeltaMedian",
    "DeltaAlpha": "DeltaAlphaMedian"
}

def clean_filename(x):
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != 1:
            raise ValueError(f"Unexpected filename structure: {x}")
        x = x[0]
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return str(x).strip()

for feature, output_col in tqdm(features.items(), desc="Processing traditional EEG features", unit="feature"):
    mat_path = FeatureData_Root / f"ICANS_All_{feature}.mat"
    mat = mat73.loadmat(str(mat_path))

    if mat["Filename"] is None:
        raise ValueError(f"{feature}: Filename could not be decoded by mat73")

    filenames = np.asarray([
        clean_filename(x) for x in tqdm(mat["Filename"], desc=f"{feature}: Filenames", leave=False)
    ])
    values = np.asarray(mat[feature], dtype=float)

    if values.shape == (19, len(filenames)):
        values = values.T

    if values.shape != (len(filenames), 19):
        raise ValueError(
            f"{feature}: Filename count={len(filenames)}, Feature shape={values.shape}"
        )

    if pd.Index(filenames).has_duplicates:
        raise ValueError(f"{feature}: Duplicate filenames found in MAT file")

    feature_map = pd.Series(np.nanmedian(values, axis=1), index=filenames)

    df_ICANS_from_NESI[output_col] = (
        df_ICANS_from_NESI["MorgothOutputFilename"]
        .astype(str)
        .str.strip()
        .map(feature_map)
    )

    matched = df_ICANS_from_NESI[output_col].notna().sum()
    unmatched = len(df_ICANS_from_NESI) - matched

    tqdm.write(f"{feature}: Total MAT EEGs={len(filenames)}, Matched={matched}, Unmatched={unmatched}")

print(f"\nFinal ICANS Test DataFrame: {df_ICANS_from_NESI.shape}")
print("Columns:", df_ICANS_from_NESI.columns.tolist())

print(df_ICANS_from_NESI.head())


# In[8]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from tqdm.auto import tqdm

y_true = df_ICANS_from_NESI["TransformedScore"].values
features = ["NESI", "SEF95Median", "RelativeDeltaMedian", "DeltaAlphaMedian"]
feature_values = {f: df_ICANS_from_NESI[f].values for f in features}

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000
rng = np.random.default_rng(42)

boot_corrs = {f: np.zeros(n_boot) for f in features}

for i in tqdm(range(n_boot), desc="Bootstrapping"):
    idx = rng.choice(n, n, replace=True)
    for f in features:
        boot_corrs[f][i], _ = spearmanr(y_true[idx], feature_values[f][idx])

# ---------------- stats ----------------
results = []

for f in features:
    mean_corr = np.mean(boot_corrs[f])
    ci = np.percentile(boot_corrs[f], [2.5, 97.5])

    results.append({
        "Feature": f,
        "Mean Spearman": mean_corr,
        "95% CI Lower": ci[0],
        "95% CI Upper": ci[1]
    })

df_corr_results = pd.DataFrame(results)

# ---------------- WILCOXON TEST ----------------
comparison_results = []

for f in features[1:]:
    stat, p_value = wilcoxon(
        boot_corrs["NESI"],
        boot_corrs[f],
        alternative="greater"
    )

    if p_value < 0.05:
        winner = "NESI significantly better"
    else:
        winner = "No significant difference (or traditional feature not worse)"

    comparison_results.append({
        "Comparison": f"NESI vs {f}",
        "NESI Mean Spearman": np.mean(boot_corrs["NESI"]),
        "Feature Mean Spearman": np.mean(boot_corrs[f]),
        "p-value": p_value,
        "Result": winner
    })

df_comparison_results = pd.DataFrame(comparison_results)

# ---------------- output ----------------
print(f"ICANS Test EEGs: {n}")
print(f"Bootstrap iterations: {n_boot}")

print("\nSPEARMAN CORRELATION RESULTS")
print(df_corr_results)

print("\nSTATISTICAL TEST (Wilcoxon Rank Test, paired)")
print(df_comparison_results)


# # **GCS cohort NESI vs Traditional Features**

# In[9]:


from pathlib import Path
import numpy as np
import pandas as pd
import mat73
from tqdm.auto import tqdm

FeatureData_Root = FeatureCompare_Root / "CohortFeatureData"

df_NESI_result = pd.read_csv(NESI_result_path)

df_NESI_test = df_NESI_result[df_NESI_result["Split"] == "Test"].copy()
df_GCS_from_NESI = df_NESI_test[df_NESI_test["Dataset"] == "GCS"].reset_index(drop=True)

features = {
    "SEF95": "SEF95Median",
    "RelativeDelta": "RelativeDeltaMedian",
    "DeltaAlpha": "DeltaAlphaMedian"
}

def clean_filename(x):
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != 1:
            raise ValueError(f"Unexpected filename structure: {x}")
        x = x[0]
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return str(x).strip()

for feature, output_col in tqdm(features.items(), desc="Processing traditional EEG features", unit="feature"):
    mat_path = FeatureData_Root / f"GCS_Test_{feature}.mat"
    mat = mat73.loadmat(str(mat_path))

    if mat["Filename"] is None:
        raise ValueError(f"{feature}: Filename could not be decoded by mat73")

    filenames = np.asarray([
        clean_filename(x) for x in tqdm(mat["Filename"], desc=f"{feature}: Filenames", leave=False)
    ])
    values = np.asarray(mat[feature], dtype=float)

    if values.shape == (19, len(filenames)):
        values = values.T

    if values.shape != (len(filenames), 19):
        raise ValueError(
            f"{feature}: Filename count={len(filenames)}, Feature shape={values.shape}"
        )

    if pd.Index(filenames).has_duplicates:
        raise ValueError(f"{feature}: Duplicate filenames found in MAT file")

    feature_map = pd.Series(np.nanmedian(values, axis=1), index=filenames)

    df_GCS_from_NESI[output_col] = (
        df_GCS_from_NESI["MorgothOutputFilename"]
        .astype(str)
        .str.strip()
        .map(feature_map)
    )

    matched = df_GCS_from_NESI[output_col].notna().sum()
    unmatched = len(df_GCS_from_NESI) - matched

    tqdm.write(f"{feature}: Total MAT EEGs={len(filenames)}, Matched={matched}, Unmatched={unmatched}")

print(f"\nFinal GCS Test DataFrame: {df_GCS_from_NESI.shape}")
print("Columns:", df_GCS_from_NESI.columns.tolist())

print(df_GCS_from_NESI)


# In[10]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from tqdm.auto import tqdm

y_true = df_GCS_from_NESI["TransformedScore"].values
features = ["NESI", "SEF95Median", "RelativeDeltaMedian", "DeltaAlphaMedian"]
feature_values = {f: df_GCS_from_NESI[f].values for f in features}

# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000
rng = np.random.default_rng(42)

boot_corrs = {f: np.zeros(n_boot) for f in features}

for i in tqdm(range(n_boot), desc="Bootstrapping"):
    idx = rng.choice(n, n, replace=True)
    for f in features:
        boot_corrs[f][i], _ = spearmanr(y_true[idx], feature_values[f][idx])

# ---------------- stats ----------------
results = []

for f in features:
    mean_corr = np.mean(boot_corrs[f])
    ci = np.percentile(boot_corrs[f], [2.5, 97.5])

    results.append({
        "Feature": f,
        "Mean Spearman": mean_corr,
        "95% CI Lower": ci[0],
        "95% CI Upper": ci[1]
    })

df_corr_results = pd.DataFrame(results)

# ---------------- WILCOXON TEST ----------------
comparison_results = []

for f in features[1:]:
    stat, p_value = wilcoxon(
        boot_corrs["NESI"],
        boot_corrs[f],
        alternative="greater"
    )

    if p_value < 0.05:
        winner = "NESI significantly better"
    else:
        winner = "No significant difference (or traditional feature not worse)"

    comparison_results.append({
        "Comparison": f"NESI vs {f}",
        "NESI Mean Spearman": np.mean(boot_corrs["NESI"]),
        "Feature Mean Spearman": np.mean(boot_corrs[f]),
        "p-value": p_value,
        "Result": winner
    })

df_comparison_results = pd.DataFrame(comparison_results)

# ---------------- output ----------------
print(f"GCS Test EEGs: {n}")
print(f"Bootstrap iterations: {n_boot}")

print("\nSPEARMAN CORRELATION RESULTS")
print(df_corr_results)

print("\nSTATISTICAL TEST (Wilcoxon Rank Test, paired)")
print(df_comparison_results)


# # **Compare NESI vs Traditional Feature**

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
from tqdm.auto import tqdm

def plot_traditional_features_vs_NESI(
    df_RASS_from_NESI, df_GCS_from_NESI,
    df_CAMS_from_NESI, df_ICANS_from_NESI,
    save_path=None, n_boot=1000, panel_A_ylims=None
):
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    datasets = {
        "RASS": df_RASS_from_NESI,
        "GCS": df_GCS_from_NESI,
        "CAMS": df_CAMS_from_NESI,
        "ICANS": df_ICANS_from_NESI
    }
    features = {
        "DeltaAlphaMedian": "DeltaAlpha",
        "RelativeDeltaMedian": "RelativeDelta",
        "SEF95Median": "SEF95",
        "NESI": "NESI"
    }
    dataset_colors = {
        "RASS": "#7EB0D5", "GCS": "#F4B860",
        "CAMS": "#3FB8A5", "ICANS": "#C77DB0"
    }
    tick_rotation = {"RASS": 0, "GCS": 45, "CAMS": 0, "ICANS": 0}
    panel_A_ylims = panel_A_ylims or {}

    box_style = dict(
        patch_artist=True, showfliers=True,
        flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'gray',
                    'markeredgecolor': 'none', 'alpha': 0.3},
        widths=0.6,
        boxprops=dict(edgecolor='black', linewidth=1.3),
        whiskerprops=dict(color='black', linewidth=1.3),
        capprops=dict(color='black', linewidth=1.3),
        medianprops=dict(color='black', linewidth=2)
    )

    def bootstrap_spearman(y, x, seed=42):
        rng = np.random.default_rng(seed)
        n = len(y)
        corrs = np.zeros(n_boot)
        for k in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            corrs[k], _ = spearmanr(y[idx], x[idx])
        return np.nanmean(corrs), np.nanpercentile(corrs, [2.5, 97.5])

    def group_data(values, yraw, levels):
        grouped, counts = [], []
        for level in levels:
            vals = values[yraw == level]
            vals = vals[np.isfinite(vals)]
            grouped.append(vals if len(vals) else np.array([np.nan]))
            counts.append(len(vals))
        return grouped, counts

    fig, axes = plt.subplots(4, 4, figsize=(22, 21), sharey=False)
    stats_results = []

    for i, (feature_col, feature_name) in enumerate(features.items()):
        for j, (dataset_name, df) in enumerate(datasets.items()):
            ax = axes[i, j]
            color = dataset_colors[dataset_name]
            required = ["RawScore", "TransformedScore", feature_col]
            data = df[required].replace([np.inf, -np.inf], np.nan).dropna()

            yraw = data["RawScore"].to_numpy()
            ytrue = data["TransformedScore"].to_numpy(dtype=float)
            values = data[feature_col].to_numpy(dtype=float)

            if dataset_name == "RASS":
                levels = [-5, -4, -3, -2, -1, 0]
            elif dataset_name == "GCS":
                levels = list(range(3, 16))
            else:
                levels = sorted(df["RawScore"].dropna().unique())

            grouped, counts = group_data(values, yraw, levels)
            bp = ax.boxplot(grouped, **box_style)
            for box in bp["boxes"]:
                box.set_facecolor(color)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_edgecolor("black")
                ax.spines[spine].set_linewidth(1.5)

            ax.set_xticks(range(1, len(levels) + 1))
            tick_labels = [f"{level}\nn={n}" for level, n in zip(levels, counts)]
            rotation = tick_rotation[dataset_name]
            if rotation == 0:
                ax.set_xticklabels(tick_labels, fontsize=10)
            else:
                ax.set_xticklabels(tick_labels, fontsize=10, rotation=rotation,
                                   ha="right", rotation_mode="anchor")

            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            if i == 0 and panel_A_ylims.get(dataset_name) is not None:
                ax.set_ylim(panel_A_ylims[dataset_name])

            rho, ci = bootstrap_spearman(ytrue, values)
            rho_text = f"{rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
            display_name = "CAM-S" if dataset_name == "CAMS" else dataset_name
            ax.set_title(f"{display_name}\n" + r"$\mathbf{\rho}$ = " + rho_text,
                         fontweight="bold", fontsize=10)
            
            # ax.set_title(f"{dataset_name}\n" + r"$\mathbf{\rho}$ = " + rho_text,
            #              fontweight="bold", fontsize=15)

            if j == 0:
                ax.set_ylabel(feature_name, fontweight="normal", fontsize=10)
            if i == 3:
                ax.set_xlabel(f"{dataset_name} score", fontweight="normal", fontsize=10)

            stats_results.append({
                "Dataset": dataset_name, "Feature": feature_name,
                "Spearman": rho, "CI_Lower": ci[0],
                "CI_Upper": ci[1], "N": len(data)
            })

    for y, letter in zip([0.975, 0.735, 0.495, 0.255], ["A", "B", "C", "D"]):
        fig.text(0.005, y, letter, fontsize=10, fontweight="bold")

    fig.text(0.5, 0.005,
             "Boxes: median & IQR; whiskers: 1.5×IQR; points: outliers.",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0.015, 0.02, 1, 0.985], h_pad=3.0, w_pad=1.5)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()
    return pd.DataFrame(stats_results)



# In[12]:


# ---------------- CALL ----------------
df_figure_stats = plot_traditional_features_vs_NESI(
    df_RASS_from_NESI,
    df_GCS_from_NESI,
    df_CAMS_from_NESI,
    df_ICANS_from_NESI,
    save_path=FeatureCompare_Root / "FeatureVsNESI_CorrealationPlot.png",
    n_boot=1000,
    panel_A_ylims={
        "RASS": (-3, 150),
        "GCS": (-3, 150),
        "CAMS": (-3, 100),
        "ICANS": (-3, 60)
    }
)



# In[ ]:




