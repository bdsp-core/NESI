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

# ---------------- Machine Learning (sklearn) ----------------
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score, average_precision_score
)
from scipy.stats import spearmanr, wilcoxon
import scipy.io as sio
import mat73
# ---------------- Deep Learning (PyTorch) ----------------
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torchsummary import summary


# # **load Result CSV**

# In[2]:


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

FeatureCompare_Root = NESI_ROOT / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI"
NESI_result_path = NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "Final_NESIscores_full_dataset_result.csv"
Result_path = NESI_ROOT / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI" / "Results_NESI_from_qEEGFea"

files = {
    'NESI_from3qEEGFea': 'Final_NESI_AllqEEGFea.csv',
    'NESI_fromRelativeDelta': 'Final_NESI_RelativeDelta.csv',
    'NESI_fromDeltaAlpha': 'Final_NESI_DeltaAlpha.csv',
    'NESI_fromSEF95': 'Final_NESI_SEF95.csv'
}

metadata_cols = ['BDSPPatientID', 'Dataset', 'MorgothOutputFilename',
                 'RawScore', 'GroupedScore', 'TransformedScore', 'Split', 'NESI']

merge_cols = ['BDSPPatientID', 'Dataset', 'MorgothOutputFilename']

df_combined = None

for feature_col, filename in files.items():
    df = pd.read_csv(Result_path / filename, low_memory=False)

    if df_combined is None:
        df_combined = df[metadata_cols + [feature_col]].copy()
    else:
        df_combined = df_combined.merge(
            df[merge_cols + [feature_col]],
            on=merge_cols,
            how='left',
            validate='one_to_one'
        )

final_cols = metadata_cols + list(files.keys())
df_combined = df_combined[final_cols]

print("Final shape:", df_combined.shape)
print(df_combined[final_cols[8:]].isna().sum())
print(df_combined.head())



# # **Result Vidulaization**

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

def plot_qEEG_baseline_vs_NESI(df_combined, save_path=None, n_boot=1000, seed=42,
                               subplot_ylims=None, row_ylabels=None):
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    datasets = ["RASS", "GCS", "CAMS", "ICANS"]
    features = {
        "NESI_fromDeltaAlpha": "DeltaAlpha NESI",
        "NESI_fromRelativeDelta": "RelativeDelta NESI",
        "NESI_fromSEF95": "SEF95 NESI",
        "NESI_from3qEEGFea": "NESI allfea",
        "NESI": "Proposed NESI"
    }
    dataset_colors = {
        "RASS": "#7EB0D5", "GCS": "#F4B860",
        "CAMS": "#3FB8A5", "ICANS": "#C77DB0"
    }
    tick_rotation = {"RASS": 0, "GCS": 45, "CAMS": 0, "ICANS": 0}
    subplot_ylims = subplot_ylims or {}
    row_ylabels = row_ylabels or {}

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

    def bootstrap_spearman(y, x):
        rng = np.random.default_rng(seed)
        n = len(y)
        corrs = np.full(n_boot, np.nan)
        for k in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            if np.unique(y[idx]).size > 1 and np.unique(x[idx]).size > 1:
                corrs[k] = spearmanr(y[idx], x[idx]).statistic
        return np.nanmean(corrs), np.nanpercentile(corrs, [2.5, 97.5])

    def group_data(values, yraw, levels):
        grouped, counts = [], []
        for level in levels:
            vals = values[yraw == level]
            vals = vals[np.isfinite(vals)]
            grouped.append(vals if len(vals) else np.array([np.nan]))
            counts.append(len(vals))
        return grouped, counts

    df_test = df_combined[df_combined["Split"] == "Test"].copy()
    fig, axes = plt.subplots(5, 4, figsize=(22, 27), sharey=False)
    stats_results = []

    for i, (feature_col, feature_name) in enumerate(features.items()):
        for j, dataset_name in enumerate(datasets):
            ax = axes[i, j]
            color = dataset_colors[dataset_name]
            df = df_test[df_test["Dataset"] == dataset_name].copy()

            required = ["RawScore", "TransformedScore", feature_col]
            data = df[required].copy()
            for col in required:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.replace([np.inf, -np.inf], np.nan).dropna()

            yraw = data["RawScore"].to_numpy()
            ytrue = data["TransformedScore"].to_numpy(dtype=float)
            values = data[feature_col].to_numpy(dtype=float)

            if dataset_name == "RASS":
                levels = [-5, -4, -3, -2, -1, 0]
            elif dataset_name == "GCS":
                levels = list(range(3, 16))
            else:
                levels = sorted(data["RawScore"].unique())

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
            tick_labels = [
                f"{int(level) if float(level).is_integer() else level}\nn={n}"
                for level, n in zip(levels, counts)
            ]

            rotation = tick_rotation[dataset_name]
            if rotation == 0:
                ax.set_xticklabels(tick_labels, fontsize=10)
            else:
                ax.set_xticklabels(
                    tick_labels, fontsize=10, rotation=rotation,
                    ha="right", rotation_mode="anchor"
                )

            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            if subplot_ylims.get((i, j)) is not None:
                ax.set_ylim(subplot_ylims[(i, j)])

            rho, ci = bootstrap_spearman(ytrue, values)
            rho_text = f"{rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
            display_name = "CAM-S" if dataset_name == "CAMS" else dataset_name

            ax.set_title(
                f"{display_name}\n" + r"$\mathbf{\rho}$ = " + rho_text,
                fontweight="bold", fontsize=15
            )

            if j == 0:
                ax.set_ylabel(
                    row_ylabels.get(i, feature_name),
                    fontweight="normal", fontsize=17
                )

            if i == 4:
                ax.set_xlabel(f"{display_name} score", fontweight="normal", fontsize=15)

            stats_results.append({
                "Dataset": display_name,
                "Feature": feature_name,
                "Spearman": rho,
                "CI_Lower": ci[0],
                "CI_Upper": ci[1],
                "N": len(data)
            })

    fig.suptitle(
        "Comparison of NESI Derived from Conventional qEEG Features and the Proposed NESI\n"
        "Across Clinical Neurological Severity Scales",
        fontsize=21, fontweight="bold", y=0.975
    )

    plt.tight_layout(rect=[0.025, 0.065, 1, 0.955], h_pad=3.0, w_pad=1.5)

    for i, letter in enumerate(["A", "B", "C", "D", "E"]):
        pos = axes[i, 0].get_position()
        fig.text(0.005, pos.y1 + 0.005, letter, fontsize=20, fontweight="bold")

    fig.text(
        0.5, 0.025,
        "Boxes represent the median and interquartile range (IQR); whiskers extend to 1.5×IQR; points indicate outliers.\n"
        "n denotes the number of EEG observations at each clinical score level. Higher NESI values indicate more severe neurological dysfunction.\n"
        "Spearman's ρ and corresponding 95% confidence intervals were estimated using 1,000 bootstrap iterations.",
        ha="center", va="center", fontsize=15, color="black", linespacing=1.6
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()

    df_corr_results = pd.DataFrame(stats_results)
    df_corr_results = df_corr_results[
        ["Dataset", "Feature", "Spearman", "CI_Lower", "CI_Upper", "N"]
    ]

    print(f"\nSPEARMAN CORRELATION RESULTS ({n_boot} Bootstrap, 95% CI)")
    print(df_corr_results)

    return df_corr_results




row_ylabels = {
    0: "NESI from Delta/Alpha Ratio",
    1: "NESI from Relative Delta Power",
    2: "NESI from SEF95",
    3: "NESI from Combined qEEG Features",
    4: "Proposed NESI"
}

subplot_ylims = {
    # A: DeltaAlpha NESI
    (0, 0): (-2, 3), (0, 1): (-2.8, 3), (0, 2): None, (0, 3): None,

    # B: RelativeDelta NESI
    (1, 0): None, (1, 1): None, (1, 2): None, (1, 3): None,

    # C: SEF95 NESI
    (2, 0): None, (2, 1): None, (2, 2): None, (2, 3): None,

    # D: Combined qEEG NESI
    (3, 0): (-3.5, 3.5), (3, 1): None, (3, 2): None, (3, 3): None,

    # E: Proposed NESI
    (4, 0): None, (4, 1): None, (4, 2): None, (4, 3): None
}

df_corr_results = plot_qEEG_baseline_vs_NESI(
    df_combined,
    save_path=NESI_ROOT / "RevisionRelatedMaterials" / "TraditionalFeaturevsNESI" / "qEEG_based_NESI.png",
    n_boot=1000,
    seed=42,
    subplot_ylims=subplot_ylims,
    row_ylabels=row_ylabels
)
