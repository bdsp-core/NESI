#!/usr/bin/env python
# coding: utf-8

# In[4]:


# ---------------- Standard Libraries ----------------
import os
import pickle
from pathlib import Path
from datetime import datetime
import time
# ---------------- Data Handling ----------------
import numpy as np
import pandas as pd
from pathlib import Path
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


# In[5]:
current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")

ICANS_Result_dir = NESI_ROOT / "RevisionRelatedMaterials" / "ICANS_Awake_Segment_Ablation"
ICANS_random_awake_csv = ICANS_Result_dir/ "RandomAwakeBasedAnalysis" / "RandomAwake_Results.csv"
ICANS_all_day_cluster_result_csv = ICANS_Result_dir/ "AllSegmentsAnalysis" / "AllSegmenst_DayCluster_Results.csv"
NESI_ICANS_csv = NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "Final_NESIscores_full_dataset_result.csv"



df_all_proposed_nesi = pd.read_csv(NESI_ICANS_csv)
df_icans_nesi_proposed =  df_all_proposed_nesi[df_all_proposed_nesi['Dataset']=='ICANS']

df_icans_random_result = pd.read_csv(ICANS_random_awake_csv)
df_icans_allday_result = pd.read_csv(ICANS_all_day_cluster_result_csv)

print('*'*40)
print(df_icans_nesi_proposed.columns)

print('*'*40)
print(df_icans_random_result.columns)

print('*'*40)
print(df_icans_allday_result.columns)


# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
from IPython.display import display

def plot_ICANS_segment_ablation(df_random, df_allday, df_proposed, save_path=None,
                               n_boot=1000, seed=42, figsize=(19, 6.5),
                               subplot_ylims=None, subplot_height=0.65):

    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    approaches = [
        ("Random Awake Selection", df_random, "NESI_RandomAwake", "#D9A3CA"),
        ("All-Day Segments", df_allday, "NESI_RandomAwake", "#C77DB0"),
        ("Proposed Maximum-Awake Selection", df_proposed, "NESI", "#A95A96")
    ]

    subplot_ylims = subplot_ylims or {}

    def bootstrap_spearman(y_true, y_pred):
        n = len(y_true)
        corrs = np.zeros(n_boot)
        rng = np.random.default_rng(seed)

        for i in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            corrs[i], _ = spearmanr(y_true[idx], y_pred[idx])

        mean_rho = np.mean(corrs)
        ci = np.percentile(corrs, [2.5, 97.5])

        return mean_rho, ci

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    stats_results = []

    box_style = dict(
        patch_artist=True,
        showfliers=True,
        widths=0.55,
        whis=1.5,
        flierprops=dict(
            marker='o',
            markersize=3,
            markerfacecolor='gray',
            markeredgecolor='gray',
            markeredgewidth=0.3,
            alpha=0.65
        ),
        boxprops=dict(edgecolor='black', linewidth=1.3),
        whiskerprops=dict(color='black', linewidth=1.3),
        capprops=dict(color='black', linewidth=1.3),
        medianprops=dict(color='black', linewidth=2)
    )

    for i, (name, df_input, feature_col, color) in enumerate(approaches):

        ax = axes[i]

        df = df_input.copy()

        df = df[
            (df["Split"].astype(str).str.strip().str.lower() == "test") &
            (df["Dataset"].astype(str).str.strip().str.upper() == "ICANS")
        ].copy()

        required = ["BDSPPatientID", "RawScore", feature_col]
        df = df[required].copy()

        df["RawScore"] = pd.to_numeric(df["RawScore"], errors="coerce")
        df[feature_col] = pd.to_numeric(df[feature_col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if df.empty:
            raise ValueError(f"No valid Test observations for {name}")

        y = df["RawScore"].to_numpy(dtype=float)
        x = df[feature_col].to_numpy(dtype=float)

        levels = sorted(df["RawScore"].unique())
        grouped = [x[y == level] for level in levels]
        counts = [len(vals) for vals in grouped]

        bp = ax.boxplot(grouped, **box_style)

        for box in bp["boxes"]:
            box.set_facecolor(color)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.5)

        ax.set_xticks(range(1, len(levels) + 1))

        ax.set_xticklabels(
            [f"{int(level)}\nn={n}" for level, n in zip(levels, counts)],
            fontsize=14
        )

        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=14)

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        if subplot_ylims.get(i) is not None:
            ax.set_ylim(subplot_ylims[i])

        # Exact original bootstrap approach
        rho, ci = bootstrap_spearman(y, x)

        ax.set_title(
            f"{name}\n" + r"$\mathbf{\rho}$" +
            f" = {rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
            fontsize=15,
            fontweight="bold",
            pad=12
        )

        ax.set_xlabel("ICANS score", fontsize=15)

        if i == 0:
            ax.set_ylabel("NESI", fontsize=16)

        stats_results.append({
            "Approach": name,
            "Spearman": rho,
            "CI_Lower": ci[0],
            "CI_Upper": ci[1],
            "N_EEG": len(df),
            "N_Patients": df["BDSPPatientID"].nunique()
        })

    fig.suptitle(
        "ICANS Ablation Study: Effect of EEG Segment Selection on NESI",
        fontsize=20,
        fontweight="bold",
        y=1.21
    )

    fig.text(
        0.5, 1.13,
        "A: Random selection of awake EEG segments  |  "
        "B: All-day EEG segments irrespective of vigilance state  |  "
        "C: Proposed selection of maximally awake EEG segments",
        ha="center",
        va="center",
        fontsize=14,
        color="black"
    )

    plt.tight_layout(rect=[0.025, 0.18, 1, 0.88], w_pad=2.5)

    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, subplot_height])

    for i, letter in enumerate(["A", "B", "C"]):
        pos = axes[i].get_position()

        fig.text(
            pos.x0 - 0.025,
            pos.y1 + 0.025,
            letter,
            fontsize=20,
            fontweight="bold"
        )

    fig.text(
        0.5, 0.055,
        "Boxes: median and IQR; whiskers: 1.5×IQR; gray points: outliers.\n"
        "n denotes the number of EEG observations at each ICANS grade. Higher NESI values indicate more severe neurological dysfunction.\n"
        "Spearman's ρ and 95% CIs were estimated using 1,000 observation-level bootstrap iterations.",
        ha="center",
        va="center",
        fontsize=15,
        color="black",
        linespacing=1.7
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()

    df_results = pd.DataFrame(stats_results)

    print("\nICANS SEGMENT-LEVEL SPEARMAN CORRELATIONS (95% CI)")
    display(df_results.round(4))

    return df_results


df_all_proposed_nesi = pd.read_csv(NESI_ICANS_csv)

df_icans_nesi_proposed = df_all_proposed_nesi[
    df_all_proposed_nesi["Dataset"] == "ICANS"
].copy()

df_icans_random_result = pd.read_csv(ICANS_random_awake_csv)
df_icans_allday_result = pd.read_csv(ICANS_all_day_cluster_result_csv)

subplot_ylims = {
    0: None,
    1: None,
    2: None
}

df_icans_ablation_results = plot_ICANS_segment_ablation(
    df_random=df_icans_random_result,
    df_allday=df_icans_allday_result,
    df_proposed=df_icans_nesi_proposed,
    save_path= ICANS_Result_dir / "ICANS_Segment_Selection_Ablation.png",
    n_boot=1000,
    seed=42,
    figsize=(19, 6.5),
    subplot_ylims=subplot_ylims,
    subplot_height=0.65
)


# In[24]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests
from IPython.display import display

def compare_ICANS_ablation_significance(df_random, df_allday, df_proposed,
                                        n_boot=1000, seed=42):

    def prepare(df, score_col):
        df = df[
            (df["Split"].astype(str).str.strip().str.lower() == "test") &
            (df["Dataset"].astype(str).str.strip().str.upper() == "ICANS")
        ][["BDSPPatientID", "RawScore", score_col]].copy()

        df["RawScore"] = pd.to_numeric(df["RawScore"], errors="coerce")
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df["BDSPPatientID"] = df["BDSPPatientID"].astype(str)

        return df.rename(columns={score_col: "Score"})

    def bootstrap_spearman(df):
        y_true = df["RawScore"].to_numpy(dtype=float)
        y_pred = df["Score"].to_numpy(dtype=float)

        n = len(y_true)
        corrs = np.zeros(n_boot)
        rng = np.random.default_rng(seed)

        for i in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            corrs[i], _ = spearmanr(y_true[idx], y_pred[idx])

        return corrs

    proposed = prepare(df_proposed, "NESI")

    alternatives = {
        "Random Awake": prepare(df_random, "NESI_RandomAwake"),
        "All-Day Segments": prepare(df_allday, "NESI_RandomAwake")
    }

    proposed_corrs = bootstrap_spearman(proposed)
    results = []

    for name, alternative in alternatives.items():

        alternative_corrs = bootstrap_spearman(alternative)

        valid = np.isfinite(proposed_corrs) & np.isfinite(alternative_corrs)
        global_corrs = proposed_corrs[valid]
        bespoke_corrs = alternative_corrs[valid]

        if len(global_corrs) == 0:
            raise ValueError(f"No valid bootstrap correlations for {name}")

        global_mean = np.mean(global_corrs)
        bespoke_mean = np.mean(bespoke_corrs)

        global_ci = np.percentile(global_corrs, [2.5, 97.5])
        bespoke_ci = np.percentile(bespoke_corrs, [2.5, 97.5])

        delta_corrs = global_corrs - bespoke_corrs
        delta_mean = np.mean(delta_corrs)
        delta_ci = np.percentile(delta_corrs, [2.5, 97.5])

        stat, p_value = wilcoxon(
            global_corrs,
            bespoke_corrs,
            alternative="greater"
        )

        results.append({
            "Comparison": f"Proposed vs {name}",
            "N_Proposed": len(proposed),
            "N_Alternative": len(alternative),
            "N_Common_Patients": len(
                set(proposed["BDSPPatientID"]) &
                set(alternative["BDSPPatientID"])
            ),
            "Proposed_rho": global_mean,
            "Proposed_CI_Lower": global_ci[0],
            "Proposed_CI_Upper": global_ci[1],
            "Alternative_rho": bespoke_mean,
            "Alternative_CI_Lower": bespoke_ci[0],
            "Alternative_CI_Upper": bespoke_ci[1],
            "Delta_rho": delta_mean,
            "Delta_CI_Lower": delta_ci[0],
            "Delta_CI_Upper": delta_ci[1],
            "Wilcoxon_stat": stat,
            "p_value": p_value
        })

    df_results = pd.DataFrame(results)

    df_results["p_Holm"] = multipletests(
        df_results["p_value"],
        method="holm"
    )[1]

    df_results["Significant"] = df_results["p_Holm"] < 0.05

    df_results["Winner"] = np.where(
        df_results["Significant"],
        "Proposed Maximum-Awake",
        "No significant improvement"
    )

    print("\nICANS ABLATION: BOOTSTRAP SPEARMAN + WILCOXON + HOLM")
    display(df_results.round(4))

    return df_results


df_icans_ablation_significance = compare_ICANS_ablation_significance(
    df_random=df_icans_random_result,
    df_allday=df_icans_allday_result,
    df_proposed=df_icans_nesi_proposed,
    n_boot=1000,
    seed=42
)