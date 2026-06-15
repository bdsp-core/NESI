#!/usr/bin/env python
# coding: utf-8

# In[53]:


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

# ---------------- Deep Learning (PyTorch) ----------------
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torchsummary import summary


# # **Plot helper function**

# In[70]:





# # **Load NESI and all four Bespoke model results**

# In[47]:
from pathlib import Path

if "__file__" in globals():
    current = Path(__file__).resolve()
else:
    current = Path.cwd()

NESI_ROOT = None

for parent in current.parents:
    if (parent / "NESI").exists():
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI folder not found")

Bespoke_Root = NESI_ROOT / "NESI" / "Bespoke_models" / "Results"


NESI_result_path = Bespoke_Root / "NESIscores_full_dataset_results.csv"
RASS_bespoke_result_path = Bespoke_Root / "RASSBespokescore_results.csv"
GCS_bespoke_result_path = Bespoke_Root / "GCSBespokescore_results.csv"
CAMS_bespoke_result_path = Bespoke_Root / "CAMSBespokescore_results.csv"
ICANS_bespoke_result_path = Bespoke_Root / "ICANSBespokescore_results.csv"

#------------------ NESI model's results ---------------------
df_NESI_result = pd.read_csv(NESI_result_path)
df_NESI_test = df_NESI_result[df_NESI_result["Split"]=="Test"]

df_RASS_from_NESI = df_NESI_test[(df_NESI_test['Dataset']=='RASS')].reset_index(drop=True)
df_GCS_from_NESI = df_NESI_test[(df_NESI_test['Dataset']=='GCS')].reset_index(drop=True)
df_CAMS_from_NESI = df_NESI_test[(df_NESI_test['Dataset']=='CAMS')].reset_index(drop=True)
df_ICANS_from_NESI = df_NESI_test[(df_NESI_test['Dataset']=='ICANS')].reset_index(drop=True)


# ---------------- BESPOKE model's results -------------------------
df_RASS_bespoke_result = pd.read_csv(RASS_bespoke_result_path)
df_RASS_bespoke_test = df_RASS_bespoke_result[df_RASS_bespoke_result["Split"]=="Test"].reset_index(drop=True)

df_GCS_bespoke_result = pd.read_csv(GCS_bespoke_result_path)
df_GCS_bespoke_test = df_GCS_bespoke_result[df_GCS_bespoke_result["Split"]=="Test"].reset_index(drop=True)

df_CAMS_bespoke_result = pd.read_csv(CAMS_bespoke_result_path)
df_CAMS_bespoke_test = df_CAMS_bespoke_result[df_CAMS_bespoke_result["Split"]=="Test"].reset_index(drop=True)

df_ICANS_bespoke_result = pd.read_csv(ICANS_bespoke_result_path)
df_ICANS_bespoke_test = df_ICANS_bespoke_result[df_ICANS_bespoke_result["Split"]=="Test"].reset_index(drop=True)


# # **NESI vs Bespoke**

# ## **ICANS results compare: NESI vs Bespoke**

# In[56]:


# ---------------- mask ----------------
Yraw_tst_ICANS = df_ICANS_bespoke_test['RawScore']
Yraw_transformed_tst_global_ICANS = df_ICANS_bespoke_test['TransformedScore']
tst_badness_ICANS = df_ICANS_bespoke_test['BespokeBadnessScore']
tst_NESI_ICANS = df_ICANS_from_NESI['NESI']

# ---------------- dataframe ----------------
df_icans = pd.DataFrame({
    "TrueRawICANS": Yraw_tst_ICANS,
    "TrueTransformedICANS": Yraw_transformed_tst_global_ICANS,
    "GlobalModelScore": tst_NESI_ICANS,
    "BespokeModelScore": tst_badness_ICANS
})

# ---------------- extract ----------------
y_true = df_icans["TrueTransformedICANS"].values
y_global = df_icans["GlobalModelScore"].values
y_bespoke = df_icans["BespokeModelScore"].values


# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)
bespoke_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y_global[idx])
    bespoke_corrs[i], _ = spearmanr(y_true[idx], y_bespoke[idx])

# ---------------- stats ----------------
global_mean_ICANS = np.mean(global_corrs)
bespoke_mean_ICANS = np.mean(bespoke_corrs)

global_ci_ICANS = np.percentile(global_corrs, [2.5, 97.5])
bespoke_ci_ICANS = np.percentile(bespoke_corrs, [2.5, 97.5])

# ---------------- WILCOXON TEST ----------------
stat, p_value_ICANS = wilcoxon(global_corrs, bespoke_corrs, alternative="greater")

# ---------------- decide winner ----------------
if p_value_ICANS < 0.05:
    winner = "GLOBAL MODEL is significantly better than BESPOKE MODEL"
else:
    winner = "No significant difference (or BESPOKE not worse)"

# ---------------- output ----------------
print("GLOBAL MODEL")
print(f"Mean Spearman: {global_mean_ICANS:.4f}")
print(f"95% CI: [{global_ci_ICANS[0]:.4f}, {global_ci_ICANS[1]:.4f}]")

print("\nBESPOKE MODEL")
print(f"Mean Spearman: {bespoke_mean_ICANS:.4f}")
print(f"95% CI: [{bespoke_ci_ICANS[0]:.4f}, {bespoke_ci_ICANS[1]:.4f}]")

print("\nSTATISTICAL TEST (Wilcoxon Rank Test, paired)")
print(f"p-value: {p_value_ICANS:.3e}")
print(f"Result: {winner}")


# ## **CAMS results compare: NESI vs Bespoke**

# In[57]:


# ---------------- mask ----------------
Yraw_tst_CAMS = df_CAMS_bespoke_test['RawScore']
Yraw_transformed_tst_global_CAMS = df_CAMS_bespoke_test['TransformedScore']
tst_badness_CAMS = df_CAMS_bespoke_test['BespokeBadnessScore']
tst_NESI_CAMS = df_CAMS_from_NESI['NESI']

# ---------------- dataframe ----------------
df_icans = pd.DataFrame({
    "TrueRawCAMS": Yraw_tst_CAMS,
    "TrueTransformedCAMS": Yraw_transformed_tst_global_CAMS,
    "GlobalModelScore": tst_NESI_CAMS,
    "BespokeModelScore": tst_badness_CAMS
})

# ---------------- extract ----------------
y_true = df_icans["TrueTransformedCAMS"].values
y_global = df_icans["GlobalModelScore"].values
y_bespoke = df_icans["BespokeModelScore"].values


# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)
bespoke_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y_global[idx])
    bespoke_corrs[i], _ = spearmanr(y_true[idx], y_bespoke[idx])

# ---------------- stats ----------------
global_mean_CAMS = np.mean(global_corrs)
bespoke_mean_CAMS = np.mean(bespoke_corrs)

global_ci_CAMS = np.percentile(global_corrs, [2.5, 97.5])
bespoke_ci_CAMS = np.percentile(bespoke_corrs, [2.5, 97.5])

# ---------------- WILCOXON TEST ----------------
stat, p_value_CAMS = wilcoxon(global_corrs, bespoke_corrs, alternative="greater")

# ---------------- decide winner ----------------
if p_value_CAMS < 0.05:
    winner = "GLOBAL MODEL is significantly better than BESPOKE MODEL"
else:
    winner = "No significant difference (or BESPOKE not worse)"

# ---------------- output ----------------
print("GLOBAL MODEL")
print(f"Mean Spearman: {global_mean_CAMS:.4f}")
print(f"95% CI: [{global_ci_CAMS[0]:.4f}, {global_ci_CAMS[1]:.4f}]")

print("\nBESPOKE MODEL")
print(f"Mean Spearman: {bespoke_mean_CAMS:.4f}")
print(f"95% CI: [{bespoke_ci_CAMS[0]:.4f}, {bespoke_ci_CAMS[1]:.4f}]")

print("\nSTATISTICAL TEST (Wilcoxon Rank Test, paired)")
print(f"p-value: {p_value_CAMS:.3e}")
print(f"Result: {winner}")


# ## **RASS results compare: NESI vs Bespoke**

# In[58]:


# ---------------- mask ----------------
Yraw_tst_RASS = df_RASS_bespoke_test['RawScore']
Yraw_transformed_tst_global_RASS = df_RASS_bespoke_test['TransformedScore']
tst_badness_RASS = df_RASS_bespoke_test['BespokeBadnessScore']
tst_NESI_RASS = df_RASS_from_NESI['NESI']

# ---------------- dataframe ----------------
df_icans = pd.DataFrame({
    "TrueRawRASS": Yraw_tst_RASS,
    "TrueTransformedRASS": Yraw_transformed_tst_global_RASS,
    "GlobalModelScore": tst_NESI_RASS,
    "BespokeModelScore": tst_badness_RASS
})

# ---------------- extract ----------------
y_true = df_icans["TrueTransformedRASS"].values
y_global = df_icans["GlobalModelScore"].values
y_bespoke = df_icans["BespokeModelScore"].values


# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)
bespoke_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y_global[idx])
    bespoke_corrs[i], _ = spearmanr(y_true[idx], y_bespoke[idx])

# ---------------- stats ----------------
global_mean_RASS = np.mean(global_corrs)
bespoke_mean_RASS = np.mean(bespoke_corrs)

global_ci_RASS = np.percentile(global_corrs, [2.5, 97.5])
bespoke_ci_RASS = np.percentile(bespoke_corrs, [2.5, 97.5])

# ---------------- WILCOXON TEST ----------------
stat, p_value_RASS = wilcoxon(global_corrs, bespoke_corrs, alternative="greater")

# ---------------- decide winner ----------------
if p_value_RASS < 0.05:
    winner = "GLOBAL MODEL is significantly better than BESPOKE MODEL"
else:
    winner = "No significant difference (or BESPOKE not worse)"

# ---------------- output ----------------
print("GLOBAL MODEL")
print(f"Mean Spearman: {global_mean_RASS:.4f}")
print(f"95% CI: [{global_ci_RASS[0]:.4f}, {global_ci_RASS[1]:.4f}]")

print("\nBESPOKE MODEL")
print(f"Mean Spearman: {bespoke_mean_RASS:.4f}")
print(f"95% CI: [{bespoke_ci_RASS[0]:.4f}, {bespoke_ci_RASS[1]:.4f}]")

print("\nSTATISTICAL TEST (Wilcoxon Rank Test, paired)")
print(f"p-value: {p_value_RASS:.3e}")
print(f"Result: {winner}")


# ## **GCS results compare: NESI vs Bespoke**

# ---------------- mask ----------------
Yraw_tst_GCS = df_GCS_bespoke_test['RawScore']
Yraw_transformed_tst_global_GCS = df_GCS_bespoke_test['TransformedScore']
tst_badness_GCS = df_GCS_bespoke_test['BespokeBadnessScore']
tst_NESI_GCS = df_GCS_from_NESI['NESI']

# ---------------- dataframe ----------------
df_icans = pd.DataFrame({
    "TrueRawGCS": Yraw_tst_GCS,
    "TrueTransformedGCS": Yraw_transformed_tst_global_GCS,
    "GlobalModelScore": tst_NESI_GCS,
    "BespokeModelScore": tst_badness_GCS
})

# ---------------- extract ----------------
y_true = df_icans["TrueTransformedGCS"].values
y_global = df_icans["GlobalModelScore"].values
y_bespoke = df_icans["BespokeModelScore"].values


# ---------------- bootstrap ----------------
n = len(y_true)
n_boot = 1000

global_corrs = np.zeros(n_boot)
bespoke_corrs = np.zeros(n_boot)

rng = np.random.default_rng(42)

for i in range(n_boot):
    idx = rng.choice(n, n, replace=True)

    global_corrs[i], _ = spearmanr(y_true[idx], y_global[idx])
    bespoke_corrs[i], _ = spearmanr(y_true[idx], y_bespoke[idx])

# ---------------- stats ----------------
global_mean_GCS = np.mean(global_corrs)
bespoke_mean_GCS = np.mean(bespoke_corrs)

global_ci_GCS = np.percentile(global_corrs, [2.5, 97.5])
bespoke_ci_GCS = np.percentile(bespoke_corrs, [2.5, 97.5])

# ---------------- WILCOXON TEST ----------------
stat, p_value_GCS = wilcoxon(global_corrs, bespoke_corrs, alternative="greater")

# ---------------- decide winner ----------------
if p_value_GCS < 0.05:
    winner = "GLOBAL MODEL is significantly better than BESPOKE MODEL"
else:
    winner = "No significant difference (or BESPOKE not worse)"

# ---------------- output ----------------
print("GLOBAL MODEL")
print(f"Mean Spearman: {global_mean_GCS:.4f}")
print(f"95% CI: [{global_ci_GCS[0]:.4f}, {global_ci_GCS[1]:.4f}]")

print("\nBESPOKE MODEL")
print(f"Mean Spearman: {bespoke_mean_GCS:.4f}")
print(f"95% CI: [{bespoke_ci_GCS[0]:.4f}, {bespoke_ci_GCS[1]:.4f}]")

print("\nSTATISTICAL TEST (Wilcoxon Rank Test, paired)")
print(f"p-value: {p_value_GCS:.3e}")
print(f"Result: {winner}")


# In[71]:
# # **Plots**

# In[59]:


# ------------- Universal Model ---------------
tst_spearman_conf_global_RASS = (
    f"{global_mean_RASS:.4f} "
    f"([{global_ci_RASS[0]:.4f}, {global_ci_RASS[1]:.4f}])"
)
tst_spearman_conf_global_GCS = (
    f"{global_mean_GCS:.4f} "
    f"([{global_ci_GCS[0]:.4f}, {global_ci_GCS[1]:.4f}])"
)
tst_spearman_conf_global_CAMS = (
    f"{global_mean_CAMS:.4f} "
    f"([{global_ci_CAMS[0]:.4f}, {global_ci_CAMS[1]:.4f}])"
)
tst_spearman_conf_global_ICANS = (
    f"{global_mean_ICANS:.4f} "
    f"([{global_ci_ICANS[0]:.4f}, {global_ci_ICANS[1]:.4f}])"
)

# -------------Bespoke models ---------------
tst_spearman_conf_bespoke_ICANS = (
    f"{bespoke_mean_ICANS:.4f} "
    f"([{bespoke_ci_ICANS[0]:.4f}, {bespoke_ci_ICANS[1]:.4f}])"
)
tst_spearman_conf_bespoke_CAMS = (
    f"{bespoke_mean_CAMS:.4f} "
    f"([{bespoke_ci_CAMS[0]:.4f}, {bespoke_ci_CAMS[1]:.4f}])"
)
tst_spearman_conf_bespoke_RASS = (
    f"{bespoke_mean_RASS:.4f} "
    f"([{bespoke_ci_RASS[0]:.4f}, {bespoke_ci_RASS[1]:.4f}])"
)
tst_spearman_conf_bespoke_GCS = (
    f"{bespoke_mean_GCS:.4f} "
    f"([{bespoke_ci_GCS[0]:.4f}, {bespoke_ci_GCS[1]:.4f}])"
)


# In[73]:


import numpy as np
import matplotlib.pyplot as plt


def plot_individual_vs_global_correct(
    # ---------------- individual models ----------------
    rass_bad, rass_y,
    gcs_bad, gcs_y,
    cams_bad, cams_y,
    icans_bad, icans_y,

    # ---------------- global model ----------------
    global_scores, global_yraw, global_dataset_names,

    # ---------------- SPEARMAN STRINGS ----------------
    spearman_bespoke_RASS,
    spearman_bespoke_GCS,
    spearman_bespoke_CAMS,
    spearman_bespoke_ICANS,

    spearman_global_RASS,
    spearman_global_GCS,
    spearman_global_CAMS,
    spearman_global_ICANS,

    title="Individual vs Global Model Comparison"
):
    plt.rcParams.update({'font.size': 12})

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.array(x)

    def group_data(badness, yraw, levels):
        grouped = []
        for l in levels:
            vals = badness[yraw == l]
            grouped.append(vals if len(vals) > 0 else np.array([np.nan]))
        return grouped

    # ---------------- convert ----------------
    rass_bad = to_numpy(rass_bad); rass_y = to_numpy(rass_y)
    gcs_bad  = to_numpy(gcs_bad);  gcs_y  = to_numpy(gcs_y)
    cams_bad = to_numpy(cams_bad); cams_y = to_numpy(cams_y)
    icans_bad = to_numpy(icans_bad); icans_y = to_numpy(icans_y)

    global_scores = to_numpy(global_scores)
    global_yraw = np.array(global_yraw)
    global_dataset_names = np.array(global_dataset_names)

    # ---------------- levels ----------------
    rass_levels = [-5, -4, -3, -2, -1, 0]
    gcs_levels  = list(range(3, 16))
    cams_levels = sorted(np.unique(cams_y))
    icans_levels = sorted(np.unique(icans_y))

    colors = ["#AEC6CF", "#FFB7B2", "#B2E2F2", "#CFCFC4", "#FDFD96", "#B39EB5", "#FFD1DC"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=False)

    box_style = dict(
        patch_artist=True,
        showfliers=True,
        flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'black', 'alpha': 0.3},
        widths=0.6,
        boxprops=dict(edgecolor='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        medianprops=dict(color='black', linewidth=2)
    )

    datasets_info = [
        ("RASS", rass_bad, rass_y, rass_levels, spearman_bespoke_RASS, spearman_global_RASS),
        ("GCS", gcs_bad, gcs_y, gcs_levels, spearman_bespoke_GCS, spearman_global_GCS),
        ("CAMS", cams_bad, cams_y, cams_levels, spearman_bespoke_CAMS, spearman_global_CAMS),
        ("ICANS", icans_bad, icans_y, icans_levels, spearman_bespoke_ICANS, spearman_global_ICANS),
    ]

    for i, (name, bad, y, levels, rho_ind, rho_global) in enumerate(datasets_info):

        # --- ROW 1: Individual Models ---
        ax_top = axes[0, i]
        grouped_ind = group_data(bad, y, levels)
        bp1 = ax_top.boxplot(grouped_ind, **box_style)

        for j, box in enumerate(bp1['boxes']):
            box.set_facecolor(colors[j % len(colors)])

        # --- ROW 2: Global Model ---
        ax_bot = axes[1, i]
        mask = global_dataset_names == name
        grouped_glob = group_data(global_scores[mask], global_yraw[mask], levels)
        bp2 = ax_bot.boxplot(grouped_glob, **box_style)

        for j, box in enumerate(bp2['boxes']):
            box.set_facecolor(colors[j % len(colors)])

        # --- formatting ---
        for ax in [ax_top, ax_bot]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

            ax.set_xticks(range(1, len(levels) + 1))
            ax.set_xticklabels(levels, rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        # --- TITLES WITH YOUR STRINGS ---
        ax_top.set_title(
            f"{name} \n ρ={rho_ind}",
            fontweight='bold'
        )

        ax_bot.set_title(
            f"{name} \n ρ={rho_global}",
            fontweight='bold'
        )

    axes[0, 0].set_ylabel("Bespoke Model's NESI", fontweight='bold')
    axes[1, 0].set_ylabel("Universal Model's  NESI", fontweight='bold')

    #fig.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# In[74]:


df_NESI_test = df_NESI_result[df_NESI_result["Split"]=="Test"]
tst_scores_global = df_NESI_result['NESI'].to_numpy()
Yraw_tst_global = df_NESI_result['RawScore'].to_numpy()
tst_dataset_names_global = df_NESI_result['Dataset'].to_numpy()

plot_individual_vs_global_correct(
    tst_badness_RASS, Yraw_tst_RASS,
    tst_badness_GCS,  Yraw_tst_GCS,
    tst_badness_CAMS, Yraw_tst_CAMS,
    tst_badness_ICANS, Yraw_tst_ICANS,

    tst_scores_global,
    Yraw_tst_global,
    tst_dataset_names_global,

    tst_spearman_conf_bespoke_RASS,
    tst_spearman_conf_bespoke_GCS,
    tst_spearman_conf_bespoke_CAMS,
    tst_spearman_conf_bespoke_ICANS,

    tst_spearman_conf_global_RASS,
    tst_spearman_conf_global_GCS,
    tst_spearman_conf_global_CAMS,
    tst_spearman_conf_global_ICANS,
    
    title="Individual vs Global Model Comparison (Same Hold out Testing set)"
)
