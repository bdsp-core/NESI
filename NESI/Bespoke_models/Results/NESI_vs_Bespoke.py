#!/usr/bin/env python
# coding: utf-8

# In[3]:


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



# # **Plot helper function**

# In[70]:





# # **Load NESI and all four Bespoke model results**

# In[47]:



Bespoke_Root = Path("/Users/arkaroy457/Desktop/NESI paper Death/Results")



NESI_result_path = Bespoke_Root / "NESIscores_full_dataset_result.csv"
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




# In[27]:


import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# STEP 1 — Reformat the rho/CI strings to match Figure 2's style:
#          "0.66 [0.65, 0.68]"  (2 decimals, brackets, no parentheses)
#          Replace your existing tst_spearman_conf_* assignments with these.
# =========================================================================

# ------------- Universal (Global) Model ---------------
tst_spearman_conf_global_RASS = f"{global_mean_RASS:.2f} [{global_ci_RASS[0]:.2f}, {global_ci_RASS[1]:.2f}]"
tst_spearman_conf_global_GCS  = f"{global_mean_GCS:.2f} [{global_ci_GCS[0]:.2f}, {global_ci_GCS[1]:.2f}]"
tst_spearman_conf_global_CAMS = f"{global_mean_CAMS:.2f} [{global_ci_CAMS[0]:.2f}, {global_ci_CAMS[1]:.2f}]"
tst_spearman_conf_global_ICANS = f"{global_mean_ICANS:.2f} [{global_ci_ICANS[0]:.2f}, {global_ci_ICANS[1]:.2f}]"

# ------------- Bespoke Models ---------------
tst_spearman_conf_bespoke_RASS = f"{bespoke_mean_RASS:.2f} [{bespoke_ci_RASS[0]:.2f}, {bespoke_ci_RASS[1]:.2f}]"
tst_spearman_conf_bespoke_GCS  = f"{bespoke_mean_GCS:.2f} [{bespoke_ci_GCS[0]:.2f}, {bespoke_ci_GCS[1]:.2f}]"
tst_spearman_conf_bespoke_CAMS = f"{bespoke_mean_CAMS:.2f} [{bespoke_ci_CAMS[0]:.2f}, {bespoke_ci_CAMS[1]:.2f}]"
tst_spearman_conf_bespoke_ICANS = f"{bespoke_mean_ICANS:.2f} [{bespoke_ci_ICANS[0]:.2f}, {bespoke_ci_ICANS[1]:.2f}]"


# =========================================================================
# STEP 2 — Figure-2-styled plotting function
#          (same signature as before, same data/stat inputs — only the
#          visual formatting is changed)
# =========================================================================

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

    title=None  # kept for signature compatibility; not shown (matches Fig 2, no suptitle)
):
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.array(x)

    def group_data(badness, yraw, levels):
        grouped, counts = [], []
        for l in levels:
            vals = badness[yraw == l]
            grouped.append(vals if len(vals) > 0 else np.array([np.nan]))
            counts.append(len(vals))
        return grouped, counts

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

    # ---------------- one color per dataset (matches Fig 2) ----------------
    dataset_colors = {
        "RASS":  "#7EB0D5",   # blue
        "GCS":   "#F4B860",   # orange/gold
        "CAMS":  "#3FB8A5",   # teal/green
        "ICANS": "#C77DB0",   # purple/mauve
    }

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=False)

    box_style = dict(
        patch_artist=True,
        showfliers=True,
        flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'gray',
                    'markeredgecolor': 'none', 'alpha': 0.3},
        widths=0.6,
        boxprops=dict(edgecolor='black', linewidth=1.3),
        whiskerprops=dict(color='black', linewidth=1.3),
        capprops=dict(color='black', linewidth=1.3),
        medianprops=dict(color='black', linewidth=2)
    )

    # per-dataset x-tick rotation: GCS is crowded (13 levels) so its
    # "score / n=count" labels collide at 0 degrees — rotate just that one.
    tick_rotation = {"RASS": 0, "GCS": 45, "CAMS": 0, "ICANS": 0}
    tick_fontsize  = {"RASS": 8, "GCS": 7, "CAMS": 8, "ICANS": 8}

    datasets_info = [
        ("RASS", rass_bad, rass_y, rass_levels, spearman_bespoke_RASS, spearman_global_RASS, "RASS score"),
        ("GCS", gcs_bad, gcs_y, gcs_levels, spearman_bespoke_GCS, spearman_global_GCS, "GCS score"),
        ("CAMS", cams_bad, cams_y, cams_levels, spearman_bespoke_CAMS, spearman_global_CAMS, "CAMS score"),
        ("ICANS", icans_bad, icans_y, icans_levels, spearman_bespoke_ICANS, spearman_global_ICANS, "ICANS score"),
    ]

    for i, (name, bad, y, levels, rho_ind, rho_global, xlabel) in enumerate(datasets_info):

        color = dataset_colors[name]

        # --- ROW A: Bespoke (individual) Models ---
        ax_top = axes[0, i]
        grouped_ind, n_ind = group_data(bad, y, levels)
        bp1 = ax_top.boxplot(grouped_ind, **box_style)
        for box in bp1['boxes']:
            box.set_facecolor(color)

        # --- ROW B: Universal (Global) Model ---
        ax_bot = axes[1, i]
        mask = global_dataset_names == name
        grouped_glob, n_glob = group_data(global_scores[mask], global_yraw[mask], levels)
        bp2 = ax_bot.boxplot(grouped_glob, **box_style)
        for box in bp2['boxes']:
            box.set_facecolor(color)

        # --- formatting shared by both rows ---
        for ax, n_counts in [(ax_top, n_ind), (ax_bot, n_glob)]:
            # L-shaped axes: keep only left + bottom spines, drop top + right
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_edgecolor('black')
            ax.spines['bottom'].set_edgecolor('black')
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            ax.set_xticks(range(1, len(levels) + 1))
            # two-line tick labels: score value on top, "n=###" below
            tick_labels = [f"{lvl}\nn={n}" for lvl, n in zip(levels, n_counts)]

            rot = tick_rotation[name]
            fsz = tick_fontsize[name]
            if rot == 0:
                ax.set_xticklabels(tick_labels, fontsize=10)
            else:
                # rotated labels need right-alignment so they tuck under their tick
                ax.set_xticklabels(tick_labels, fontsize=10, rotation=rot, ha='right',
                                    rotation_mode='anchor')

            ax.tick_params(axis='x', length=0)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        # --- titles: "NAME \n ρ = mean [lo, hi]" ---
        ax_top.set_title(f"{name}\n" + r"$\mathbf{\rho}$ = " + f"{rho_ind}", fontweight='bold', fontsize=15)
        ax_bot.set_title(f"{name}\n" + r"$\mathbf{\rho}$ = " + f"{rho_global}", fontweight='bold', fontsize=15)

        # --- x-axis label only on the bottom row ---
        ax_bot.set_xlabel(xlabel, fontweight='normal', fontsize=15)

    # --- y-axis labels only on the leftmost column ---
    axes[0, 0].set_ylabel("Bespoke model's NESI", fontweight='normal', fontsize=14)
    axes[1, 0].set_ylabel("Universal model's NESI", fontweight='normal', fontsize=14)

    # --- panel letters A / B, top-left of each row ---
    fig.text(0.005, 0.96, "A", fontsize=20, fontweight='bold')
    fig.text(0.005, 0.475, "B", fontsize=20, fontweight='bold')

    # --- caption at bottom, matching Fig 2 ---
    fig.text(0.5, 0.0, "Boxes: median & IQR; whiskers: 1.5×IQR; points: outliers.",
              ha='center', fontsize=14, style='italic')

    plt.tight_layout(rect=[0.015, 0.02, 1, 0.97], h_pad=3.0, w_pad=1.5)
    plt.show()
    fig.savefig("Figure2.pdf", dpi=600, bbox_inches="tight")

# =========================================================================
# STEP 3 — Call exactly as before (unchanged)
# =========================================================================

df_NESI_test = df_NESI_result[df_NESI_result["Split"] == "Test"]
tst_scores_global = df_NESI_test['NESI'].to_numpy()
Yraw_tst_global = df_NESI_test['RawScore'].to_numpy()
tst_dataset_names_global = df_NESI_test['Dataset'].to_numpy()

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
)

