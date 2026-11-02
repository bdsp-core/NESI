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


# In[2]:
current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")

NESI_result_path = NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "Final_NESIscores_full_dataset_result.csv"


df_rass_metadata =pd.read_csv(NESI_ROOT / "RevisionRelatedMaterials" /"Leave_one_Cohort_Out_Model" / "RASS_Holdout_Result.csv")
df_gcs_metadata =pd.read_csv(NESI_ROOT / "RevisionRelatedMaterials" / "Leave_one_Cohort_Out_Model" / "GCS_Holdout_Result.csv")
df_cams_metadata =pd.read_csv(NESI_ROOT / "RevisionRelatedMaterials" / "Leave_one_Cohort_Out_Model" / "CAMS_Holdout_Result.csv")
df_icans_metadata =pd.read_csv(NESI_ROOT / "RevisionRelatedMaterials" / "Leave_one_Cohort_Out_Model" / "ICANS_Holdout_Result.csv")


# In[3]:


for name, df in zip(
    ["RASS", "GCS", "CAM-S", "ICANS"],
    [df_rass_metadata, df_gcs_metadata, df_cams_metadata, df_icans_metadata]
):
    print(f"{name}: {df.shape}")
    print(df.columns.tolist(), "\n")


# # **Full held-out cohort**

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def plot_holdout_spearman(df_rass_metadata, df_gcs_metadata, df_cams_metadata, df_icans_metadata,
                          fig_path=None, figsize=(22, 5.5), dpi=400, n_boot=1000, seed=42):

    datasets = {
        "RASS": (df_rass_metadata, "NESI_RASSHoldout", "#7EB0D5"),
        "GCS": (df_gcs_metadata, "NESI_GCSHoldout", "#F4B860"),
        "CAM-S": (df_cams_metadata, "NESI_CAMSHoldout", "#3FB8A5"),
        "ICANS": (df_icans_metadata, "NESI_ICANSHoldout", "#C77DB0")
    }

    fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=dpi)
    results = []

    for ax, (name, (df, nesi_col, color)) in zip(axes, datasets.items()):

        data = df[["RawScore", "TransformedScore", nesi_col]].copy()
        data = data.dropna()
        data = data[np.isfinite(data["TransformedScore"]) & np.isfinite(data[nesi_col])]

        y_true = data["TransformedScore"].to_numpy(dtype=float)
        y_pred = data[nesi_col].to_numpy(dtype=float)

        n = len(y_true)
        corrs = np.zeros(n_boot)
        rng = np.random.default_rng(seed)

        for i in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            corrs[i], _ = spearmanr(y_true[idx], y_pred[idx])

        rho = np.mean(corrs)
        ci = np.percentile(corrs, [2.5, 97.5])

        results.append({
            "Dataset": name,
            "N": n,
            "Spearman_rho": rho,
            "CI_Lower": ci[0],
            "CI_Upper": ci[1]
        })

        scores = sorted(data["RawScore"].unique())
        box_data = [data.loc[data["RawScore"] == s, nesi_col].to_numpy() for s in scores]
        counts = [len(v) for v in box_data]

        bp = ax.boxplot(
            box_data,
            positions=np.arange(len(scores)),
            widths=0.60,
            patch_artist=True,
            whis=1.5,
            showfliers=True,
            boxprops=dict(edgecolor="black", linewidth=1),
            medianprops=dict(color="black", linewidth=1.3),
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1),
            flierprops=dict(marker="o", markersize=1.8, markerfacecolor="gray",
                            markeredgecolor="none", alpha=0.35)
        )

        for box in bp["boxes"]:
            box.set_facecolor(color)

        ax.set_xticks(np.arange(len(scores)))
        ax.set_xticklabels(
            [f"{s:g}\nn={count}" for s, count in zip(scores, counts)],
            fontsize=9
        )

        if name == "GCS":
            ax.tick_params(axis="x", labelrotation=45)

        ax.set_title(
            f"{name}\nρ = {rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
            fontsize=13, fontweight="bold", pad=12
        )

        ax.set_xlabel(f"{name} score", fontsize=12)
        ax.set_ylabel("NESI", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        print(f"{name}: N={n}, Spearman ρ={rho:.4f}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    fig.suptitle(
        "Leave-One-Cohort-Out Validation",
        fontsize=17, fontweight="bold", y=1.03
    )

    fig.text(
        0.5, -0.12,
        "Boxes: median and IQR; whiskers: 1.5×IQR; points: outliers. "
        "n denotes EEG observations at each clinical score.\n"
        "Spearman's ρ and 95% confidence intervals estimated using 1,000 bootstrap iterations.",
        ha="center", va="center", fontsize=11, style="italic"
    )

    plt.tight_layout(w_pad=2.5)

    if fig_path is not None:
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {fig_path}")

    plt.show()

    df_holdout_spearman_results = pd.DataFrame(results)
    print(df_holdout_spearman_results)

    return df_holdout_spearman_results


df_holdout_spearman_results = plot_holdout_spearman(
    df_rass_metadata,
    df_gcs_metadata,
    df_cams_metadata,
    df_icans_metadata,
    fig_path=None
)


# # **Matched test subset**

# In[10]:

df_proposed_nesi_result = pd.read_csv(NESI_result_path)


merge_cols = ['BDSPPatientID', 'Dataset', 'MorgothOutputFilename']

df_proposed_merge = df_proposed_nesi_result[
    merge_cols + ['NESI', 'Split']
].rename(columns={
    'NESI': 'NESI_proposed',
    'Split': 'Split_Universal_Model'
})

assert not df_proposed_merge.duplicated(merge_cols).any(), "Duplicate matching keys found!"

dfs = {
    'RASS': df_rass_metadata,
    'GCS': df_gcs_metadata,
    'CAM-S': df_cams_metadata,
    'ICANS': df_icans_metadata
}

for name, df in dfs.items():
    dfs[name] = df.merge(df_proposed_merge, on=merge_cols, how='left', validate='many_to_one')
    print(f"{name}: Total={len(dfs[name])}, Matched={dfs[name]['NESI_proposed'].notna().sum()}, "
          f"Unmatched={dfs[name]['NESI_proposed'].isna().sum()}")

df_rass_metadata = dfs['RASS']
df_gcs_metadata = dfs['GCS']
df_cams_metadata = dfs['CAM-S']
df_icans_metadata = dfs['ICANS']


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

df_rass_test = df_rass_metadata.loc[df_rass_metadata["Split_Universal_Model"].astype(str).str.strip().str.lower().eq("test")].copy()
df_gcs_test = df_gcs_metadata.loc[df_gcs_metadata["Split_Universal_Model"].astype(str).str.strip().str.lower().eq("test")].copy()
df_cams_test = df_cams_metadata.loc[df_cams_metadata["Split_Universal_Model"].astype(str).str.strip().str.lower().eq("test")].copy()
df_icans_test = df_icans_metadata.loc[df_icans_metadata["Split_Universal_Model"].astype(str).str.strip().str.lower().eq("test")].copy()

def plot_holdout_vs_proposed(df_rass_test, df_gcs_test, df_cams_test, df_icans_test,
                             fig_path=None, figsize=(22, 11), dpi=400, n_boot=1000, seed=42):

    datasets = {
        "RASS": (df_rass_test, "NESI_RASSHoldout", "#7EB0D5"),
        "GCS": (df_gcs_test, "NESI_GCSHoldout", "#F4B860"),
        "CAM-S": (df_cams_test, "NESI_CAMSHoldout", "#3FB8A5"),
        "ICANS": (df_icans_test, "NESI_ICANSHoldout", "#C77DB0")
    }

    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=dpi)
    results, p_values = [], []

    for col, (name, (df, holdout_col, color)) in enumerate(datasets.items()):

        data = df[["BDSPPatientID", "RawScore", "TransformedScore", holdout_col, "NESI_proposed"]].copy()

        for c in ["RawScore", "TransformedScore", holdout_col, "NESI_proposed"]:
            data[c] = pd.to_numeric(data[c], errors="coerce")

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=["RawScore", "TransformedScore", holdout_col, "NESI_proposed"])

        y_true = data["TransformedScore"].values
        y_holdout = data[holdout_col].values
        y_proposed = data["NESI_proposed"].values

        n = len(y_true)
        holdout_corrs = np.zeros(n_boot)
        proposed_corrs = np.zeros(n_boot)

        rng = np.random.default_rng(seed)

        for i in range(n_boot):
            idx = rng.choice(n, n, replace=True)

            holdout_corrs[i], _ = spearmanr(y_true[idx], y_holdout[idx])
            proposed_corrs[i], _ = spearmanr(y_true[idx], y_proposed[idx])

        holdout_mean = np.mean(holdout_corrs)
        proposed_mean = np.mean(proposed_corrs)

        holdout_ci = np.percentile(holdout_corrs, [2.5, 97.5])
        proposed_ci = np.percentile(proposed_corrs, [2.5, 97.5])

        delta_corrs = proposed_corrs - holdout_corrs
        delta_mean = np.mean(delta_corrs)
        delta_ci = np.percentile(delta_corrs, [2.5, 97.5])

        if np.allclose(delta_corrs, 0):
            p_value = 1.0
        else:
            stat, p_value = wilcoxon(proposed_corrs, holdout_corrs, alternative="greater")

        p_values.append(p_value)

        results.append({
            "Dataset": name,
            "N_EEG": n,
            "N_Patients": data["BDSPPatientID"].nunique(),
            "Holdout_rho": holdout_mean,
            "Holdout_CI_Lower": holdout_ci[0],
            "Holdout_CI_Upper": holdout_ci[1],
            "Proposed_rho": proposed_mean,
            "Proposed_CI_Lower": proposed_ci[0],
            "Proposed_CI_Upper": proposed_ci[1],
            "Delta_rho": delta_mean,
            "Delta_CI_Lower": delta_ci[0],
            "Delta_CI_Upper": delta_ci[1],
            "Wilcoxon_p": p_value
        })

        scores = sorted(data["RawScore"].unique())
        positions = np.arange(len(scores))

        for row, (score_col, rho, ci) in enumerate([
            (holdout_col, holdout_mean, holdout_ci),
            ("NESI_proposed", proposed_mean, proposed_ci)
        ]):

            ax = axes[row, col]

            box_data = [
                data.loc[data["RawScore"] == s, score_col].values
                for s in scores
            ]

            counts = [len(v) for v in box_data]

            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.60,
                patch_artist=True,
                whis=1.5,
                showfliers=True,
                boxprops=dict(edgecolor="black", linewidth=1),
                medianprops=dict(color="black", linewidth=1.3),
                whiskerprops=dict(color="black", linewidth=1),
                capprops=dict(color="black", linewidth=1),
                flierprops=dict(
                    marker="o",
                    markersize=1.8,
                    markerfacecolor="gray",
                    markeredgecolor="none",
                    alpha=0.35
                )
            )

            for box in bp["boxes"]:
                box.set_facecolor(color)

            ax.set_xticks(positions)
            ax.set_xticklabels(
                [f"{s:g}\nn={count}" for s, count in zip(scores, counts)],
                fontsize=9
            )

            if name == "GCS":
                ax.tick_params(axis="x", labelrotation=45)

            panel_letter = chr(65 + row * 4 + col)
            model_label = "Leave-one-cohort-out" if row == 0 else "Proposed NESI"

            ax.set_title(
                f"{panel_letter}. {name} ({model_label})\n"
                f"ρ = {rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
                fontsize=12,
                fontweight="bold",
                pad=12
            )

            ax.set_xlabel(f"{name} score", fontsize=11)
            ax.set_ylabel("NESI", fontsize=11)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        print(
            f"{name}: N={n}, Patients={data['BDSPPatientID'].nunique()}\n"
            f"  Holdout  ρ={holdout_mean:.4f}, 95% CI=[{holdout_ci[0]:.4f}, {holdout_ci[1]:.4f}]\n"
            f"  Proposed ρ={proposed_mean:.4f}, 95% CI=[{proposed_ci[0]:.4f}, {proposed_ci[1]:.4f}]\n"
            f"  Δρ={delta_mean:.4f}, 95% CI=[{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]\n"
            f"  Wilcoxon p={p_value:.6g}\n"
        )

    reject, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")

    df_results = pd.DataFrame(results)
    df_results["p_Holm"] = p_holm
    df_results["Significant_Holm"] = reject

    fig.suptitle(
        "Leave-One-Cohort-Out vs Proposed NESI: Matched Test Cohorts",
        fontsize=17,
        fontweight="bold",
        y=0.995
    )

    fig.text(
        0.5, 0.012,
        "Boxes: median and IQR; whiskers: 1.5×IQR; points: outliers; n: EEG observations per score.\n"
        "Spearman's ρ and 95% CI: 1,000 paired bootstrap iterations (seed=42).",
        ha="center",
        va="bottom",
        fontsize=10,
        style="italic"
    )

    fig.tight_layout(rect=[0, 0.065, 1, 0.95], w_pad=2.5, h_pad=3.5)

    if fig_path is not None:
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {fig_path}")

    plt.show()
    print(df_results)

    return df_results


df_holdout_vs_proposed_results = plot_holdout_vs_proposed(
    df_rass_test,
    df_gcs_test,
    df_cams_test,
    df_icans_test,
    fig_path=None
)

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

def plot_holdout_vs_proposed(
    df_rass_metadata, df_gcs_metadata, df_cams_metadata, df_icans_metadata,
    fig_path=None, figsize=(23, 11), dpi=400, n_boot=1000, seed=42
):
    datasets = {
        "RASS": (df_rass_metadata, "NESI_RASSHoldout", "#7EB0D5"),
        "GCS": (df_gcs_metadata, "NESI_GCSHoldout", "#F4B860"),
        "CAM-S": (df_cams_metadata, "NESI_CAMSHoldout", "#3FB8A5"),
        "ICANS": (df_icans_metadata, "NESI_ICANSHoldout", "#C77DB0")
    }

    fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=dpi)
    results, p_values = [], []

    for col, (name, (df, holdout_col, color)) in enumerate(datasets.items()):

        data = df.copy()
        for c in ["RawScore", "TransformedScore", holdout_col, "NESI_proposed"]:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan)

        # Row 1: Full held-out cohort
        full = data.dropna(subset=["RawScore", "TransformedScore", holdout_col]).copy()

        # Row 2: Matched universal-model Test subset
        test = data.loc[
            data["Split_Universal_Model"].astype(str).str.strip().str.lower().eq("test")
        ].dropna(subset=["RawScore", "TransformedScore", holdout_col, "NESI_proposed"]).copy()

        if len(full) < 3 or len(test) < 3:
            raise ValueError(f"{name}: insufficient observations in full or matched Test set.")

        # Full held-out cohort bootstrap
        y_full = full["TransformedScore"].to_numpy()
        pred_full = full[holdout_col].to_numpy()

        n_full = len(full)
        full_corrs = np.zeros(n_boot)
        rng_full = np.random.default_rng(seed)

        for i in range(n_boot):
            idx = rng_full.choice(n_full, n_full, replace=True)
            full_corrs[i], _ = spearmanr(y_full[idx], pred_full[idx])

        full_corrs = full_corrs[np.isfinite(full_corrs)]
        if len(full_corrs) == 0:
            raise ValueError(f"{name}: no valid full-cohort bootstrap correlations.")

        rho_full = np.mean(full_corrs)
        ci_full = np.percentile(full_corrs, [2.5, 97.5])

        # Matched Test subset: paired bootstrap
        y_true = test["TransformedScore"].to_numpy()
        y_holdout = test[holdout_col].to_numpy()
        y_proposed = test["NESI_proposed"].to_numpy()

        n = len(test)
        holdout_corrs = np.zeros(n_boot)
        proposed_corrs = np.zeros(n_boot)
        rng = np.random.default_rng(seed)

        for i in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            holdout_corrs[i], _ = spearmanr(y_true[idx], y_holdout[idx])
            proposed_corrs[i], _ = spearmanr(y_true[idx], y_proposed[idx])

        valid = np.isfinite(holdout_corrs) & np.isfinite(proposed_corrs)
        holdout_corrs = holdout_corrs[valid]
        proposed_corrs = proposed_corrs[valid]

        if len(holdout_corrs) == 0:
            raise ValueError(f"{name}: no valid matched bootstrap correlations.")

        rho_holdout = np.mean(holdout_corrs)
        rho_proposed = np.mean(proposed_corrs)

        ci_holdout = np.percentile(holdout_corrs, [2.5, 97.5])
        ci_proposed = np.percentile(proposed_corrs, [2.5, 97.5])

        delta_corrs = proposed_corrs - holdout_corrs
        delta_mean = np.mean(delta_corrs)
        delta_ci = np.percentile(delta_corrs, [2.5, 97.5])

        if np.allclose(delta_corrs, 0):
            p_value = 1.0
        else:
            _, p_value = wilcoxon(proposed_corrs, holdout_corrs, alternative="greater")

        p_values.append(p_value)

        results.append({
            "Dataset": name,
            "Full_Holdout_N_EEG": len(full),
            "Full_Holdout_N_Patients": full["BDSPPatientID"].nunique(),
            "Full_Holdout_rho": rho_full,
            "Full_Holdout_CI_Lower": ci_full[0],
            "Full_Holdout_CI_Upper": ci_full[1],
            "Matched_Test_N_EEG": len(test),
            "Matched_Test_N_Patients": test["BDSPPatientID"].nunique(),
            "Matched_Holdout_rho": rho_holdout,
            "Matched_Holdout_CI_Lower": ci_holdout[0],
            "Matched_Holdout_CI_Upper": ci_holdout[1],
            "Proposed_rho": rho_proposed,
            "Proposed_CI_Lower": ci_proposed[0],
            "Proposed_CI_Upper": ci_proposed[1],
            "Delta_rho": delta_mean,
            "Delta_CI_Lower": delta_ci[0],
            "Delta_CI_Upper": delta_ci[1],
            "Wilcoxon_p": p_value
        })

        panels = [
            (full, holdout_col, rho_full, ci_full),
            (test, holdout_col, rho_holdout, ci_holdout)
        ]

        for row, (panel_data, score_col, rho, ci) in enumerate(panels):

            ax = axes[row, col]
            scores = sorted(panel_data["RawScore"].unique())
            positions = np.arange(len(scores))

            box_data = [
                panel_data.loc[panel_data["RawScore"] == s, score_col].to_numpy()
                for s in scores
            ]
            counts = [len(v) for v in box_data]

            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.60,
                patch_artist=True,
                whis=1.5,
                showfliers=True,
                boxprops=dict(edgecolor="black", linewidth=1),
                medianprops=dict(color="black", linewidth=1.3),
                whiskerprops=dict(color="black", linewidth=1),
                capprops=dict(color="black", linewidth=1),
                flierprops=dict(
                    marker="o", markersize=1.8,
                    markerfacecolor="gray",
                    markeredgecolor="none", alpha=0.35
                )
            )

            for box in bp["boxes"]:
                box.set_facecolor(color)

            ax.set_xticks(positions)
            ax.set_xticklabels(
                [f"{s:g}\nn={count}" for s, count in zip(scores, counts)],
                fontsize=9
            )

            if name == "GCS":
                ax.tick_params(axis="x", labelrotation=45)

            panel_letter = chr(65 + row * 4 + col)

            ax.set_title(
                f"{panel_letter}. {name}\n"
                f"ρ = {rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
                fontsize=12, fontweight="bold", pad=12
            )

            ax.set_xlabel(f"{name} score", fontsize=14)
            ax.set_ylabel("NESI", fontsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_axisbelow(True)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        print(
            f"{name}\n"
            f"  Full holdout: N={len(full)}, ρ={rho_full:.4f}, "
            f"CI=[{ci_full[0]:.4f}, {ci_full[1]:.4f}]\n"
            f"  Matched holdout: N={len(test)}, ρ={rho_holdout:.4f}, "
            f"CI=[{ci_holdout[0]:.4f}, {ci_holdout[1]:.4f}]\n"
            f"  Proposed: N={len(test)}, ρ={rho_proposed:.4f}, "
            f"CI=[{ci_proposed[0]:.4f}, {ci_proposed[1]:.4f}]\n"
            f"  Δρ={delta_mean:.4f}, CI=[{delta_ci[0]:.4f}, {delta_ci[1]:.4f}], "
            f"Wilcoxon p={p_value:.6g}\n"
        )

    reject, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")

    df_results = pd.DataFrame(results)
    df_results["p_Holm"] = p_holm
    df_results["Significant_Holm"] = reject

    fig.suptitle(
        "Leave-One-Cohort-Out NESI Generalization:\n"
        "Full Held-Out Cohorts vs Matched Universal-Model Test Subsets",
        fontsize=17, fontweight="bold", y=1.05
    )

    fig.text(
        0.5, 0.955,
        "Row 1 (A–D): Leave-one-cohort-out based NESI estimation scheme evaluated on the full held-out cohort.\n"
        "Row 2 (E–H): Leave-one-cohort-out based NESI estimation scheme evaluated on the matched Test subset of the proposed universal model (trained using all scales together).",
        ha="center", va="bottom", fontsize=14
    )

    fig.text(
        0.5, 0.015,
        "Boxes: median and IQR; whiskers: 1.5×IQR; gray points: outliers; n: EEG observations per score.\n"
        "Spearman's ρ and 95% CI: 1,000 bootstrap iterations (seed=42).",
        ha="center", va="bottom", fontsize=16, style="italic"
    )

    fig.tight_layout(rect=[0, 0.095, 1, 0.93], w_pad=2.5, h_pad=3.5)

    if fig_path is not None:
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {fig_path}")

    plt.show()
    print(df_results)

    return df_results


df_holdout_vs_proposed_results = plot_holdout_vs_proposed(
    df_rass_metadata,
    df_gcs_metadata,
    df_cams_metadata,
    df_icans_metadata,
    fig_path= NESI_ROOT / "RevisionRelatedMaterials" / "Leave_one_Cohort_Out_Model" / "LeaveoneCohortOut_Result.png"
)
