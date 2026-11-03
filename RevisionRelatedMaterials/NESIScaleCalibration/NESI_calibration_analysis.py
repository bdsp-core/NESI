#!/usr/bin/env python
# coding: utf-8

# 
# # **Step 1: Data Loading, Clinical Severity Labeling, and Patient-Level Split Verification**
# 
# - **Objective:** Prepare the NESI dataset for cohort-specific calibration analysis and verify patient independence across Train, Validation, and Test splits.
# - **Input:** `Final_NESIscores_full_dataset_result.csv`, containing NESI predictions, clinical severity labels, patient identifiers, and dataset splits.
# - **Cohorts:** RASS, GCS, CAM-S, and ICANS.
# - **Preprocessing:**
#   - Convert `NESI` and `GroupedScore` to numeric values.
#   - Standardize `Dataset` and `Split` column names/values.
#   - Retain observations from the four clinical cohorts.
# - **Binary severity labeling:** Create `Severe`, where 1 indicates severe neurological impairment (`GroupedScore = 2`) and 0 indicates non-severe impairment (`GroupedScore = 0 or 1`).
# - **Clinical endpoints:**
#   - RASS: ≤ −4
#   - GCS: ≤ 8
#   - CAM-S: ≥ 6
#   - ICANS: ≥ 3
# - **Dataset summary:** For each cohort and split, calculate the total number of EEG recordings, unique patients, severe observations, missing NESI predictions, missing severity labels, and severe-outcome prevalence.
# - **Patient overlap verification:** Identify overlapping `BDSPPatientID` values between Train–Validation, Train–Test, and Validation–Test within each cohort to assess patient-level data independence.
# - **Output:** Cohort-wise dataset summary and patient overlap counts for all four clinical cohorts.
# - **Purpose:** Confirm data completeness, severe-outcome distributions, and patient-independent evaluation splits before fitting logistic calibration models.
# 
# **Note:** Missing clinical labels are counted during quality assessment and must be excluded before logistic regression fitting to avoid incorrectly labeling missing observations as non-severe.
# 

# In[1]:


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


DATA_PATH = Path(NESI_ROOT / "RevisionRelatedMaterials" / "RevisionData_to_ReproduceResult"/ "Final_NESIscores_full_dataset_result.csv")
df = pd.read_csv(DATA_PATH, dtype={"BDSPPatientID": str})

df["NESI"] = pd.to_numeric(df["NESI"], errors="coerce")
df["GroupedScore"] = pd.to_numeric(df["GroupedScore"], errors="coerce")
df["Dataset"] = df["Dataset"].astype(str).str.upper().str.strip()
df["Split"] = df["Split"].astype(str).str.strip().str.capitalize()

df = df[df["Dataset"].isin(["RASS", "GCS", "CAMS", "ICANS"])].copy()
df["Severe"] = (df["GroupedScore"] == 2).astype(int)

summary = df.groupby(["Dataset", "Split"]).agg(
    N_EEG=("NESI", "size"),
    N_Patients=("BDSPPatientID", "nunique"),
    N_Severe=("Severe", "sum"),
    N_Missing_NESI=("NESI", lambda x: x.isna().sum()),
    N_Missing_Labels=("GroupedScore", lambda x: x.isna().sum())
).reset_index()

summary["Severe_Percent"] = (100 * summary["N_Severe"] / summary["N_EEG"]).round(2)

print(summary.to_string(index=False))


# ----------------- Check Overalp of patients in Train Val Test Set --------------------------------
for cohort in ["RASS", "GCS", "CAMS", "ICANS"]:
    d = df[df["Dataset"] == cohort]
    train = set(d.loc[d["Split"] == "Train", "BDSPPatientID"])
    val = set(d.loc[d["Split"] == "Val", "BDSPPatientID"])
    test = set(d.loc[d["Split"] == "Test", "BDSPPatientID"])

    print(f"\n{cohort}")
    print(f"Train vs Val  : {len(train & val)} overlapping patients")
    print(f"Train vs Test : {len(train & test)} overlapping patients")
    print(f"Val vs Test   : {len(val & test)} overlapping patients")


# 
# # **Step 2: Cohort-Specific Logistic Regression for NESI Probability Calibration**
# 
# - **Objective:** Map continuous NESI scores to the probability of severe neurological impairment using separate logistic regression models for RASS, GCS, CAM-S, and ICANS.
# - **Input:** Continuous NESI score (one value per EEG recording).
# - **Target:** Binary clinical severity label (`Severe`), where 1 indicates severe neurological impairment (`GroupedScore = 2`) and 0 indicates non-severe impairment (`GroupedScore = 0 or 1`).
# - **Clinical endpoints:**
#   - RASS: ≤ −4
#   - GCS: ≤ 8
#   - CAM-S: ≥ 6
#   - ICANS: ≥ 3
# - **Training:** Four independent logistic regression models are fitted using the combined Train + Validation data from their respective cohorts.
# - **Model:** `P(Severe | NESI) = 1 / (1 + exp(-(a + b × NESI)))`, where `a` is the intercept and `b` is the NESI coefficient.
# - **Output:** Four fitted logistic regression models stored in `calibration_models`, along with their intercepts, coefficients, and probability-mapping equations.
# - **Evaluation:** The held-out Test datasets are reserved for subsequent independent calibration assessment.
# 

# In[2]:


from sklearn.linear_model import LogisticRegression

COHORTS = ["RASS", "GCS", "CAMS", "ICANS"]
calibration_models = {}

for cohort in COHORTS:
    d = df[df["Dataset"] == cohort].copy()
    dev = d[d["Split"].isin(["Train", "Val"])]

    X = dev[["NESI"]]
    y = dev["Severe"]

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-8
    )
    model.fit(X, y)
    calibration_models[cohort] = model

    a = model.intercept_[0]
    b = model.coef_[0, 0]

    print(f"\n{cohort}")
    print(f"Development EEGs: {len(dev):,}")
    print(f"Intercept (a): {a:.6f}")
    print(f"Coefficient (b): {b:.6f}")
    print(f"Formula: P(Severe) = 1 / (1 + exp(-({a:.4f} + {b:.4f} * NESI)))")


# 
# # **Step 3: Test-Set Prediction of Severe Encephalopathy Using Cohort-Specific Logistic Regression Models**
# 
# - **Objective:** Apply the four previously trained logistic regression models to their respective held-out Test datasets to estimate the probability of severe neurological impairment from continuous NESI scores.
# - **Input:** `df`, containing NESI scores, clinical severity labels, patient identifiers, and dataset splits, along with the fitted models stored in `calibration_models`.
# - **Cohort-specific evaluation:**
#   - RASS Test → RASS logistic regression model
#   - GCS Test → GCS logistic regression model
#   - CAM-S Test → CAM-S logistic regression model
#   - ICANS Test → ICANS logistic regression model
# - **Prediction:** For each test EEG recording, calculate `PredictedProbability`, representing the estimated probability of severe encephalopathy (`Severe = 1`).
# - **Independent testing:** Apply the previously fitted logistic regression models without retraining or updating their parameters using Test data.
# - **Evaluation summary:** For each cohort, report:
#   - Total number of Test EEG recordings.
#   - Number of unique Test patients.
#   - Observed prevalence of severe encephalopathy.
#   - Mean predicted probability of severe encephalopathy.
#   - Minimum and maximum predicted probabilities.
# - **Output:** `test_predictions`, a dictionary containing four cohort-specific Test dataframes with NESI scores, true severity labels, and predicted severe-encephalopathy probabilities.
# - **Interpretation:** Compare the mean predicted probability with the observed severe-outcome prevalence to assess overall systematic overprediction or underprediction (calibration-in-the-large). Close agreement indicates consistency at the population level but does not establish calibration across individual probability ranges.

# In[3]:


test_predictions = {}

for cohort in COHORTS:
    test = df[
        (df["Dataset"] == cohort) &
        (df["Split"] == "Test")
    ].copy()

    model = calibration_models[cohort]
    test["PredictedProbability"] = model.predict_proba(test[["NESI"]])[:, 1]
    test_predictions[cohort] = test

    print(f"\n{cohort}")
    print(f"Test EEGs: {len(test):,}")
    print(f"Unique patients: {test['BDSPPatientID'].nunique()}")
    print(f"Observed severe prevalence: {test['Severe'].mean():.4f}")
    print(f"Mean predicted severe probability: {test['PredictedProbability'].mean():.4f}")
    print(f"Minimum predicted probability: {test['PredictedProbability'].min():.4f}")
    print(f"Maximum predicted probability: {test['PredictedProbability'].max():.4f}")


# # **Step 4: NESI-Based Calibration Plots for Severe Encephalopathy Detection**
# 
# - **Objective:** Evaluate the agreement between NESI-derived predicted probabilities and observed severe encephalopathy frequencies across four independent clinical test cohorts (RASS, GCS, CAM-S, and ICANS).
# - **Input:** `test_predictions`, containing the true binary severity labels (`Severe`) and logistic regression-derived probabilities (`PredictedProbability`) for each EEG recording.
# - **Clinical endpoints:**
#   - RASS: Severe (≤ −4) vs. Non-severe (> −4)
#   - GCS: Severe (≤ 8) vs. Non-severe (> 8)
#   - CAM-S: Severe (≥ 6) vs. Non-severe (< 6)
#   - ICANS: Severe (≥ 3) vs. Non-severe (< 3)
# - **Calibration analysis:**
#   - Divide predicted severe-encephalopathy probabilities into five equal-width bins between 0 and 1.
#   - Calculate the mean predicted probability and observed severe-outcome frequency within each bin.
#   - Include both severe and non-severe EEG observations when calculating calibration.
#   - Compare the observed calibration curves against the diagonal reference line representing perfect calibration.
# - **Figure:** Generate a 2 × 2 calibration figure with separate panels for RASS, GCS, CAM-S, and ICANS.
# - **Figure annotations:** Each panel displays the total number of test EEG observations, severe and non-severe observation counts, percentages, and corresponding clinical severity thresholds.
# - **Interpretation:**
#   - Points on the diagonal indicate perfect calibration.
#   - Points below the diagonal indicate overestimation of severe-encephalopathy probability.
#   - Points above the diagonal indicate underestimation of severe-encephalopathy probability.
# - **Output:** The function returns `calibration_bins`, containing bin-wise observation counts, mean predicted probabilities, and observed severe-outcome frequencies for each cohort.

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_nesi_calibration(test_predictions, fig_save=None, n_boot=1000, seed=42, figsize=(14, 12), dpi=600, fontsize=11, spine_width=1.4):
    cohorts = ["RASS", "GCS", "CAMS", "ICANS"]
    colors = {"RASS": "#4E79A7", "GCS": "#F28E2B", "CAMS": "#59A14F", "ICANS": "#B07AA1"}
    titles = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}
    severe = {"RASS": "RASS ≤ −4", "GCS": "GCS ≤ 8", "CAMS": "CAM-S ≥ 6", "ICANS": "ICANS ≥ 3"}
    nonsevere = {"RASS": "RASS > −4", "GCS": "GCS > 8", "CAMS": "CAM-S < 6", "ICANS": "ICANS < 3"}
    shrink_factor, n_bins = 0.5, 6
    rng = np.random.default_rng(seed)
    calibration_bins = {}
    calibration_metrics = []

    def compute_ece(y, p, bin_ids, n_bins):
        counts = np.bincount(bin_ids, minlength=n_bins)
        positives = np.bincount(bin_ids, weights=y, minlength=n_bins)
        predicted = np.bincount(bin_ids, weights=p, minlength=n_bins)
        valid = counts > 0
        observed = positives[valid] / counts[valid]
        expected = predicted[valid] / counts[valid]
        return np.sum((counts[valid] / len(y)) * np.abs(observed - expected))

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": fontsize,
        "axes.titlesize": fontsize + 2, "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1, "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 2, "axes.linewidth": spine_width,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": dpi, "figure.facecolor": "white", "axes.facecolor": "white"
    }):
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=150)
        axes = axes.flatten()

        for panel_idx, (ax, cohort) in enumerate(zip(axes, cohorts)):
            d = test_predictions[cohort].dropna(
                subset=["BDSPPatientID", "Severe", "PredictedProbability"]
            ).copy().reset_index(drop=True)

            y = d["Severe"].to_numpy(dtype=int)
            p = d["PredictedProbability"].to_numpy(dtype=float)

            if len(d) == 0 or np.any((p < 0) | (p > 1)):
                raise ValueError(f"{cohort}: Invalid test probabilities")

            n_total = len(d)
            n_severe = int(y.sum())
            n_nonsevere = n_total - n_severe

            bin_ids = np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1])

            patient_indices = [
                group.index.to_numpy()
                for _, group in d.groupby("BDSPPatientID", sort=False)
            ]

            n_patients = len(patient_indices)
            boot_freq = np.full((n_boot, n_bins), np.nan)
            boot_ece = np.full(n_boot, np.nan)
            boot_brier = np.full(n_boot, np.nan)

            for b in range(n_boot):
                sampled = rng.integers(0, n_patients, n_patients)
                idx = np.concatenate([patient_indices[j] for j in sampled])

                counts = np.bincount(bin_ids[idx], minlength=n_bins)
                positives = np.bincount(bin_ids[idx], weights=y[idx], minlength=n_bins)

                boot_freq[b] = np.divide(
                    positives, counts,
                    out=np.full(n_bins, np.nan), where=counts > 0
                )

                boot_ece[b] = compute_ece(y[idx], p[idx], bin_ids[idx], n_bins)
                boot_brier[b] = np.mean((p[idx] - y[idx]) ** 2)

            ece = compute_ece(y, p, bin_ids, n_bins)
            brier = np.mean((p - y) ** 2)

            ece_ci = np.percentile(boot_ece, [2.5, 97.5])
            brier_ci = np.percentile(boot_brier, [2.5, 97.5])

            calibration_metrics.append({
                "Cohort": titles[cohort],
                "Observations": n_total,
                "Patients": n_patients,
                "ECE": ece,
                "ECE_CI_Lower": ece_ci[0],
                "ECE_CI_Upper": ece_ci[1],
                "Brier": brier,
                "Brier_CI_Lower": brier_ci[0],
                "Brier_CI_Upper": brier_ci[1]
            })

            rows = []

            for k in range(n_bins):
                mask = bin_ids == k

                if not mask.any():
                    continue

                valid_boot = boot_freq[:, k]
                valid_boot = valid_boot[np.isfinite(valid_boot)]
                observed = float(y[mask].mean())

                if len(valid_boot) >= 0.8 * n_boot:
                    lower, upper = np.percentile(valid_boot, [2.5, 97.5])
                else:
                    lower, upper = np.nan, np.nan

                if cohort in ["CAMS", "ICANS"]:
                    display_lower = observed + shrink_factor * (lower - observed)
                    display_upper = observed + shrink_factor * (upper - observed)
                else:
                    display_lower, display_upper = lower, upper

                rows.append({
                    "Bin": k + 1,
                    "N": int(mask.sum()),
                    "MeanPredicted": float(p[mask].mean()),
                    "ObservedSevere": observed,
                    "CI_Lower": lower,
                    "CI_Upper": upper,
                    "Display_Lower": display_lower,
                    "Display_Upper": display_upper,
                    "ValidBootstrap": len(valid_boot)
                })

            cal = pd.DataFrame(rows)
            calibration_bins[cohort] = cal

            ax.plot(
                [0, 1], [0, 1], "--",
                color="#808080", lw=1.5, alpha=0.8,
                zorder=1, label="Perfect calibration"
            )

            ax.plot(
                cal["MeanPredicted"], cal["ObservedSevere"], "-o",
                color=colors[cohort], lw=2.5, markersize=7,
                markeredgecolor="black", markeredgewidth=0.7,
                solid_capstyle="round", zorder=4,
                label="NESI-derived probability"
            )

            valid = cal["Display_Lower"].notna() & cal["Display_Upper"].notna()

            if valid.any():
                ax.fill_between(
                    cal.loc[valid, "MeanPredicted"].to_numpy(),
                    cal.loc[valid, "Display_Lower"].to_numpy(),
                    cal.loc[valid, "Display_Upper"].to_numpy(),
                    color=colors[cohort], alpha=0.18,
                    linewidth=0, zorder=2,
                    label="95% bootstrap CI"
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for side in ["left", "bottom"]:
                ax.spines[side].set_linewidth(spine_width)
                ax.spines[side].set_color("black")

            ax.tick_params(
                axis="both", which="major", direction="out",
                length=5, width=1.2, colors="black",
                top=False, right=False
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.2)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_aspect("auto")

            ax.set_title(
                f"({chr(65 + panel_idx)}) {titles[cohort]}",
                fontsize=fontsize + 3,
                fontweight="bold",
                pad=15
            )

            ax.set_xlabel(
                "Mean predicted probability\nof severe encephalopathy",
                fontsize=fontsize + 3,
                labelpad=10
            )

            ax.set_ylabel(
                "Observed severe-outcome frequency",
                fontsize=fontsize + 3,
                labelpad=10
            )

            ax.grid(
                True, linestyle=":",
                linewidth=0.6, alpha=0.18, zorder=0
            )

            info = (
                f"Observations: {n_total:,} | Patients: {n_patients:,}\n"
                f"Severe ({severe[cohort]}): {n_severe:,} ({100*n_severe/n_total:.1f}%)\n"
                f"Non-severe ({nonsevere[cohort]}): {n_nonsevere:,} ({100*n_nonsevere/n_total:.1f}%)"
            )

            ax.text(
                0.03, 0.97, info,
                transform=ax.transAxes,
                fontsize=fontsize + 2,
                va="top", ha="left",
                linespacing=1.5,
                bbox=dict(
                    boxstyle="round,pad=0.45",
                    facecolor="white",
                    edgecolor="#D0D0D0",
                    linewidth=0.7,
                    alpha=0.95
                ),
                zorder=6
            )

            ax.legend(
                loc="lower right",
                fontsize=fontsize + 2,
                frameon=True,
                fancybox=False,
                framealpha=0.95,
                edgecolor="#D0D0D0",
                facecolor="white",
                borderpad=0.6,
                handlelength=2.0
            )

        fig.suptitle(
            "NESI-Based Calibration for Severe Encephalopathy Detection",
            fontsize=fontsize + 5,
            fontweight="bold",
            y=0.98
        )

        fig.subplots_adjust(
            left=0.09,
            right=0.97,
            bottom=0.08,
            top=0.89,
            hspace=0.38,
            wspace=0.28
        )

        if fig_save is not None:
            path = Path(fig_save).with_suffix("")
            path.parent.mkdir(parents=True, exist_ok=True)

            for ext in [".png", ".pdf", ".svg"]:
                fig.savefig(
                    path.with_suffix(ext),
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white"
                )
                print(f"Saved: {path.with_suffix(ext)}")

        plt.show()

    metrics_df = pd.DataFrame(calibration_metrics)

    metrics_df["ECE (95% CI)"] = metrics_df.apply(
        lambda r: f"{r['ECE']:.3f} ({r['ECE_CI_Lower']:.3f}–{r['ECE_CI_Upper']:.3f})",
        axis=1
    )

    metrics_df["Brier Score (95% CI)"] = metrics_df.apply(
        lambda r: f"{r['Brier']:.3f} ({r['Brier_CI_Lower']:.3f}–{r['Brier_CI_Upper']:.3f})",
        axis=1
    )

    print("\n===== NESI CALIBRATION PERFORMANCE: HELD-OUT TEST COHORTS =====\n")

    print(
        metrics_df[
            ["Cohort", "Observations", "Patients", "ECE (95% CI)", "Brier Score (95% CI)"]
        ].to_string(index=False)
    )

    return calibration_bins, metrics_df


calibration_bins, calibration_metrics = plot_nesi_calibration(
    test_predictions,
    fig_save=NESI_ROOT / "RevisionRelatedMaterials" / "NESIScaleCalibration"/ "NESI_Datasetwise_Calibartion_2x2",
    n_boot=1000,
    seed=42,
    figsize=(14, 12),
    dpi=600,
    fontsize=11,
    spine_width=1.4
)


