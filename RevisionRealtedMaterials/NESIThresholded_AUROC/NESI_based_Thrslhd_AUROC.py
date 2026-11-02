

# **Threshold-Specific AUROC Analysis**

# - **Objective:** Evaluate NESI performance across clinically relevant severity thresholds for RASS, GCS, CAM-S, and ICANS.
# - **Dataset:** Held-out test set for each neurological scale.
# - **Thresholds:** Clinical scores were dichotomized at every available cut-point (k) into ≤ k versus > k.
# - **AUROC Calculation:** AUROC was computed using continuous NESI scores without additional model fitting.
# - **Confidence Intervals:** 95% CIs were estimated using patient-level bootstrap resampling (1,000 iterations).
# - **Visualization:** A four-panel figure illustrates threshold-specific AUROC across all four scales, with shaded regions representing 95% CIs.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

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
df["Dataset"] = df["Dataset"].astype(str).str.strip().str.upper()
df["Split"] = df["Split"].astype(str).str.strip().str.lower()
df["RawScore"] = pd.to_numeric(df["RawScore"], errors="coerce")
df["NESI"] = pd.to_numeric(df["NESI"], errors="coerce")
df["BDSPPatientID"] = df["BDSPPatientID"].astype("string").str.strip()

df = df[df["Dataset"].isin(["GCS", "RASS", "CAMS", "ICANS"])].copy()
df = df.dropna(subset=["BDSPPatientID", "RawScore", "NESI", "Split"])

for cohort in ["GCS", "RASS", "CAMS", "ICANS"]:
    d = df[df["Dataset"] == cohort]
    trainval = d[d["Split"].isin(["train", "val", "validation"])]
    test = d[d["Split"] == "test"]
    overlap = set(trainval["BDSPPatientID"]) & set(test["BDSPPatientID"])

    print(f"\n{'='*60}")
    print(f"{cohort}")
    print(f"{'='*60}")
    print(f"Train+Val: {len(trainval)} EEGs | {trainval['BDSPPatientID'].nunique()} patients")
    print(f"Test:      {len(test)} EEGs | {test['BDSPPatientID'].nunique()} patients")
    print(f"Patient overlap: {len(overlap)}")
    print("\nTest clinical score distribution:")
    print(test["RawScore"].value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

cohorts = ["GCS", "RASS", "CAMS", "ICANS"]
all_auc_results = []

for cohort in cohorts:
    d = df[(df["Dataset"] == cohort) & (df["Split"] == "test")].copy()
    raw = d["RawScore"].to_numpy()
    nesi = d["NESI"].to_numpy()

    min_score = int(raw.min())
    max_score = int(raw.max())

    for k in range(min_score, max_score):
        if cohort in ["GCS", "RASS"]:
            y = (raw <= k).astype(int)
            endpoint = f"{cohort} ≤ {k} vs > {k}"
        else:
            y = (raw > k).astype(int)
            endpoint = f"{cohort} > {k} vs ≤ {k}"

        n_positive = int(y.sum())
        n_negative = int((y == 0).sum())

        auc = roc_auc_score(y, nesi) if len(np.unique(y)) == 2 else np.nan

        all_auc_results.append({
            "Cohort": cohort,
            "Cutoff": k,
            "Endpoint": endpoint,
            "N_Positive": n_positive,
            "N_Negative": n_negative,
            "AUROC": auc
        })

auc_df = pd.DataFrame(all_auc_results)

for cohort in cohorts:
    print(f"\n{'='*75}")
    print(f"{cohort}: THRESHOLD-SPECIFIC AUROC")
    print(f"{'='*75}")
    print(
        auc_df.loc[auc_df["Cohort"] == cohort,
                   ["Endpoint", "N_Positive", "N_Negative", "AUROC"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = 1000
SEED = 42
auc_ci_results = []

for cohort in ["GCS", "RASS", "CAMS", "ICANS"]:
    d = df[(df["Dataset"] == cohort) & (df["Split"] == "test")].copy().reset_index(drop=True)
    raw = d["RawScore"].to_numpy()
    nesi = d["NESI"].to_numpy()
    patient_codes, patients = pd.factorize(d["BDSPPatientID"], sort=False)
    n_patients = len(patients)
    rng = np.random.default_rng(SEED)

    bootstrap_weights = []
    for b in range(N_BOOT):
        sampled = rng.integers(0, n_patients, n_patients)
        counts = np.bincount(sampled, minlength=n_patients)
        bootstrap_weights.append(counts[patient_codes])

    cohort_results = auc_df[auc_df["Cohort"] == cohort]

    for _, row in cohort_results.iterrows():
        k = int(row["Cutoff"])
        y = (raw <= k).astype(int) if cohort in ["GCS", "RASS"] else (raw > k).astype(int)
        boot_aucs = []

        if len(np.unique(y)) == 2:
            for weights in bootstrap_weights:
                if weights[y == 0].sum() == 0 or weights[y == 1].sum() == 0:
                    continue
                boot_aucs.append(roc_auc_score(y, nesi, sample_weight=weights))

        if boot_aucs:
            lower, upper = np.percentile(boot_aucs, [2.5, 97.5])
        else:
            lower, upper = np.nan, np.nan

        auc_ci_results.append({
            "Cohort": cohort,
            "Cutoff": k,
            "Endpoint": row["Endpoint"],
            "N_Positive": int(row["N_Positive"]),
            "N_Negative": int(row["N_Negative"]),
            "AUROC": float(row["AUROC"]),
            "CI_Lower": lower,
            "CI_Upper": upper,
            "Valid_Bootstraps": len(boot_aucs),
            "AUROC_95CI": f"{row['AUROC']:.3f} ({lower:.3f}–{upper:.3f})"
        })

auc_ci_df = pd.DataFrame(auc_ci_results)

for cohort in ["GCS", "RASS", "CAMS", "ICANS"]:
    print(f"\n{'='*90}")
    print(f"{cohort}: THRESHOLD-SPECIFIC AUROC WITH 95% CI")
    print(f"{'='*90}")
    print(
        auc_ci_df.loc[
            auc_ci_df["Cohort"] == cohort,
            ["Endpoint", "N_Positive", "N_Negative", "AUROC_95CI", "Valid_Bootstraps"]
        ].to_string(index=False)
    )
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_threshold_auroc(auc_ci_df, layout=(1, 4), figsize=(20, 4.5),
                         ylims=None, dpi=150, fig_save_path=None):

    cohorts = ["RASS", "GCS", "CAMS", "ICANS"]
    titles = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}
    colors = {"RASS": "#4E79A7", "GCS": "#F28E2B", "CAMS": "#59A14F", "ICANS": "#B07AA1"}

    if layout[0] * layout[1] != 4:
        raise ValueError("layout must contain exactly 4 panels")

    if ylims is None:
        ylims = {cohort: (0.75, 1.02) for cohort in cohorts}

    fig, axes = plt.subplots(*layout, figsize=figsize, dpi=dpi, squeeze=False)

    for ax, cohort in zip(axes.flat, cohorts):
        d = auc_ci_df[auc_ci_df["Cohort"] == cohort].sort_values("Cutoff")
        x = d["Cutoff"].to_numpy()
        y = d["AUROC"].to_numpy()
        low = d["CI_Lower"].to_numpy()
        high = d["CI_Upper"].to_numpy()

        ax.plot(x, y, "-o", color=colors[cohort], lw=2, ms=5, label="AUROC")
        ax.fill_between(x, low, high, color=colors[cohort], alpha=0.2, label="95% CI")

        ax.set_title(titles[cohort], fontsize=14, fontweight="bold")

        if cohort in ["RASS", "GCS"]:
            xlabel = f"Comparison level ({titles[cohort]} ≤ k vs > k)"
        else:
            xlabel = f"Comparison level ({titles[cohort]} > k vs ≤ k)"

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("AUROC", fontsize=14)
        ax.set_xticks(x)
        ax.set_ylim(*ylims[cohort])
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=12, loc="lower right")

        # Remove top and right subplot borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Figure-level title
    fig.suptitle(
        "NESI Discriminatory Performance Across Clinical Severity Thresholds",
        fontsize=17,
        fontweight="bold",
        y=1.03
    )

    plt.tight_layout()

    if fig_save_path is not None:
        fig_save_path = Path(fig_save_path)
        fig_save_path.parent.mkdir(parents=True, exist_ok=True)

        if fig_save_path.suffix.lower() not in [".pdf", ".png"]:
            raise ValueError("fig_save_path must end with .pdf or .png")

        fig.savefig(fig_save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {fig_save_path}")

    plt.show()
    return fig, axes


fig, axes = plot_threshold_auroc(
    auc_ci_df,
    layout=(1, 4),
    figsize=(20, 4.5),
    ylims={
        "RASS": (0.68, 0.95),
        "GCS": (0.75, 0.95),
        "CAMS": (0.7, 1.02),
        "ICANS": (0.5, 1.02)
    },
    dpi=90,
    fig_save_path = NESI_ROOT / "RevisionRelatedMaterials" / "NESIThresholded_AUROC" / "Klevel_Threshold_Specific_AUROC.png"
)
