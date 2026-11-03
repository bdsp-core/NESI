
# **Severe Encephalopathy Detection Using NESI**

# - **Clinical endpoints:** GCS ≤ 8, RASS ≤ −4, CAM-S ≥ 6, and ICANS ≥ 3.
# - **Predictor:** Continuous NESI score (higher values indicate greater severity).
# - **Threshold selection:** Optimal NESI cutoff determined on Train+Val using Youden's J statistic.
# - **Evaluation:** Apply the fixed cutoff to the independent Test set.
# - **Metrics:** Sensitivity, specificity, FPR, PPV, NPV, and AUROC.
# - **Uncertainty:** 95% confidence intervals estimated using 1,000 patient-level bootstrap resamples to account for repeated EEG recordings.
# - **No logistic regression training:** Classification is performed directly using the NESI threshold.


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


# ------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

N_BOOT = 1000
SEED = 42

endpoints = {
    "RASS": ("RASS ≤ -4", lambda x: x <= -4),
    "GCS": ("GCS ≤ 8", lambda x: x <= 8),
    "CAMS": ("CAM-S ≥ 6", lambda x: x >= 6),
    "ICANS": ("ICANS ≥ 3", lambda x: x >= 3)
}

colors = {"RASS": "#4E79A7", "GCS": "#F28E2B", "CAMS": "#59A14F", "ICANS": "#B07AA1"}
titles = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}

def calculate_metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "FPR": fp / (fp + tn) if fp + tn else np.nan,
        "PPV": tp / (tp + fp) if tp + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

def patient_bootstrap_ci(test, threshold, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    codes, patients = pd.factorize(test["BDSPPatientID"], sort=False)
    y = test["GroundTruth"].to_numpy()
    score = test["NESI"].to_numpy()
    pred = (score >= threshold).astype(int)
    metrics = ["Sensitivity", "Specificity", "FPR", "PPV", "NPV", "AUROC"]
    values = {m: [] for m in metrics}

    for _ in range(n_boot):
        sampled = rng.integers(0, len(patients), len(patients))
        counts = np.bincount(sampled, minlength=len(patients))
        w = counts[codes]

        tp = w[(y == 1) & (pred == 1)].sum()
        fn = w[(y == 1) & (pred == 0)].sum()
        tn = w[(y == 0) & (pred == 0)].sum()
        fp = w[(y == 0) & (pred == 1)].sum()

        if tp + fn == 0 or tn + fp == 0:
            continue

        result = {
            "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp),
            "FPR": fp / (fp + tn),
            "PPV": tp / (tp + fp) if tp + fp else np.nan,
            "NPV": tn / (tn + fn) if tn + fn else np.nan,
            "AUROC": roc_auc_score(y, score, sample_weight=w)
        }

        for m in metrics:
            if np.isfinite(result[m]):
                values[m].append(result[m])

    return {
        m: np.percentile(values[m], [2.5, 97.5]) if values[m]
        else [np.nan, np.nan]
        for m in metrics
    }

def patient_bootstrap_roc_band(test, n_boot=1000, seed=42, n_points=201):
    rng = np.random.default_rng(seed)
    codes, patients = pd.factorize(test["BDSPPatientID"], sort=False)
    y = test["GroundTruth"].to_numpy()
    score = test["NESI"].to_numpy()
    fpr_grid = np.linspace(0, 1, n_points)
    boot_tpr = []

    for _ in range(n_boot):
        sampled = rng.integers(0, len(patients), len(patients))
        counts = np.bincount(sampled, minlength=len(patients))
        w = counts[codes]

        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue

        fpr_b, tpr_b, _ = roc_curve(y, score, sample_weight=w)
        tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        boot_tpr.append(tpr_interp)

    if not boot_tpr:
        raise ValueError("No valid bootstrap ROC curves")

    boot_tpr = np.asarray(boot_tpr)
    lower = np.percentile(boot_tpr, 2.5, axis=0)
    upper = np.percentile(boot_tpr, 97.5, axis=0)
    return fpr_grid, lower, upper

def format_ci(value, ci):
    return f"{value:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"

results = []
roc_data = {}

for cohort, (endpoint, condition) in endpoints.items():
    d = df[df["Dataset"] == cohort].copy()
    d["GroundTruth"] = condition(d["RawScore"]).astype(int)

    trainval = d[d["Split"].isin(["train", "val", "validation"])].copy()
    test = d[d["Split"] == "test"].copy().reset_index(drop=True)

    overlap = set(trainval["BDSPPatientID"]) & set(test["BDSPPatientID"])
    if overlap:
        raise ValueError(f"{cohort}: {len(overlap)} overlapping patients")

    if trainval["GroundTruth"].nunique() < 2 or test["GroundTruth"].nunique() < 2:
        raise ValueError(f"{cohort}: Both classes required in Train+Val and Test")

    y_train = trainval["GroundTruth"].to_numpy()
    score_train = trainval["NESI"].to_numpy()

    fpr, tpr, thresholds = roc_curve(y_train, score_train)
    youden = tpr - fpr
    valid = np.flatnonzero(np.isfinite(thresholds))
    best_idx = valid[np.argmax(youden[valid])]
    threshold = float(thresholds[best_idx])

    y_test = test["GroundTruth"].to_numpy()
    score_test = test["NESI"].to_numpy()

    metrics = calculate_metrics(y_test, score_test, threshold)
    auc = roc_auc_score(y_test, score_test)
    ci = patient_bootstrap_ci(test, threshold, N_BOOT, SEED)

    row = {
        "Cohort": cohort,
        "Endpoint": endpoint,
        "Test EEGs": len(test),
        "Test Patients": test["BDSPPatientID"].nunique(),
        "Severe EEGs": int(y_test.sum()),
        "NESI Threshold": threshold,
        "Sensitivity (95% CI)": format_ci(metrics["Sensitivity"], ci["Sensitivity"]),
        "Specificity (95% CI)": format_ci(metrics["Specificity"], ci["Specificity"]),
        "FPR (95% CI)": format_ci(metrics["FPR"], ci["FPR"]),
        "PPV (95% CI)": format_ci(metrics["PPV"], ci["PPV"]),
        "NPV (95% CI)": format_ci(metrics["NPV"], ci["NPV"]),
        "AUROC (95% CI)": format_ci(auc, ci["AUROC"]),
        "TP": metrics["TP"], "FP": metrics["FP"],
        "TN": metrics["TN"], "FN": metrics["FN"]
    }
    results.append(row)

    fpr_test, tpr_test, _ = roc_curve(y_test, score_test)
    fpr_grid, tpr_lower, tpr_upper = patient_bootstrap_roc_band(test, N_BOOT, SEED)

    roc_data[cohort] = {
        "fpr_train": fpr,
        "tpr_train": tpr,
        "fpr_test": fpr_test,
        "tpr_test": tpr_test,
        "fpr_grid": fpr_grid,
        "tpr_lower": tpr_lower,
        "tpr_upper": tpr_upper,
        "auc_train": roc_auc_score(y_train, score_train),
        "auc_test": auc,
        "auc_ci": ci["AUROC"],
        "threshold": threshold,
        "youden": youden[best_idx],
        "n_train": len(trainval),
        "n_test": len(test),
        "n_patients": test["BDSPPatientID"].nunique()
    }

    print(f"\n{'='*65}")
    print(f"{endpoint}")
    print(f"{'='*65}")
    print(f"Train+Val: {len(trainval)} EEGs")
    print(f"Test: {len(test)} EEGs | {test['BDSPPatientID'].nunique()} patients")
    print(f"Optimal NESI threshold: ≥ {threshold:.4f}")
    print(f"Train+Val Youden J: {youden[best_idx]:.4f}")
    print(f"Sensitivity: {row['Sensitivity (95% CI)']}")
    print(f"Specificity: {row['Specificity (95% CI)']}")
    print(f"AUROC: {row['AUROC (95% CI)']}")
    print(f"TP={metrics['TP']} | FP={metrics['FP']} | TN={metrics['TN']} | FN={metrics['FN']}")

severe_results_df = pd.DataFrame(results)


def plot_nesi_roc(roc_data, fig_save=None, figsize=(24, 6), dpi=600, fontsize=11, spine_width=1.4):
    cohorts = ["RASS", "GCS", "CAMS", "ICANS"]
    colors = {"RASS": "#4E79A7", "GCS": "#F28E2B", "CAMS": "#59A14F", "ICANS": "#B07AA1"}
    titles = {"RASS": "RASS ≤ −4", "GCS": "GCS ≤ 8", "CAMS": "CAM-S ≥ 6", "ICANS": "ICANS ≥ 3"}

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": fontsize,
        "axes.titlesize": fontsize + 3, "axes.labelsize": fontsize + 2,
        "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize, "axes.linewidth": spine_width,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white"
    }):
        fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=150)

        for i, (ax, cohort) in enumerate(zip(axes, cohorts)):
            r = roc_data[cohort]
            color = colors[cohort]

            ax.plot(
                [0, 1], [0, 1], "--",
                color="gray", lw=1.5, alpha=0.8,
                zorder=1, label="_nolegend_"
            )
            ax.fill_between(
                r["fpr_grid"], r["tpr_lower"], r["tpr_upper"],
                color=color, alpha=0.18, linewidth=0,
                label="95% bootstrap CI", zorder=2
            )
            ax.plot(
                r["fpr_test"], r["tpr_test"],
                "-", color=color, lw=2.5,
                label="Test ROC", zorder=4
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for side in ["left", "bottom"]:
                ax.spines[side].set_linewidth(spine_width)
                ax.spines[side].set_color("black")

            ax.tick_params(
                axis="both", which="major", direction="out",
                length=5, width=1.2, colors="black", top=False, right=False
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.03)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_aspect("auto")
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.18)

            ax.set_title(
                f"({chr(65+i)}) {titles[cohort]}",
                fontsize=fontsize + 3, fontweight="bold", pad=15
            )
            ax.set_xlabel("False positive rate", fontsize=fontsize + 5, labelpad=10)
            ax.set_ylabel("True positive rate", fontsize=fontsize + 5, labelpad=10)

            info = (
                f"Test AUC (95% CI): {r['auc_test']:.3f} "
                f"({r['auc_ci'][0]:.3f}–{r['auc_ci'][1]:.3f})\n"
                f"Optimal NESI threshold: ≥ {r['threshold']:.3f}"
            )
            ax.text(
                0.97, 0.05, info, transform=ax.transAxes,
                fontsize=fontsize+3, va="bottom", ha="right",
                linespacing=1.5,
                bbox=dict(
                    boxstyle="round,pad=0.45", facecolor="white",
                    edgecolor="#D0D0D0", linewidth=0.7, alpha=0.95
                ), zorder=6
            )
            ax.legend(
                loc="center right", fontsize=fontsize+3,
                frameon=True, fancybox=False, framealpha=0.95,
                edgecolor="#D0D0D0", facecolor="white",
                borderpad=0.6, handlelength=2
            )

        fig.suptitle(
            "NESI-Based ROC Analysis for Severe Encephalopathy Detection\n"
            "Threshold selected on Train/Validation using Youden's J",
            fontsize=fontsize + 5, fontweight="bold", y=1.03,
            linespacing=1.5
        )
        fig.subplots_adjust(
            left=0.045, right=0.99, bottom=0.16,
            top=0.79, wspace=0.22
        )

        if fig_save is not None:
            path = Path(fig_save).with_suffix("")
            path.parent.mkdir(parents=True, exist_ok=True)
            for ext in [".png", ".pdf", ".svg"]:
                fig.savefig(
                    path.with_suffix(ext), dpi=dpi,
                    bbox_inches="tight", facecolor="white"
                )
                print(f"Saved: {path.with_suffix(ext)}")

        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_nesi_roc(roc_data, fig_save=None, figsize=(14, 12), dpi=60, fontsize=11, spine_width=1.4):
    cohorts = ["RASS", "GCS", "CAMS", "ICANS"]
    colors = {"RASS": "#4E79A7", "GCS": "#F28E2B", "CAMS": "#59A14F", "ICANS": "#B07AA1"}
    titles = {"RASS": "RASS ≤ −4", "GCS": "GCS ≤ 8", "CAMS": "CAM-S ≥ 6", "ICANS": "ICANS ≥ 3"}

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": fontsize,
        "axes.titlesize": fontsize + 3, "axes.labelsize": fontsize + 2,
        "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize, "axes.linewidth": spine_width,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white"
    }):
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=150)
        axes = axes.flatten()

        for i, (ax, cohort) in enumerate(zip(axes, cohorts)):
            r = roc_data[cohort]
            color = colors[cohort]

            ax.plot(
                [0, 1], [0, 1], "--",
                color="gray", lw=1.5, alpha=0.8,
                zorder=1, label="_nolegend_"
            )

            ax.fill_between(
                r["fpr_grid"], r["tpr_lower"], r["tpr_upper"],
                color=color, alpha=0.18, linewidth=0,
                label="95% bootstrap CI", zorder=2
            )

            ax.plot(
                r["fpr_test"], r["tpr_test"],
                "-", color=color, lw=2.5,
                label="Test ROC", zorder=4
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for side in ["left", "bottom"]:
                ax.spines[side].set_linewidth(spine_width)
                ax.spines[side].set_color("black")

            ax.tick_params(
                axis="both", which="major", direction="out",
                length=5, width=1.2, colors="black", top=False, right=False
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.03)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_aspect("auto")
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.18)

            ax.set_title(
                f"({chr(65+i)}) {titles[cohort]}",
                fontsize=fontsize + 3, fontweight="bold", pad=15
            )

            ax.set_xlabel("False positive rate", fontsize=fontsize + 5, labelpad=10)
            ax.set_ylabel("True positive rate", fontsize=fontsize + 5, labelpad=10)

            info = (
                f"Test AUC (95% CI): {r['auc_test']:.3f} "
                f"({r['auc_ci'][0]:.3f}–{r['auc_ci'][1]:.3f})\n"
                f"Optimal NESI threshold: ≥ {r['threshold']:.3f}"
            )

            ax.text(
                0.97, 0.05, info, transform=ax.transAxes,
                fontsize=fontsize + 3, va="bottom", ha="right",
                linespacing=1.5,
                bbox=dict(
                    boxstyle="round,pad=0.45", facecolor="white",
                    edgecolor="#D0D0D0", linewidth=0.7, alpha=0.95
                ), zorder=6
            )

            ax.legend(
                loc="center right", fontsize=fontsize + 3,
                frameon=True, fancybox=False, framealpha=0.95,
                edgecolor="#D0D0D0", facecolor="white",
                borderpad=0.6, handlelength=2
            )

        fig.suptitle(
            "NESI-Based ROC Analysis for Severe Encephalopathy Detection\n"
            "Threshold selected on Train/Validation using Youden's J",
            fontsize=fontsize + 5, fontweight="bold", y=0.99,
            linespacing=1.5
        )

        fig.subplots_adjust(
            left=0.09, right=0.97, bottom=0.08,
            top=0.87, hspace=0.38, wspace=0.28
        )

        if fig_save is not None:
            path = Path(fig_save).with_suffix("")
            path.parent.mkdir(parents=True, exist_ok=True)

            for ext in [".png", ".pdf", ".svg"]:
                fig.savefig(
                    path.with_suffix(ext), dpi=dpi,
                    bbox_inches="tight", facecolor="white"
                )
                print(f"Saved: {path.with_suffix(ext)}")

        plt.show()


plot_nesi_roc(
    roc_data,
    fig_save=NESI_ROOT / "RevisionRelatedMaterials" / "NESISeverityDetection" / "NESI_Datasetwise_Severe_Enchepalopathy_ROC_2x2",
    figsize=(14, 12),
    dpi=50,
    fontsize=5,
    spine_width=1.4
)
print(severe_results_df.to_string(index=False))

