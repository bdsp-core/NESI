
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path

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
df["NESI"] = pd.to_numeric(df["NESI"], errors="coerce")
df["GroupedScore"] = pd.to_numeric(df["GroupedScore"], errors="coerce")
df["BDSPPatientID"] = df["BDSPPatientID"].astype("string").str.strip()
df = df.dropna(subset=["NESI", "GroupedScore", "BDSPPatientID", "Split"])
df = df[df["Dataset"].isin(["RASS", "GCS", "CAMS", "ICANS"])].copy()

models = {}
test_predictions = {}
summary = []

for cohort in ["RASS", "GCS", "CAMS", "ICANS"]:
    d = df[df["Dataset"] == cohort]
    trainval = d[d["Split"].isin(["train", "val", "validation"])].copy()
    test = d[d["Split"] == "test"].copy()

    overlap = set(trainval["BDSPPatientID"]) & set(test["BDSPPatientID"])
    if overlap:
        raise ValueError(f"{cohort}: {len(overlap)} patients overlap Train+Val and Test")

    X_train = trainval[["NESI"]].to_numpy()
    y_train = trainval["GroupedScore"].astype(int).to_numpy()
    X_test = test[["NESI"]].to_numpy()
    y_test = test["GroupedScore"].astype(int).to_numpy()

    if not set(np.unique(y_train)).issubset({0, 1, 2}):
        raise ValueError(f"{cohort}: Unexpected training categories")
    if not set(np.unique(y_test)).issubset({0, 1, 2}):
        raise ValueError(f"{cohort}: Unexpected test categories")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    models[cohort] = model
    test_predictions[cohort] = test.assign(
        TrueCategory=y_test,
        PredictedCategory=y_pred
    )

    summary.append({
        "Cohort": cohort,
        "Train+Val EEGs": len(trainval),
        "Test EEGs": len(test),
        "Test Patients": test["BDSPPatientID"].nunique()
    })

    print(f"\n{'='*65}")
    print(f"{cohort}")
    print(f"{'='*65}")
    print("Train+Val true categories:", pd.Series(y_train).value_counts().sort_index().to_dict())
    print("Test true categories:     ", pd.Series(y_test).value_counts().sort_index().to_dict())
    print("Test predicted categories:", pd.Series(y_pred).value_counts().sort_index().to_dict())
    print("Model classes:            ", model.named_steps["logisticregression"].classes_)
    print("Coefficients:             ", model.named_steps["logisticregression"].coef_.round(4).ravel())

summary_df = pd.DataFrame(summary)
print("\nCOHORT SUMMARY")
print(summary_df.to_string(index=False))


# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def bootstrap_mae_qwk(d, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby("BDSPPatientID", sort=False)]
    n_patients = len(groups)
    boot_mae, boot_qwk = [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n_patients, size=n_patients)
        sample = pd.concat([groups[i] for i in idx], ignore_index=True)
        y_true = sample["TrueCategory"].to_numpy()
        y_pred = sample["PredictedCategory"].to_numpy()

        boot_mae.append(mean_absolute_error(y_true, y_pred))
        qwk = cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2], weights="quadratic")
        if np.isfinite(qwk):
            boot_qwk.append(qwk)

    return (
        np.percentile(boot_mae, [2.5, 97.5]),
        np.percentile(boot_qwk, [2.5, 97.5])
    )

def plot_mae_qwk_bars(results_df, figsize=(12, 5), dpi=150, fontsize=13,
                      mae_ylim=(0, 0.6), qwk_ylim=(0, 1.0)):
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
    x = np.arange(len(results_df))
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    for ax, metric, ylabel, ylim in zip(
        axes,
        ["MAE", "QWK"],
        ["Mean Absolute Ordinal Error", "Quadratic-Weighted Kappa"],
        [mae_ylim, qwk_ylim]
    ):
        values = results_df[metric].to_numpy()
        lower = results_df[f"{metric}_CI_Lower"].to_numpy()
        upper = results_df[f"{metric}_CI_Upper"].to_numpy()
        errors = np.vstack([values - lower, upper - values])

        ax.bar(x, values, color=colors, width=0.65, alpha=0.85, edgecolor="black", linewidth=1)
        ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="black",
                    elinewidth=1.5, capsize=5, capthick=1.5)
        ax.set_xticks(x, ["RASS", "GCS", "CAM-S", "ICANS"])
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    return fig, axes

results_ci = []

for cohort in ["RASS", "GCS", "CAMS", "ICANS"]:
    d = test_predictions[cohort]
    y_true = d["TrueCategory"].to_numpy()
    y_pred = d["PredictedCategory"].to_numpy()

    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2], weights="quadratic")
    mae_ci, qwk_ci = bootstrap_mae_qwk(d, n_boot=1000, seed=42)

    results_ci.append({
        "Cohort": cohort,
        "N_EEGs": len(d),
        "N_Patients": d["BDSPPatientID"].nunique(),
        "MAE": mae,
        "MAE_CI_Lower": mae_ci[0],
        "MAE_CI_Upper": mae_ci[1],
        "QWK": qwk,
        "QWK_CI_Lower": qwk_ci[0],
        "QWK_CI_Upper": qwk_ci[1],
        "MAE_95CI": f"{mae:.3f} ({mae_ci[0]:.3f}–{mae_ci[1]:.3f})",
        "QWK_95CI": f"{qwk:.3f} ({qwk_ci[0]:.3f}–{qwk_ci[1]:.3f})"
    })

mae_qwk_ci_df = pd.DataFrame(results_ci)

print(mae_qwk_ci_df[
    ["Cohort", "N_EEGs", "N_Patients", "MAE_95CI", "QWK_95CI"]
].to_string(index=False))

fig, axes = plot_mae_qwk_bars(
    mae_qwk_ci_df,
    figsize=(12, 5),
    dpi=150,
    fontsize=13,
    mae_ylim=(0, 0.6),
    qwk_ylim=(0, 1.0)
)
