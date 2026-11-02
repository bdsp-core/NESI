#!/usr/bin/env python
# coding: utf-8

# In[4]:

from pathlib import Path
import pandas as pd

current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")


df_24_model_result =pd.read_csv(NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" / "Test_Result_NESI_for_24_Models.csv")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

def compare_24_nesi_models(df, dataset="RASS", split="Test", n_boot=1000, seed=42):
    data = df[(df["Dataset"].str.upper() == dataset.upper()) &
              (df["Split"].str.lower() == split.lower())].copy()

    models = ["NESI"] + [f"NESIScore_FromModel{i}" for i in range(2, 25)]
    data = data.dropna(subset=["TransformedScore"] + models)
    y_true = data["TransformedScore"].to_numpy(dtype=float)
    predictions = data[models].to_numpy(dtype=float)

    n = len(data)
    if n < 3:
        raise ValueError("Insufficient samples after filtering.")

    rng = np.random.default_rng(seed)
    boot_corrs = np.full((n_boot, len(models)), np.nan)

    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        for j in range(len(models)):
            boot_corrs[b, j] = spearmanr(y_true[idx], predictions[idx, j]).statistic

    if not np.isfinite(boot_corrs).all():
        raise ValueError("Undefined bootstrap correlations detected. Check constant scores or predictions.")

    results = []
    p_values = []

    for j, model in enumerate(models):
        rho = boot_corrs[:, j]
        mean_rho = np.mean(rho)
        ci_low, ci_high = np.percentile(rho, [2.5, 97.5])

        if j == 0:
            p_value = np.nan
            delta_rho = np.nan
        else:
            delta_rho = np.mean(boot_corrs[:, 0] - rho)
            p_value = wilcoxon(boot_corrs[:, 0], rho, alternative="two-sided").pvalue
            p_values.append(p_value)

        results.append({
            "Model": "Proposed NESI" if j == 0 else f"Model {j+1}",
            "N": n,
            "Mean_Rho": mean_rho,
            "CI_Lower": ci_low,
            "CI_Upper": ci_high,
            "Delta_Rho_ProposedMinusAlternative": delta_rho,
            "p_value": p_value
        })

    results = pd.DataFrame(results)
    _, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    results["p_Holm"] = np.r_[np.nan, p_holm]
    results["Significant"] = results["p_Holm"] < 0.05
    results["Winner"] = np.where(
        results["Significant"],
        np.where(results["Delta_Rho_ProposedMinusAlternative"] > 0, "Proposed NESI", "Alternative"),
        "No significant difference"
    )
    results.loc[0, ["Significant", "Winner"]] = [False, "Reference"]

    print(f"\nDataset: {dataset.upper()} | Split: {split} | N = {n}")
    print(f"Bootstrap: {n_boot} | Seed: {seed} | Holm comparisons: 23\n")
    print(results.round(4).to_string(index=False))

    return results, boot_corrs


import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

def compare_24_nesi_models2(df, dataset="RASS", split="Test", n_boot=1000, seed=42):
    data = df[(df["Dataset"].str.upper() == dataset.upper()) &
              (df["Split"].str.lower() == split.lower())].copy()

    models = ["NESI"] + [f"NESIScore_FromModel{i}" for i in range(2, 25)]
    data = data.dropna(subset=["TransformedScore"] + models)

    y_true = data["TransformedScore"].to_numpy(dtype=float)
    predictions = data[models].to_numpy(dtype=float)
    n = len(data)

    if n < 3:
        raise ValueError("Insufficient samples after filtering.")

    rng = np.random.default_rng(seed)
    boot_corrs = np.full((n_boot, len(models)), np.nan)

    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        for j in range(len(models)):
            boot_corrs[b, j] = spearmanr(y_true[idx], predictions[idx, j]).statistic

    if not np.isfinite(boot_corrs).all():
        raise ValueError("Undefined bootstrap correlations detected.")

    results, p_values = [], []

    for j, model in enumerate(models):
        rho = boot_corrs[:, j]
        mean_rho = np.mean(rho)
        ci_low, ci_high = np.percentile(rho, [2.5, 97.5])

        if j == 0:
            delta_rho, delta_ci_low, delta_ci_high, p_value = np.nan, np.nan, np.nan, np.nan
        else:
            delta = boot_corrs[:, 0] - rho
            delta_rho = np.mean(delta)
            delta_ci_low, delta_ci_high = np.percentile(delta, [2.5, 97.5])
            p_value = min(1.0, 2 * min((np.sum(delta <= 0) + 1) / (n_boot + 1),
                                       (np.sum(delta >= 0) + 1) / (n_boot + 1)))
            p_values.append(p_value)

        results.append({
            "Model": "Proposed NESI" if j == 0 else f"Model {j+1}",
            "N": n,
            "Mean_Rho": mean_rho,
            "CI_Lower": ci_low,
            "CI_Upper": ci_high,
            "Delta_Rho": delta_rho,
            "Delta_CI_Lower": delta_ci_low,
            "Delta_CI_Upper": delta_ci_high,
            "p_value": p_value
        })

    results = pd.DataFrame(results)
    _, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    results["p_Holm"] = np.r_[np.nan, p_holm]
    results["Significant"] = results["p_Holm"] < 0.05
    results["Winner"] = np.where(
        results["Significant"],
        np.where(results["Delta_Rho"] > 0, "Proposed NESI", "Alternative"),
        "No significant difference"
    )
    results.loc[0, ["Significant", "Winner"]] = [False, "Reference"]

    print(f"\nDataset: {dataset.upper()} | Split: {split} | N = {n}")
    print(f"Bootstrap: {n_boot} | Seed: {seed} | Holm comparisons: 23\n")
    print(results.round(4).to_string(index=False))

    return results, boot_corrs


results_RASS, boot_RASS = compare_24_nesi_models2(df_24_model_result, dataset="RASS")
results_RASS.to_csv(NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" / "RASS24_model_Result.csv",
    index=False)


results_GCS, boot_GCS = compare_24_nesi_models2(df_24_model_result, dataset="GCS")
results_GCS.to_csv(NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" / "GC24_model_Result.csv",
    index=False)


results_CAMS, boot_CAMS = compare_24_nesi_models2(df_24_model_result, dataset="CAMS")
results_CAMS.to_csv(NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" / "CAMS24_model_Result.csv",
    index=False)


results_ICANS, boot_ICANS = compare_24_nesi_models2(df_24_model_result, dataset="ICANS")
results_ICANS.to_csv(NESI_ROOT / "RevisionRelatedMaterials" / "SPECTRA_SeverityDefinition_Ablation" / "ICANS24_model_Result.csv",
    index=False)
