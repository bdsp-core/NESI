import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_pickle(filepath):
    """
    Load a Python object from a pickle file.
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj

current = Path(__file__).resolve()
NESI_ROOT = None
for parent in current.parents:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI folder not found")

# ---------------- LOAD RESULTS ----------------
gcs_death_result = NESI_ROOT / "DeathPrediction_NESIvsGCS" / "Results" / "results_GCS_Death.pkl"
nesi_death_result = NESI_ROOT / "DeathPrediction_NESIvsGCS" / "Results" / "results_NESI_Death.pkl"

results_lr_gcs = load_pickle(gcs_death_result)
results_lr_nesi = load_pickle(nesi_death_result)


# ---------------- SETTINGS ----------------
hour_points = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


# ---------------- FUNCTION ----------------
def compute_mean_ci(results_obj, metric='auroc'):
    
    metric_matrix = {h: [] for h in hour_points}

    # results_obj = list of folds
    for fold_result in results_obj:
        for h in hour_points:
            if h in fold_result:
                metric_matrix[h].append(fold_result[h][metric])

    means = []
    ci_lower = []
    ci_upper = []

    for h in hour_points:
        vals = np.array(metric_matrix[h])

        mean = vals.mean()
        std = vals.std(ddof=1)

        ci = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0

        means.append(mean)
        ci_lower.append(mean - ci)
        ci_upper.append(mean + ci)

    return means, ci_lower, ci_upper


# ---------------- COMPUTE ----------------
gcs_mean, gcs_low, gcs_up = compute_mean_ci(results_lr_gcs)
nesi_mean, nesi_low, nesi_up = compute_mean_ci(results_lr_nesi)


# ---------------- PLOT ----------------
plt.figure(figsize=(7,5), dpi=200)

# GCS
plt.plot(hour_points, gcs_mean, marker='o', label='GCS')
plt.fill_between(hour_points, gcs_low, gcs_up, alpha=0.2)

# NESI
plt.plot(hour_points, nesi_mean, marker='s', label='NESI')
plt.fill_between(hour_points, nesi_low, nesi_up, alpha=0.2)

plt.xlabel('Hour')
plt.ylabel('AUROC')
plt.xticks(hour_points)

plt.grid(True)
plt.legend()

plt.show()