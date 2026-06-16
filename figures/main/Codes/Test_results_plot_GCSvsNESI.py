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

# ---------------- LOAD RESULTS ----------------
gcs_death_result = NESI_ROOT / "NESI" /  "DeathPrediction_NESIvsGCS" / "Results" / "results_GCS_Death.pkl"
nesi_death_result = NESI_ROOT / "NESI" / "DeathPrediction_NESIvsGCS" / "Results" / "results_NESI_Death.pkl"

results_lr_gcs = load_pickle(gcs_death_result)
results_lr_nesi = load_pickle(nesi_death_result)


# ---------------- SETTINGS ----------------
hour_points = [4, 6, 8, 10, 12, 14, 16, 18, 20]


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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nesi_fig_style import apply_style, save_fig, SERIES_COLORS, FS_LEGEND
apply_style()

fig, ax = plt.subplots(figsize=(6.0, 4.2))

# NESI (orange) plotted first so it reads as the lead series
ax.plot(hour_points, nesi_mean, marker='s', markersize=5, linewidth=1.8,
        linestyle='-', color=SERIES_COLORS['NESI'], markeredgecolor='white',
        markeredgewidth=0.6, label='NESI', zorder=3)
ax.fill_between(hour_points, nesi_low, nesi_up, color=SERIES_COLORS['NESI'],
                alpha=0.15, linewidth=0, zorder=1)

ax.plot(hour_points, gcs_mean, marker='o', markersize=5, linewidth=1.8,
        linestyle='--', color=SERIES_COLORS['GCS'], markeredgecolor='white',
        markeredgewidth=0.6, label='GCS', zorder=3)
ax.fill_between(hour_points, gcs_low, gcs_up, color=SERIES_COLORS['GCS'],
                alpha=0.18, linewidth=0, zorder=1)

ax.set_xlabel('Hours since monitoring onset')
ax.set_ylabel('AUROC')
ax.set_xticks(hour_points)
ax.grid(axis='y', linewidth=0.5, alpha=0.35)
ax.set_axisbelow(True)

leg = ax.legend(loc='lower left', frameon=False, title='Score (shaded: 95% CI)',
                title_fontsize=FS_LEGEND)
leg._legend_box.align = 'left'

save_fig(fig, 'Figure5')
plt.close(fig)