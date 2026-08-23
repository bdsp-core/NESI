"""
Plot continuous CORN ordinal score over fixed RASS background strips.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

current = Path(__file__).resolve()
CONTPRED_ROOT = None

for parent in current.parents:
    if parent.name == "KIMCHI_LAB_DATA":
        CONTPRED_ROOT = parent
        break

if CONTPRED_ROOT is None:
    raise RuntimeError("KIMCHI_LAB_DATA folder not found.")

PRED_ROOT = CONTPRED_ROOT / "RASSPredictions"
prediction_csv_files = sorted(PRED_ROOT.glob("*.csv"))

if not prediction_csv_files:
    raise FileNotFoundError(f"No prediction CSV files found in {PRED_ROOT}")

for pred_csv_path in tqdm(prediction_csv_files, desc="Plotting RASS predictions"):
    pred_csv_filename = pred_csv_path.stem
    subject_id = pred_csv_filename.split("_")[0]

    preds_df = pd.read_csv(pred_csv_path)

    logit_cols = ["logit_0", "logit_1", "logit_2", "logit_3", "logit_4"]
    logits = preds_df[logit_cols].values

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    cond_probs = sigmoid(logits)
    ordinal_score = cond_probs.sum(axis=1)
    rass_score = ordinal_score - 5.0

    rass_hard = preds_df["RASSMappingClass"].values

    print(f"N samples: {len(rass_score)}")
    print(f"Continuous RASS score range: [{rass_score.min():.3f}, {rass_score.max():.3f}]")

    fig, ax = plt.subplots(figsize=(20, 4))

    rass_labels = ["RASS 0", "RASS -1", "RASS -2", "RASS -3", "RASS -4", "RASS -5"]
    rass_centers = [0, -1, -2, -3, -4, -5]
    strip_colors = [
        "#c7c7ff",
        "#aebbf0",
        "#b3eee0",
        "#f3f3b0",
        "#f5cf9e",
        "#e3b3a8",
    ]

    y_min, y_max = -5.5, 0.5
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, len(rass_score))

    for center, color in zip(rass_centers, strip_colors):
        ax.axhspan(center - 0.5, center + 0.5, color=color, zorder=0)

    for boundary in np.arange(y_min, y_max + 0.001, 1.0):
        ax.axhline(boundary, color="black", linestyle="--", linewidth=1, zorder=1)

    ax.plot(np.arange(len(rass_score)), rass_score, color="blue", linewidth=0.8, zorder=2)

    ax.set_yticks(rass_centers)
    ax.set_yticklabels(rass_labels, fontsize=11, ha="left")
    ax.tick_params(axis="y", pad=60)

    ax.set_xlabel("Time index (window)")
    ax.set_title(f"Continuous CORN Ordinal Score over RASS Levels (SubjectID: {subject_id})")

    plt.tight_layout()

    OUT_PATH = CONTPRED_ROOT / "Plots_RASSPred" / f"{subject_id}_RASS_continuous_score.png"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=250)
    print(f"Saved plot to: {OUT_PATH}")

    plt.show(block=False)
    plt.pause(2)
    plt.close()