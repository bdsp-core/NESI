#!/usr/bin/env python3
"""Clinician-oriented graphical abstract (R3.19).

Typical scores are observed data: the median clinical score among all EEG segments
whose NESI lies within ±0.5 of each level (Final_NESIscores_full_dataset_result.csv,
all 8,274 patients). Agreement values are the held-out test-set correlations with
patient-level bootstrap CIs, and the external values are those in Results (Fig. 7).
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(os.environ["NESI_DATA_ROOT"])
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "figures" / "final"
SCORES = ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv"
INK, MUTED, EDGE = "#222222", "#555555", "#9a9a9a"
LEVELS = [-2, -1, 0, 1, 2]
ROWS = [("GCS", "GCS", "0.72 (0.68–0.76)"), ("RASS", "RASS", "0.72 (0.66–0.76)"),
        ("CAM-S", "CAMS", "0.84 (0.79–0.88)"), ("ICANS", "ICANS", "0.83 (0.72–0.90)")]

d = pd.read_csv(SCORES, usecols=["BDSPPatientID", "Dataset", "RawScore", "NESI"], low_memory=False)
typical, counts = {}, {}
for _, ds, _ in ROWS:
    x = d[d.Dataset == ds]
    for k in LEVELS:
        near = x[(x.NESI - k).abs() < 0.5]
        typical[ds, k], counts[ds, k] = near.RawScore.median(), len(near)
        print(f"{ds} NESI {k:+d}: median score {typical[ds, k]:.1f} ({len(near)} segments)")
sparse = min(counts.values())

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9})
fig = plt.figure(figsize=(7.2, 5.05))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 7.2), ax.set_ylim(-0.2, 4.85), ax.axis("off")

ax.text(0.3, 4.58, "NESI: an EEG-derived index of encephalopathy severity", fontsize=13, fontweight="bold", color=INK, va="center")
ax.text(0.3, 4.32, "along the delirium–coma continuum, in non-agitated states", fontsize=10, color=MUTED, va="center")

# Flow
steps = [(0.3, 2.35, "Scalp EEG obtained\nfor clinical reasons"), (2.85, 1.65, "10-minute\nsegment"), (4.7, 2.2, "NESI: one continuous score\nacross four clinical scales")]
for x, w, label in steps:
    ax.add_patch(FancyBboxPatch((x, 3.45), w, 0.56, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=EDGE, lw=0.9))
    ax.text(x + w / 2, 3.73, label, ha="center", va="center", fontsize=9, color=INK)
for a, b in [(2.65, 2.85), (4.5, 4.7)]:
    ax.add_patch(FancyArrowPatch((a, 3.73), (b, 3.73), arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1, shrinkA=0, shrinkB=0))

# Severity axis
x0, x1, right = 0.95, 4.75, 5.15
pos = {k: x0 + (k + 3) / 6 * (x1 - x0) for k in LEVELS}  # same scale as the color bar, -3 to +3
grad = np.linspace(-3, 3, 400).reshape(1, -1)
ax.imshow(grad, extent=(x0, x1, 2.77, 2.95), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, zorder=1)
ax.set_xlim(0, 7.2), ax.set_ylim(-0.2, 4.85)
ax.text(x0, 3.05, "Less impaired", ha="left", fontsize=8.5, color=MUTED)
ax.text(x1, 3.05, "More impaired", ha="right", fontsize=8.5, color=MUTED)
ax.text(0.3, 2.86, "NESI", fontsize=9, fontweight="bold", va="center", color=INK)
for k, x in pos.items():
    ax.plot([x, x], [2.71, 2.77], color=INK, lw=0.8)
    ax.text(x, 2.6, f"{k:+d}".replace("-", "−") if k else "0", ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
ax.text(right, 2.86, "Agreement with clinical score\non held-out patients, ρ (95% CI)", fontsize=8, color=MUTED, va="center")
ax.plot([0.3, 6.9], [2.46, 2.46], color="#dddddd", lw=0.8)
y = 2.28
for name, ds, agree in ROWS:
    ax.text(0.3, y, name, fontsize=9, color=INK, va="center")
    for k, x in pos.items():
        ax.text(x, y, f"{typical[ds, k]:g}".replace("-", "−"), ha="center", va="center", fontsize=9, color=INK)
    ax.text(right, y, agree, fontsize=9, color=INK, va="center")
    y -= 0.25
ax.text(0.3, y + 0.03, "Numbers under the axis: median observed score among EEG segments with NESI within ±0.5 of each level",
        fontsize=7.5, color=MUTED, va="center", style="italic")
ax.text(0.3, y - 0.13, f"(8,274 patients; fewest segments in any cell: {sparse}, ICANS at +2).",
        fontsize=7.5, color=MUTED, va="center", style="italic")
ax.text(0.3, y - 0.4, "External cohort (990 patients, model not retrained): ρ 0.60 (0.55–0.64) for RASS, 0.65 (0.62–0.69) for GCS.",
        fontsize=9, color=INK, va="center")

# How to read it
notes = [
    "Higher NESI means greater impairment, whatever the cause. It includes the effects of sedatives",
    "and does not separate disease from drug effects.",
    "Results apply to adults undergoing clinically indicated EEG, in non-agitated states. EEG is not",
    "proposed for every patient in place of observational assessments. Not yet tested prospectively.",
]
for i, line in enumerate(notes):
    ax.text(0.3, 0.6 - i * 0.17 - (0.05 if i > 1 else 0), line, fontsize=9, color=INK, va="center")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "GraphicalAbstract.png", dpi=400, facecolor="white")
fig.savefig(OUT / "GraphicalAbstract.pdf", facecolor="white")
