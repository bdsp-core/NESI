"""
Figure 6 (main paper): PaCMAP embedding atlas, rendered from CACHED results.

Consumes the cached canonical embedding + IIC colors produced by the pipeline
(NESI/MorgothFeatureEmbedding/nesi_pacmap_main.py) -- no PaCMAP recompute, no
model inference. Produces a harmonized Figure6.{png,pdf} for MainPaperFigures/:
  A = NESI (continuous, red-blue diverging) WITH colorbar
  B = Dominant IIC category (standard ACNS palette) WITH frameless legend
Adds bold A/B panel labels (top-left) and PaCMAP axis labels.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CODES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODES_DIR))
from nesi_fig_style import (apply_style, save_fig, panel_label,
                            NESI_CMAP, NESI_VMIN, NESI_VMAX, FS_LEGEND,
                            NESI_CBAR_LABEL)

# ---- pull cached data + IIC constants from the pipeline module --------------
MFE = CODES_DIR.parents[1] / "NESI" / "MorgothFeatureEmbedding"
spec = importlib.util.spec_from_file_location("npm", MFE / "nesi_pacmap_main.py")
npm = importlib.util.module_from_spec(spec)
sys.modules["npm"] = npm
spec.loader.exec_module(npm)

coords = pd.read_csv(MFE / "NESI_pacmap_coords.csv")
d = np.load(MFE / "NESI_pacmap_iiic_data.npz", allow_pickle=True)
rgb, labels = d["rgb"], d["labels"]
assert len(coords) == len(labels) == len(rgb), "cache alignment broke"

XY = coords[["pacmap_x_nesi", "pacmap_y_nesi"]].to_numpy()
nesi = coords["NESI"].to_numpy().astype(float)
counts = pd.Series(labels).value_counts()

# jitter exactly as the pipeline does (density-matched appearance)
XY_aug, rgb_aug = npm.jitter_xy_rgb(XY, rgb)
labels_aug = np.tile(labels, npm.JITTER_COPIES + 1)
nesi_aug = np.tile(nesi, npm.JITTER_COPIES + 1)
DOT = npm.DOT_SIZE

apply_style()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.2))

# ---- Panel A: NESI continuous + colorbar -----------------------------------
order = np.argsort(np.abs(nesi_aug))
scA = axA.scatter(XY_aug[order, 0], XY_aug[order, 1], c=nesi_aug[order],
                  cmap=NESI_CMAP, vmin=NESI_VMIN, vmax=NESI_VMAX,
                  s=DOT, marker='o', edgecolors='none', alpha=0.55,
                  rasterized=True)
cb = fig.colorbar(scA, ax=axA, fraction=0.046, pad=0.02, shrink=0.85)
cb.set_label(NESI_CBAR_LABEL, fontsize=FS_LEGEND)
cb.ax.tick_params(labelsize=FS_LEGEND)

# ---- Panel B: IIC categorical (standard ACNS palette) + frameless legend ----
plot_order = ['Other', 'GRDA', 'LRDA', 'GPD', 'LPD', 'Seizure', 'Burst']
for cat in plot_order:
    m = labels_aug == cat
    if m.any():
        axB.scatter(XY_aug[m, 0], XY_aug[m, 1], c=rgb_aug[m], s=DOT,
                    marker='o', edgecolors='none', alpha=0.55, rasterized=True)
handles = [mpatches.Patch(color=npm.IIIC_PALETTE[cat],
                          label=f"{cat} (n={int(counts.get(cat, 0))})")
           for cat in npm.IIIC_LEGEND_ORDER]
axB.legend(handles=handles, loc='upper right', fontsize=FS_LEGEND,
           frameon=False, title="IIC category", title_fontsize=FS_LEGEND)

# ---- shared panel cosmetics ------------------------------------------------
for ax, letter in ((axA, "A"), (axB, "B")):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("PaCMAP 1"); ax.set_ylabel("PaCMAP 2")
    for sp in ax.spines.values():
        sp.set_visible(False)
    panel_label(ax, letter, x=0.0, y=1.02)

save_fig(fig, "Figure6")
plt.close(fig)
