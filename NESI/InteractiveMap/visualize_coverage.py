"""
Visualize how well the deployed subsample covers the full PaCMAP map.
Renders a 4-panel comparison at different target-N values so we can pick
the sweet spot between coverage and gh-pages payload size.

Output: NESI/InteractiveMap/coverage.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
COORDS_CSV = (SCRIPT_DIR.parent / "MorgothFeatureEmbedding"
              / "NESI_pacmap_coords.csv")
PNG_DIR = SCRIPT_DIR / "eeg_pngs"
OUT = SCRIPT_DIR / "coverage.png"

TARGETS = [500, 1500, 3000, 5000]
PNG_BYTES_AVG = 780_000     # measured during the batch render


def _grid_for(target_n: int) -> int:
    # Heuristic: with ~50% cell fill on this map, grid^2 ≈ 2 * target_n
    # gives close to `target_n` occupied cells.
    return int(np.ceil(np.sqrt(target_n * 2.2)))


def subsample(pool: pd.DataFrame, target_n: int, grid: int | None = None):
    if grid is None:
        grid = _grid_for(target_n)
    xs = pool.pacmap_x_nesi.to_numpy()
    ys = pool.pacmap_y_nesi.to_numpy()
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xc = np.clip(((xs - xmin) / (xmax - xmin + 1e-9) * grid).astype(int),
                  0, grid - 1)
    yc = np.clip(((ys - ymin) / (ymax - ymin + 1e-9) * grid).astype(int),
                  0, grid - 1)
    p = pool.assign(cell=xc * grid + yc)
    selected = []
    cell_w = (xmax - xmin) / grid
    cell_h = (ymax - ymin) / grid
    for cell_id, grp in p.groupby('cell'):
        ix = cell_id // grid
        iy = cell_id %  grid
        cx = xmin + (ix + 0.5) * cell_w
        cy = ymin + (iy + 0.5) * cell_h
        dx = grp.pacmap_x_nesi - cx
        dy = grp.pacmap_y_nesi - cy
        selected.append(grp.iloc[int(np.argmin(dx * dx + dy * dy))])
    sel = pd.DataFrame(selected)
    if len(sel) > target_n:
        sel = sel.sample(n=target_n, random_state=42)
    return sel


def main():
    df = pd.read_csv(COORDS_CSV)
    df = df.assign(stem=df.MorgothOutputFilename.str.replace(
        r"\.csv$", "", regex=True))

    clickable = df[df.Dataset.isin(['CAMS', 'ICANS', 'RASS'])].copy()
    print(f"clickable rows: {len(clickable)}")

    # Pool = unique (Dataset, stem) with PNG on disk.
    pool = clickable.drop_duplicates(['Dataset', 'stem']).copy()
    pool = pool[pool.apply(
        lambda r: (PNG_DIR / r.Dataset / (r.stem + '.png')).exists(), axis=1)]
    print(f"unique PNGs on disk: {len(pool)}")

    # All clickable (background); GCS (for context, light grey)
    gcs = df[df.Dataset == 'GCS']

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(20, 5.5), dpi=130)
    for ax, n in zip(axes, TARGETS):
        sel = subsample(pool, target_n=n)
        # Background: all clickable as small pale dots
        ax.scatter(gcs.pacmap_x_nesi, gcs.pacmap_y_nesi,
                    s=0.6, c='#dddddd', alpha=0.35, linewidths=0,
                    rasterized=True)
        ax.scatter(clickable.pacmap_x_nesi, clickable.pacmap_y_nesi,
                    s=1.0, c='#bcdff1', alpha=0.5, linewidths=0,
                    rasterized=True)
        # Foreground: the subsample
        ax.scatter(sel.pacmap_x_nesi, sel.pacmap_y_nesi,
                    s=10, c='#d4071e', alpha=0.85, linewidths=0,
                    rasterized=True)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        mb = len(sel) * PNG_BYTES_AVG / 1e6
        ax.set_title(
            f"target-N = {n}   →   {len(sel)} selected   "
            f"({mb:.0f} MB)",
            fontsize=11,
        )

    fig.suptitle(
        "PaCMAP coverage by subsample size  •  "
        "pale blue = all 21,290 clickable points  •  "
        "grey = 22,540 GCS points (no EEG)  •  "
        "red = subsample",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
