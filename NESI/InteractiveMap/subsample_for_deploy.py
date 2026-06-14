"""
Pick spatially-uniform PNGs from eeg_pngs/ and copy them into
deploy/eeg_pngs/ for hosting on GitHub Pages.

Algorithm: grid the PaCMAP layout into NxN cells; pick one available
PNG per cell (the row closest to that cell's centroid). If the result
exceeds --target-n, randomly subsample to that size.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
COORDS_CSV = (SCRIPT_DIR.parent / "MorgothFeatureEmbedding"
              / "NESI_pacmap_coords.csv")
PNG_DIR = SCRIPT_DIR / "eeg_pngs"
DEPLOY_DIR = SCRIPT_DIR / "deploy"
DEPLOY_PNG_DIR = DEPLOY_DIR / "eeg_pngs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-n", type=int, default=1000)
    ap.add_argument("--grid", type=int, default=36,
                    help="grid resolution (NxN cells)")
    ap.add_argument("--reset", action="store_true",
                    help="wipe deploy/eeg_pngs before copying")
    args = ap.parse_args()

    df = pd.read_csv(COORDS_CSV)
    df = df.assign(stem=df.MorgothOutputFilename.str.replace(
        r"\.csv$", "", regex=True))
    df = df[df.Dataset.isin(['CAMS', 'ICANS', 'RASS'])].copy()

    # Unique (Dataset, stem) with a PNG on disk; keep first row per pair.
    df = df.drop_duplicates(['Dataset', 'stem'])
    have = []
    for r in df.itertuples(index=False):
        if (PNG_DIR / r.Dataset / (r.stem + '.png')).exists():
            have.append(r)
    if not have:
        raise SystemExit(f"No PNGs found under {PNG_DIR}")
    pool = pd.DataFrame(have)
    print(f"unique PNGs on disk:              {len(pool)}")

    # Grid-cell assignment in PaCMAP coords.
    xs = pool.pacmap_x_nesi.to_numpy()
    ys = pool.pacmap_y_nesi.to_numpy()
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xc = np.clip(((xs - xmin) / (xmax - xmin + 1e-9) *
                   args.grid).astype(int), 0, args.grid - 1)
    yc = np.clip(((ys - ymin) / (ymax - ymin + 1e-9) *
                   args.grid).astype(int), 0, args.grid - 1)
    pool = pool.assign(cell=xc * args.grid + yc)

    # For each occupied cell, pick the row closest to that cell's centroid.
    selected = []
    cell_w = (xmax - xmin) / args.grid
    cell_h = (ymax - ymin) / args.grid
    for cell_id, grp in pool.groupby('cell'):
        ix = cell_id // args.grid
        iy = cell_id %  args.grid
        cx = xmin + (ix + 0.5) * cell_w
        cy = ymin + (iy + 0.5) * cell_h
        dx = grp.pacmap_x_nesi - cx
        dy = grp.pacmap_y_nesi - cy
        best = grp.iloc[int(np.argmin(dx * dx + dy * dy))]
        selected.append(best)
    sel = pd.DataFrame(selected)
    print(f"after grid pick (one per cell):   {len(sel)}")

    if len(sel) > args.target_n:
        sel = sel.sample(n=args.target_n, random_state=42)
        print(f"after random downsample:          {len(sel)}")

    # Per-dataset summary
    print("\nselected per dataset:")
    print(sel.Dataset.value_counts().to_string())

    if args.reset and DEPLOY_PNG_DIR.exists():
        shutil.rmtree(DEPLOY_PNG_DIR)
    DEPLOY_PNG_DIR.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for r in sel.itertuples(index=False):
        src = PNG_DIR / r.Dataset / (r.stem + '.png')
        dst = DEPLOY_PNG_DIR / r.Dataset / (r.stem + '.png')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total_bytes += dst.stat().st_size

    print(f"\nCopied {len(sel)} PNGs ({total_bytes / 1e6:.1f} MB) "
          f"-> {DEPLOY_PNG_DIR}")


if __name__ == "__main__":
    main()
