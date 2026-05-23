"""
Run PCA + (t-SNE) + UMAP + PaCMAP on any features CSV, save a single PNG.

Usage:
    python3 embedding_preview_cli.py INPUT_CSV OUTPUT_PNG [--skip-tsne]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pacmap

SCALE_MARKER = {'GCS': 'o', 'RASS': 's', 'CAMS': '^', 'ICANS': 'D'}
FEATURE_COLS = [
    'Awake', 'N1', 'N2',
    'Normal_vs_Abnormal',
    'Burst_vs_NoBurst',
    'NoSpike', 'FocalSpike', 'GenSpike',
    'NoSlowing', 'FocalSlowing', 'GenSlowing',
    'IIIC_Other', 'IIIC_Seizure', 'IIIC_LPD', 'IIIC_GPD',
    'IIIC_LRDA', 'IIIC_GRDA',
]


def scatter(ax, XY, df, title, point_size=6):
    norm = Normalize(vmin=-3, vmax=3)
    sc = None
    for scale, marker in SCALE_MARKER.items():
        m = (df.Dataset == scale).to_numpy()
        if not m.any():
            continue
        sc = ax.scatter(
            XY[m, 0], XY[m, 1],
            c=df.NESI[m], cmap='coolwarm', norm=norm,
            s=point_size, marker=marker, edgecolors='none', alpha=0.7,
            label=f"{scale} (n={int(m.sum())})",
        )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc='best', framealpha=0.7)
    return sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("output_png")
    p.add_argument("--skip-tsne", action="store_true")
    p.add_argument("--point-size", type=float, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    n = len(df)
    print(f"Loaded {n} rows from {args.input_csv}")
    X = df[FEATURE_COLS].to_numpy()
    ps = args.point_size if args.point_size is not None else max(1.5, 60 / np.sqrt(n))

    embeddings = {}

    t = time.time()
    print("Computing PCA...")
    pca = PCA(n_components=2, svd_solver='full', random_state=0).fit(X)
    embeddings['(a) PCA'] = pca.transform(X)
    print(f"  done in {time.time()-t:.1f}s "
          f"(PC1={pca.explained_variance_ratio_[0]:.0%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.0%})")

    if not args.skip_tsne:
        t = time.time()
        print(f"Running t-SNE (n={n}, may be slow)...")
        embeddings['(b) t-SNE'] = TSNE(
            n_components=2, perplexity=30, init='pca',
            random_state=0, max_iter=1000, n_jobs=-1,
        ).fit_transform(X)
        print(f"  done in {time.time()-t:.1f}s")
    else:
        print("Skipping t-SNE")

    t = time.time()
    print(f"Running UMAP (n={n})...")
    embeddings['(c) UMAP'] = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        random_state=0, n_jobs=-1,
    ).fit_transform(X)
    print(f"  done in {time.time()-t:.1f}s")

    t = time.time()
    print(f"Running PaCMAP (n={n})...")
    embeddings['(d) PaCMAP'] = pacmap.PaCMAP(
        n_components=2, n_neighbors=10,
        MN_ratio=0.5, FP_ratio=2.0, random_state=0,
    ).fit_transform(X, init='random')
    print(f"  done in {time.time()-t:.1f}s")

    nplots = len(embeddings)
    nrows = 2 if nplots == 4 else 1
    ncols = 2 if nplots == 4 else nplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows),
                              dpi=160)
    axes_flat = np.array(axes).reshape(-1)
    sc_last = None
    for ax, (title, XY) in zip(axes_flat, embeddings.items()):
        sc_last = scatter(ax, XY, df, title, point_size=ps)
        if title == '(a) PCA':
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", fontsize=9)
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontsize=9)

    cbar = fig.colorbar(sc_last, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("NESI", fontsize=10)

    fig.suptitle(f"MORGOTH-feature embeddings, n={n} (colored by NESI)",
                 fontsize=12, fontweight='bold', y=0.995)
    fig.savefig(args.output_png, dpi=180, bbox_inches='tight')
    print(f"Saved {args.output_png}")


if __name__ == '__main__':
    main()
