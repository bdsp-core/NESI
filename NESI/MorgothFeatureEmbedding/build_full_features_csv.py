"""
Build a tidy "one row per case" CSV for the FULL ~210k-snippet dataset.

For every row of UniversalBadnessModelResult_Full.csv (NESI + raw score per
case), this script loads the case's MORGOTH activation CSVs from each of the
six heads (SLEEP, NM, BS, FOCGEN, SLOWING, IIIC), takes the median over rows
(matching morgoth_10minfea_matrix() in FAST_NESI_group_morgothactivation.py),
and writes a 17-d feature vector + metadata to a single output CSV.

Run only after the GCS and RASS MorgothActivations have been synced from
s3://bdsp-opendata-credentialed/yama/ (CAMS and ICANS were already local).

Rows whose activation files are missing are skipped and logged to
NESI_full_features_missing.csv so you can audit coverage.

Uses a process pool for speed (set N_WORKERS=1 to debug single-threaded).
"""

from __future__ import annotations

from pathlib import Path
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
YAMA_NESI = SCRIPT_DIR.parent                 # YAMA/NESI/
YAMA_ROOT = YAMA_NESI.parent                  # YAMA/
META_CSV = (YAMA_NESI / "MORGOTHActivationViz_GroupedbyNESI"
            / "UniversalBadnessModelResult_Full.csv")
OUT_CSV = SCRIPT_DIR / "NESI_full_features.csv"
MISSING_CSV = SCRIPT_DIR / "NESI_full_features_missing.csv"

N_WORKERS = max(1, os.cpu_count() - 2)

HEADS = ['SLEEP', 'NM', 'BS', 'FOCGEN', 'SLOWING', 'IIIC']
FEATURE_NAMES = [
    'Awake', 'N1', 'N2',
    'Normal_vs_Abnormal',
    'Burst_vs_NoBurst',
    'NoSpike', 'FocalSpike', 'GenSpike',
    'NoSlowing', 'FocalSlowing', 'GenSlowing',
    'IIIC_Other', 'IIIC_Seizure', 'IIIC_LPD', 'IIIC_GPD',
    'IIIC_LRDA', 'IIIC_GRDA',
]


def head_dir(dataset: str, head: str) -> Path:
    return YAMA_ROOT / dataset / "MorgothActivations" / head


def load_feature(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "pred_class" in df.columns:
        df = df.drop(columns=["pred_class"])
    return df.values


def process_row(args) -> tuple[int, np.ndarray | None, str | None]:
    """Returns (row_index, median_features_or_None, missing_head_or_None)."""
    idx, dataset, fname = args
    pieces = []
    for head in HEADS:
        arr = load_feature(head_dir(dataset, head) / fname)
        if arr is None:
            return idx, None, head
        pieces.append(arr)
    stacked = np.concatenate(pieces, axis=1)
    return idx, np.median(stacked, axis=0), None


def main():
    if not META_CSV.exists():
        sys.exit(f"Missing metadata CSV: {META_CSV}")

    meta = pd.read_csv(META_CSV)
    print(f"Loaded {len(meta)} rows from {META_CSV}")

    # Quick coverage probe so we fail fast if a scale's data isn't synced
    for ds in ['GCS', 'RASS', 'CAMS', 'ICANS']:
        head0 = head_dir(ds, HEADS[0])
        n_files = sum(1 for _ in head0.glob("*.csv")) if head0.exists() else 0
        n_meta = (meta.Dataset == ds).sum()
        print(f"  {ds}: meta rows={n_meta}, local SLEEP files={n_files}")

    work = [(i, row.Dataset, row.MorgothOutputFilename)
            for i, row in meta.iterrows()]

    features = np.full((len(meta), len(FEATURE_NAMES)), np.nan)
    missing_rows = []

    with mp.Pool(N_WORKERS) as pool:
        for idx, vec, missing in tqdm(
            pool.imap_unordered(process_row, work, chunksize=200),
            total=len(work), desc="aggregate",
        ):
            if vec is None:
                missing_rows.append((idx, missing))
            else:
                features[idx] = vec

    out = meta.copy()
    feat_df = pd.DataFrame(features, columns=FEATURE_NAMES,
                            index=meta.index)
    out = pd.concat([out, feat_df], axis=1)

    keep = ~out[FEATURE_NAMES[0]].isna()
    print(f"\nRows with complete features: {keep.sum()} / {len(out)} "
          f"({100*keep.mean():.1f}%)")

    out[keep].to_csv(OUT_CSV, index=False)
    print(f"Wrote {keep.sum()} rows x {out.shape[1]} cols -> {OUT_CSV}")

    if missing_rows:
        miss = pd.DataFrame(missing_rows, columns=['row_index', 'first_missing_head'])
        miss = miss.merge(meta[['Dataset', 'MorgothOutputFilename']],
                          left_on='row_index', right_index=True)
        miss.to_csv(MISSING_CSV, index=False)
        print(f"Logged {len(miss)} missing rows -> {MISSING_CSV}")
        print("  by scale:", miss.Dataset.value_counts().to_dict())


if __name__ == '__main__':
    main()
