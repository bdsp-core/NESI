"""
Build a window-level feature CSV (no per-snippet median).

For each patient:
  1. Randomly pick up to MAX_SNIPPETS_PER_PATIENT of their snippets.
  2. For each picked snippet, load the six MORGOTH head CSVs, concat
     column-wise to (n_windows x 17), then drop the 3 spike-localization
     columns and IIIC_Other -> (n_windows x 13).
  3. Pool all window-rows across the patient's picked snippets, then
     randomly sample MAX_VECTORS_PER_PATIENT of them.

Each output row is one window-vector with its NESI, raw score, dataset, and
source snippet attached.

The keep-13 feature set per the latest revision:
    Awake, N1, N2, Normal_vs_Abnormal, Burst_vs_NoBurst,
    NoSlowing, FocalSlowing, GenSlowing,
    IIIC_Seizure, IIIC_LPD, IIIC_GPD, IIIC_LRDA, IIIC_GRDA
"""

from __future__ import annotations

from pathlib import Path
import multiprocessing as mp
import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
YAMA_NESI = SCRIPT_DIR.parent                 # YAMA/NESI/
YAMA_ROOT = YAMA_NESI.parent                  # YAMA/
META_CSV = (YAMA_NESI / "MORGOTHActivationViz_GroupedbyNESI"
            / "UniversalBadnessModelResult_Full.csv")
OUT_CSV = SCRIPT_DIR / "NESI_window_features.csv"

MAX_SNIPPETS_PER_PATIENT = 10
MAX_VECTORS_PER_PATIENT = 10
SEED = 42
N_WORKERS = max(1, os.cpu_count() - 2)

HEADS = ['SLEEP', 'NM', 'BS', 'FOCGEN', 'SLOWING', 'IIIC']
# Concatenated column order matches morgoth_10minfea_matrix(): SLEEP, NM, BS,
# FOCGEN, SLOWING, IIIC.
ALL_FEATURE_NAMES = [
    'Awake', 'N1', 'N2',
    'Normal_vs_Abnormal',
    'Burst_vs_NoBurst',
    'NoSpike', 'FocalSpike', 'GenSpike',
    'NoSlowing', 'FocalSlowing', 'GenSlowing',
    'IIIC_Other', 'IIIC_Seizure', 'IIIC_LPD', 'IIIC_GPD',
    'IIIC_LRDA', 'IIIC_GRDA',
]
DROP_FEATURES = set()  # Keep all 17 cols; downstream scripts select subset
KEEP_FEATURES = [c for c in ALL_FEATURE_NAMES if c not in DROP_FEATURES]
KEEP_IDX = np.array([i for i, n in enumerate(ALL_FEATURE_NAMES)
                     if n not in DROP_FEATURES])


def extract_pid(fn: str) -> str:
    m = re.match(r"^(sub-[A-Za-z0-9]+)", str(fn))
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)_", str(fn))
    if m:
        return f"icans-{m.group(1)}"
    return str(fn).split("_")[0]


def head_dir(dataset: str, head: str) -> Path:
    return YAMA_ROOT / dataset / "MorgothActivations" / head


def load_head(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "pred_class" in df.columns:
        df = df.drop(columns=["pred_class"])
    return df.values


def load_snippet_matrix(dataset: str, fname: str) -> np.ndarray | None:
    """(n_windows x 13) for one snippet, or None on missing files."""
    pieces = []
    for head in HEADS:
        p = head_dir(dataset, head) / fname
        if not p.exists():
            return None
        pieces.append(load_head(p))
    stacked = np.concatenate(pieces, axis=1)  # (n_windows, 17)
    return stacked[:, KEEP_IDX]                # (n_windows, 13)


def process_patient(args):
    """Returns list of dicts (one per kept window) for this patient."""
    pid, group_rows, seed = args
    rng = np.random.default_rng(seed)

    # Step A: cap snippet count per patient
    if len(group_rows) > MAX_SNIPPETS_PER_PATIENT:
        idx = rng.choice(len(group_rows), MAX_SNIPPETS_PER_PATIENT, replace=False)
        group_rows = [group_rows[i] for i in idx]

    rows_out = []  # list of (window_vec, metadata_dict)
    for meta in group_rows:
        mat = load_snippet_matrix(meta['Dataset'], meta['MorgothOutputFilename'])
        if mat is None:
            continue
        for i in range(mat.shape[0]):
            rows_out.append((mat[i], meta))

    if not rows_out:
        return []

    # Step B: cap window-vectors per patient
    if len(rows_out) > MAX_VECTORS_PER_PATIENT:
        idx = rng.choice(len(rows_out), MAX_VECTORS_PER_PATIENT, replace=False)
        rows_out = [rows_out[i] for i in idx]

    out = []
    for vec, meta in rows_out:
        d = {
            'PatientID': pid,
            'Dataset': meta['Dataset'],
            'MorgothOutputFilename': meta['MorgothOutputFilename'],
            'TrueRawScores': meta['TrueRawScores'],
            'NESI': meta['NESI'],
            'WhichSet': meta['WhichSet'],
        }
        for name, v in zip(KEEP_FEATURES, vec):
            d[name] = v
        out.append(d)
    return out


def main():
    if not META_CSV.exists():
        sys.exit(f"Missing metadata: {META_CSV}")

    meta = pd.read_csv(META_CSV)
    meta['PatientID'] = meta['MorgothOutputFilename'].apply(extract_pid)
    print(f"Loaded {len(meta)} snippet rows, {meta.PatientID.nunique()} unique patients")

    # Pre-bucket meta rows by patient id (cross-dataset patient stays one bucket)
    per_pat = meta.groupby('PatientID', sort=False)
    buckets = []
    rng = np.random.default_rng(SEED)
    for pid, sub in per_pat:
        group_rows = sub.to_dict('records')
        seed = int(rng.integers(1 << 31))
        buckets.append((pid, group_rows, seed))
    print(f"Built {len(buckets)} patient buckets")

    all_rows = []
    with mp.Pool(N_WORKERS) as pool:
        for rows in tqdm(
            pool.imap_unordered(process_patient, buckets, chunksize=32),
            total=len(buckets), desc="patients",
        ):
            all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    # Stable column order
    front = ['PatientID', 'Dataset', 'MorgothOutputFilename',
             'TrueRawScores', 'NESI', 'WhichSet']
    out = out[front + KEEP_FEATURES]
    print(f"Output shape: {out.shape}")
    print("Vectors per patient: mean=%.2f median=%d max=%d"
          % (out.groupby('PatientID').size().mean(),
             out.groupby('PatientID').size().median(),
             out.groupby('PatientID').size().max()))
    print("Per scale:", out.Dataset.value_counts().to_dict())
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == '__main__':
    main()
