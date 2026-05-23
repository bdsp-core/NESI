"""
Build a single tidy CSV with one row per case for the balanced ~4367-case
subset used in MORGOTHActivationViz_GroupedbyNESI/. Each row contains:

    Dataset, TrueRawScores, NESI, MorgothOutputFilename, NESI_Bin,
    <17 MORGOTH median feature columns>

The MORGOTH features come from the X matrices stored in
NESIbin{1,2,3}_data.pkl (median over each case's 10-min snippets).
NESI per case is recovered by reproducing Arka's seed=42 balanced sample —
which is byte-identical to the pickled arrays (verified by exact match on
TrueRawScores and Dataset).
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
YAMA_DIR = SCRIPT_DIR.parent / "MORGOTHActivationViz_GroupedbyNESI"
OUT_PATH = SCRIPT_DIR / "NESI_balanced_features.csv"

NESI_BINS = [-3, -1, 1, 3]
NESI_BIN_LABELS = ['[-3, -1)', '[-1, 1)', '[1, 3]']

FEATURE_NAMES = [
    'Awake', 'N1', 'N2',
    'Normal_vs_Abnormal',
    'Burst_vs_NoBurst',
    'NoSpike', 'FocalSpike', 'GenSpike',
    'NoSlowing', 'FocalSlowing', 'GenSlowing',
    'IIIC_Other', 'IIIC_Seizure', 'IIIC_LPD', 'IIIC_GPD',
    'IIIC_LRDA', 'IIIC_GRDA',
]


def create_balanced_bin_df(df_bin, target_total_large=500, min_per_class=10,
                            random_seed=42):
    parts = []
    for dataset in ['GCS', 'RASS', 'CAMS', 'ICANS']:
        sub = df_bin[df_bin['Dataset'] == dataset].copy()
        if len(sub) == 0:
            continue
        if dataset in ['GCS', 'RASS']:
            class_counts = sub['TrueRawScores'].value_counts().sort_index()
            class_props = class_counts / class_counts.sum()
            raw_alloc = (class_props * target_total_large).round().astype(int)
            alloc = raw_alloc.clip(lower=min_per_class)
            alloc = alloc.combine(class_counts, min)
            sp = []
            for score, n in alloc.items():
                cls_df = sub[sub['TrueRawScores'] == score]
                sp.append(cls_df.sample(n=int(n), random_state=random_seed))
            parts.append(pd.concat(sp).reset_index(drop=True))
        else:
            parts.append(sub)
    return pd.concat(parts).reset_index(drop=True)


def main():
    df = pd.read_csv(YAMA_DIR / "UniversalBadnessModelResult_Full.csv")
    df['NESI_Bin'] = pd.cut(df['NESI'], bins=NESI_BINS,
                             labels=NESI_BIN_LABELS, include_lowest=True)

    rows = []
    for i, label in enumerate(NESI_BIN_LABELS, start=1):
        df_bin = df[df.NESI_Bin == label].copy().reset_index(drop=True)
        bal = create_balanced_bin_df(df_bin)

        with open(YAMA_DIR / f"NESIbin{i}_data.pkl", "rb") as f:
            d = pickle.load(f)
        X = d[f"X_bin{i}"]
        Y = d[f"Y_raw_bin{i}"]
        names = d[f"dataset_names_bin{i}"]

        assert np.array_equal(bal['TrueRawScores'].to_numpy(), Y), \
            f"bin{i}: TrueRawScores mismatch — random_state reproduction broke"
        assert np.array_equal(bal['Dataset'].to_numpy(), names), \
            f"bin{i}: Dataset mismatch"

        meta = bal[['Dataset', 'TrueRawScores', 'NESI',
                    'MorgothOutputFilename']].copy()
        meta['NESI_Bin'] = label
        feat = pd.DataFrame(X, columns=FEATURE_NAMES)
        rows.append(pd.concat([meta.reset_index(drop=True),
                               feat.reset_index(drop=True)], axis=1))

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows x {out.shape[1]} cols -> {OUT_PATH}")
    print("\nDataset counts:", out.Dataset.value_counts().to_dict())
    print("\nFirst row:")
    print(out.iloc[0])


if __name__ == '__main__':
    main()
