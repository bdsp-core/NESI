"""
Per-patient aggregation of NESI_full_features.csv.

Each output row is one patient (extracted from MorgothOutputFilename), with
median across the patient's snippets for every numeric column. This removes
within-patient redundancy that otherwise dominates density in non-linear
embeddings (one GCS patient contributes up to 610 snippets in the raw data).
"""

from pathlib import Path
import re

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
IN_CSV = SCRIPT_DIR / "NESI_full_features.csv"
OUT_CSV = SCRIPT_DIR / "NESI_full_features_perpatient.csv"


def extract_pid(fn: str) -> str:
    """Patient ID from MorgothOutputFilename.

    GCS / RASS / CAMS examples use "sub-XXXXX_...".
    ICANS examples use "<digits>_seg2_...".
    """
    m = re.match(r"^(sub-[A-Za-z0-9]+)", str(fn))
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)_", str(fn))
    if m:
        return f"icans-{m.group(1)}"
    return str(fn).split("_")[0]


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Input: {len(df)} snippets")

    df['PatientID'] = df['MorgothOutputFilename'].apply(extract_pid)

    feature_cols = [c for c in df.columns if c not in
                    ('MorgothOutputFilename', 'Dataset', 'WhichSet',
                     'PatientID')]
    # NOTE: TrueRawScores, TransformedRawScores, and NESI go through median too.

    agg = (df.groupby(['PatientID', 'Dataset'], sort=False)[feature_cols]
             .median()
             .reset_index())

    # Snippet counts per patient
    counts = df.groupby('PatientID').size().rename('NSnippets').reset_index()
    agg = agg.merge(counts, on='PatientID')

    print(f"Output: {len(agg)} patients")
    print("\nCounts per scale:")
    print(agg.Dataset.value_counts())

    agg.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == '__main__':
    main()
