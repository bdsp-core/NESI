# build_i0001_asm_antipsych_exposure.py
"""
Build per-EEG ASM and antipsychotic exposure flags for the I0001 cohort.

Loads GCS and RASS metadata separately (different timestamp columns:
GCSRecordedDTS vs RASSRecordedDTS), computes 24h-prior-window exposure for
each, and concatenates into a combined output file.

For each EEG row, flags:
  LongActingASM_Exposure
  Antipsychotic_Exposure

Reads per-category admin CSVs from extract_i0001_asm_antipsych_administrations.py.

Writes to I0001_ASM_ANTIPSYCH_DIR/i0001_per_eeg_asm_antipsych_exposure.csv.

Paths read from med_config.py.
"""

import glob
import duckdb

import pandas as pd

import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


WINDOW_HOURS = 24

# Per-sub-cohort config: which metadata files, which timestamp column,
# and the cohort label to tag in the output.
SUB_COHORTS = [
    {
        "label": "GCS",
        "metadata_files": med_config.I0001_GCS_METADATA_CSVS,
        "ts_col": "GCSRecordedDTS",
    },
    {
        "label": "RASS",
        "metadata_files": med_config.I0001_RASS_METADATA_CSVS,
        "ts_col": "RASSRecordedDTS",
    },
]

CATEGORIES = [
    ("LongActingASM_Exposure", "long_acting_asm"),
    ("Antipsychotic_Exposure", "antipsychotic"),
]


def norm_id(s):
    return (
        s.astype(str).str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .replace({"nan": None, "None": None, "": None})
    )

def load_metadata_files(paths_or_glob, cohort_label, ts_col):
    """
    Read all metadata files (either a list of paths or a glob pattern),
    concatenate, normalize patient IDs, and add timestamp / window-start
    columns.
    """
    # Accept either a list/tuple of paths or a single glob pattern string
    if isinstance(paths_or_glob, (list, tuple)):
        files = [str(p) for p in paths_or_glob]
    else:
        files = sorted(glob.glob(str(paths_or_glob)))
    if not files:
        raise FileNotFoundError(f"No files matched: {paths_or_glob}")
    frames = []
    for f in files:
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    if ts_col not in df.columns:
        raise ValueError(
            f"{cohort_label}: column '{ts_col}' not found. "
            f"Available: {df.columns.tolist()[:20]}..."
        )

    df["BDSPPatientID"] = norm_id(df["BDSPPatientID"])
    df["_eeg_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["_window_start"] = df["_eeg_ts"] - pd.Timedelta(hours=WINDOW_HOURS)
    df["_cohort"] = cohort_label

    return df


def compute_category_exposure(eeg_df, category_name):
    """
    For each EEG row, return 1 if the patient had any admin of any drug in
    this category in [eeg_ts - 24h, eeg_ts], else 0. Uses DuckDB to push the
    date-window filter into the join so we don't materialize the cartesian
    product in pandas (which OOMs at I0001 scale).
    """
    admin_path = med_config.I0001_ASM_ANTIPSYCH_DIR / f"{category_name}_administrations.csv"
    if not admin_path.exists():
        print(f"  WARNING: {admin_path.name} missing, all 0s")
        return pd.Series(0, index=eeg_df.index, dtype=int)

    con = duckdb.connect()

    # Register the EEG slice as a view
    eeg_slice = eeg_df[["_row_id", "BDSPPatientID", "_eeg_ts", "_window_start"]].copy()
    eeg_slice["BDSPPatientID"] = eeg_slice["BDSPPatientID"].astype(str)
    con.register("eegs", eeg_slice)

    admin_path_str = str(admin_path).replace("\\", "/")

    result = con.execute(f"""
        WITH adm AS (
          SELECT
            CAST(BDSPPatientID AS VARCHAR) AS BDSPPatientID,
            CAST(MedicationTakenDTS AS TIMESTAMP) AS _med_ts
          FROM read_csv_auto('{admin_path_str}')
          WHERE BDSPPatientID IS NOT NULL
            AND MedicationTakenDTS IS NOT NULL
        )
        SELECT
          e._row_id,
          MAX(CASE WHEN adm._med_ts BETWEEN e._window_start AND e._eeg_ts THEN 1 ELSE 0 END) AS exposure
        FROM eegs e
        LEFT JOIN adm
          ON e.BDSPPatientID = adm.BDSPPatientID
         AND adm._med_ts >= e._window_start
         AND adm._med_ts <= e._eeg_ts
        GROUP BY e._row_id
    """).df()

    con.close()

    # Build the per-EEG series aligned to eeg_df.index
    exposure_map = dict(zip(result["_row_id"], result["exposure"]))
    return eeg_df["_row_id"].map(exposure_map).fillna(0).astype(int)


try:
    all_pieces = []

    for sc in SUB_COHORTS:
        print(f"\n=== {sc['label']} sub-cohort ===")
        meta = load_metadata_files(sc["metadata_files"], sc["label"], sc["ts_col"])
        print(f"  Total EEG rows: {len(meta):,}")
        meta["_row_id"] = range(len(meta))

        n_invalid_pt = meta["BDSPPatientID"].isna().sum()
        n_invalid_ts = meta["_eeg_ts"].isna().sum()
        print(f"  Rows with missing patient ID:    {n_invalid_pt:,}")
        print(f"  Rows with missing EEG timestamp: {n_invalid_ts:,}")

        for col_name, category in CATEGORIES:
            print(f"  Computing {col_name} ({category})...")
            meta[col_name] = compute_category_exposure(meta, category)
            n_exposed_eegs = int(meta[col_name].sum())
            n_pts_exposed = meta.loc[meta[col_name] == 1, "BDSPPatientID"].nunique()
            print(f"    EEGs exposed:    {n_exposed_eegs:,}")
            print(f"    Patients exposed:{n_pts_exposed:,}")

        all_pieces.append(meta)

    # Concatenate GCS + RASS
    combined = pd.concat(all_pieces, ignore_index=True, sort=False)

    # Drop internal cols
    out = combined.drop(columns=["_row_id", "_eeg_ts", "_window_start"])

    out_path = med_config.I0001_ASM_ANTIPSYCH_DIR / "i0001_per_eeg_asm_antipsych_exposure.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved combined output: {out_path}")
    print(f"  Total rows: {len(out):,}")
    print(f"  Total patients: {combined['BDSPPatientID'].nunique():,}")
    for col_name, _ in CATEGORIES:
        n_total = int(combined[col_name].sum())
        n_pts = combined.loc[combined[col_name] == 1, "BDSPPatientID"].nunique()
        print(f"  {col_name:<26}  EEGs: {n_total:>6,}   patients: {n_pts:>5,}")

    print("\nDone.")
    if HAS_WINSOUND:
        winsound.MessageBeep(winsound.MB_OK)
    else:
        try:
            print("\a")
        except Exception:
            pass

except Exception as e:
    print(f"\nERROR: {e}")
    if HAS_WINSOUND:
        winsound.MessageBeep(winsound.MB_ICONHAND)
    else:
        try:
            print("\a")
        except Exception:
            pass
    raise