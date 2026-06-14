# build_cams_asm_antipsych_exposure.py
"""
Per-EEG ASM and antipsychotic exposure flags for the I0001 CAMS cohort.

For patients with NO medication administration records in
I0001_CAMS_COHORT_MEDS_PARQUET, exposure is set to NA (not 0).

Uses Snippet_StartDTS as the EEG timestamp column and the 24h-prior window.

Paths read from med_config.py.
"""

import duckdb
import pandas as pd

import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


EEG_TIMESTAMP_COL = "Snippet_StartDTS"
WINDOW_HOURS = 24

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


def get_patients_with_med_data(cohort_meds_parquet):
    con = duckdb.connect()
    parquet_str = str(cohort_meds_parquet).replace("\\", "/")
    df = con.execute(f"""
        SELECT DISTINCT CAST(BDSPPatientID AS VARCHAR) AS pid
        FROM '{parquet_str}'
    """).df()
    con.close()
    return set(norm_id(df["pid"]).dropna())


def apply_na_mask(eeg_df, exposure_series, patients_with_med_data):
    has_data = eeg_df["BDSPPatientID"].isin(patients_with_med_data)
    result = exposure_series.astype("Int64")
    result[~has_data] = pd.NA
    return result


def compute_category_exposure(eeg_df, category_name, patients_with_med_data):
    admin_path = med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR / f"{category_name}_administrations.csv"
    if not admin_path.exists():
        print(f"  WARNING: {admin_path.name} missing, all 0s")
        series = pd.Series(0, index=eeg_df.index, dtype="Int64")
        return apply_na_mask(eeg_df, series, patients_with_med_data)

    con = duckdb.connect()
    eeg_slice = eeg_df[["_row_id", "BDSPPatientID", "_eeg_ts", "_window_start"]].copy()
    eeg_slice["BDSPPatientID"] = eeg_slice["BDSPPatientID"].astype(str)
    con.register("eegs", eeg_slice)

    admin_path_str = str(admin_path).replace("\\", "/")
    result = con.execute(f"""
        WITH adm AS (
          SELECT CAST(BDSPPatientID AS VARCHAR) AS BDSPPatientID,
                 CAST(MedicationTakenDTS AS TIMESTAMP) AS _med_ts
          FROM read_csv_auto('{admin_path_str}')
          WHERE BDSPPatientID IS NOT NULL AND MedicationTakenDTS IS NOT NULL
        )
        SELECT e._row_id,
               MAX(CASE WHEN adm._med_ts BETWEEN e._window_start AND e._eeg_ts THEN 1 ELSE 0 END) AS exposure
        FROM eegs e
        LEFT JOIN adm ON e.BDSPPatientID = adm.BDSPPatientID
                     AND adm._med_ts >= e._window_start
                     AND adm._med_ts <= e._eeg_ts
        GROUP BY e._row_id
    """).df()
    con.close()
    exposure_map = dict(zip(result["_row_id"], result["exposure"]))
    series = eeg_df["_row_id"].map(exposure_map).fillna(0).astype(int)
    return apply_na_mask(eeg_df, series, patients_with_med_data)


def main():
    try:
        patients_with_med_data = get_patients_with_med_data(
            med_config.I0001_CAMS_COHORT_MEDS_PARQUET
        )
        print(f"Patients with med data: {len(patients_with_med_data):,}")

        pieces = []
        for f in med_config.I0001_CAMS_METADATA_CSVS:
            pieces.append(pd.read_csv(str(f), low_memory=False))
        meta = pd.concat(pieces, ignore_index=True)
        print(f"Total CAMS EEG rows: {len(meta):,}")

        if EEG_TIMESTAMP_COL not in meta.columns:
            raise ValueError(f"Column '{EEG_TIMESTAMP_COL}' not found.")

        meta["_row_id"] = range(len(meta))
        meta["BDSPPatientID"] = norm_id(meta["BDSPPatientID"])
        meta["_eeg_ts"] = pd.to_datetime(meta[EEG_TIMESTAMP_COL], errors="coerce")
        meta["_window_start"] = meta["_eeg_ts"] - pd.Timedelta(hours=WINDOW_HOURS)

        for col_name, category in CATEGORIES:
            print(f"\nComputing {col_name}...")
            meta[col_name] = compute_category_exposure(meta, category, patients_with_med_data)
            n_e = int((meta[col_name] == 1).sum())
            n_p = meta.loc[meta[col_name] == 1, "BDSPPatientID"].nunique()
            n_na = int(meta[col_name].isna().sum())
            print(f"  EEGs exposed:    {n_e:,}")
            print(f"  Patients exposed: {n_p:,}")
            print(f"  EEGs with NA (no med data): {n_na:,}")

        out = meta.drop(columns=["_row_id", "_eeg_ts", "_window_start"])
        out_path = med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR / "i0001_cams_per_eeg_asm_antipsych_exposure.csv"
        out.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

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


if __name__ == "__main__":
    main()