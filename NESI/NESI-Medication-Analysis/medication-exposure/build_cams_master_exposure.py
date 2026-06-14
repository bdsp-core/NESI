# build_cams_master_exposure.py
"""
Master per-EEG exposure file for the I0001 CAMS cohort.

For patients with NO medication administration records in
I0001_CAMS_COHORT_MEDS_PARQUET, exposure columns are set to NA (not 0).

Anchor timestamp: Snippet_StartDTS. Window: 24h prior.

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


WINDOW_HOURS = 24
EEG_TIMESTAMP_COL = "Snippet_StartDTS"

BENZO_DRUGS = ["lorazepam", "diazepam", "other_benzo"]
OPIATE_DRUGS = ["fentanyl", "morphine", "hydromorphone"]
SEDATIVE_INFUSION_DRUGS = ["propofol", "midazolam", "ketamine", "dexmedetomidine"]
OPIATE_INFUSION_DRUGS = ["fentanyl", "morphine", "hydromorphone"]


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


def _exposure_via_duckdb(eeg_df, admins_df, patients_with_med_data):
    con = duckdb.connect()
    eeg_slice = eeg_df[["_row_id", "BDSPPatientID", "_eeg_ts", "_window_start"]].copy()
    eeg_slice["BDSPPatientID"] = eeg_slice["BDSPPatientID"].astype(str)
    adm = admins_df.copy()
    adm["BDSPPatientID"] = adm["BDSPPatientID"].astype(str)
    con.register("eegs", eeg_slice)
    con.register("adm", adm)
    result = con.execute("""
        SELECT e._row_id,
               MAX(CASE WHEN adm._med_ts BETWEEN e._window_start AND e._eeg_ts THEN 1 ELSE 0 END) AS exposure
        FROM eegs e
        LEFT JOIN adm ON e.BDSPPatientID = adm.BDSPPatientID
                     AND adm._med_ts >= e._window_start
                     AND adm._med_ts <= e._eeg_ts
        GROUP BY e._row_id
    """).df()
    con.close()
    m = dict(zip(result["_row_id"], result["exposure"]))
    series = eeg_df["_row_id"].map(m).fillna(0).astype(int)
    return apply_na_mask(eeg_df, series, patients_with_med_data)


def _exposure_via_duckdb_intervals(eeg_df, intervals_df, patients_with_med_data):
    con = duckdb.connect()
    eeg_slice = eeg_df[["_row_id", "BDSPPatientID", "_eeg_ts", "_window_start"]].copy()
    eeg_slice["BDSPPatientID"] = eeg_slice["BDSPPatientID"].astype(str)
    iv = intervals_df.copy()
    iv["BDSPPatientID"] = iv["BDSPPatientID"].astype(str)
    con.register("eegs", eeg_slice)
    con.register("iv", iv)
    result = con.execute("""
        SELECT e._row_id,
               MAX(CASE WHEN iv._start <= e._eeg_ts AND iv._end >= e._window_start THEN 1 ELSE 0 END) AS exposure
        FROM eegs e
        LEFT JOIN iv ON e.BDSPPatientID = iv.BDSPPatientID
                    AND iv._start <= e._eeg_ts
                    AND iv._end   >= e._window_start
        GROUP BY e._row_id
    """).df()
    con.close()
    m = dict(zip(result["_row_id"], result["exposure"]))
    series = eeg_df["_row_id"].map(m).fillna(0).astype(int)
    return apply_na_mask(eeg_df, series, patients_with_med_data)


def compute_bolus_exposure(eeg_df, drug_list, bucket, patients_with_med_data):
    frames = []
    for drug in drug_list:
        path = med_config.I0001_CAMS_BOLUS_EXPOSURE_DIR / f"{drug}_boluses_{bucket}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, usecols=["BDSPPatientID", "TakenTime"]))
    if not frames:
        series = pd.Series(0, index=eeg_df.index, dtype="Int64")
        return apply_na_mask(eeg_df, series, patients_with_med_data)
    admins = pd.concat(frames, ignore_index=True)
    admins["BDSPPatientID"] = norm_id(admins["BDSPPatientID"])
    admins["_med_ts"] = pd.to_datetime(admins["TakenTime"], errors="coerce")
    admins = admins[admins["BDSPPatientID"].notna() & admins["_med_ts"].notna()]
    if len(admins) == 0:
        series = pd.Series(0, index=eeg_df.index, dtype="Int64")
        return apply_na_mask(eeg_df, series, patients_with_med_data)
    return _exposure_via_duckdb(eeg_df, admins[["BDSPPatientID", "_med_ts"]], patients_with_med_data)


def compute_infusion_exposure(eeg_df, drug, source_dir, patients_with_med_data):
    path = source_dir / f"{drug}_intervals.csv"
    if not path.exists():
        series = pd.Series(0, index=eeg_df.index, dtype="Int64")
        return apply_na_mask(eeg_df, series, patients_with_med_data)
    intervals = pd.read_csv(path, usecols=["BDSPPatientID", "StartTime", "EndTime"])
    intervals["BDSPPatientID"] = norm_id(intervals["BDSPPatientID"])
    intervals["_start"] = pd.to_datetime(intervals["StartTime"], errors="coerce")
    intervals["_end"]   = pd.to_datetime(intervals["EndTime"],   errors="coerce")
    intervals = intervals[
        intervals["BDSPPatientID"].notna()
        & intervals["_start"].notna()
        & intervals["_end"].notna()
    ]
    if len(intervals) == 0:
        series = pd.Series(0, index=eeg_df.index, dtype="Int64")
        return apply_na_mask(eeg_df, series, patients_with_med_data)
    return _exposure_via_duckdb_intervals(eeg_df, intervals[["BDSPPatientID", "_start", "_end"]], patients_with_med_data)


def compute_midazolam_bolus_without_infusion(eeg_df, patients_with_med_data):
    bolus_path = med_config.I0001_CAMS_BOLUS_EXPOSURE_DIR / "midazolam_boluses_iv.csv"
    if not bolus_path.exists():
        series = pd.Series(0, index=eeg_df.index, dtype="Int64")
        return apply_na_mask(eeg_df, series, patients_with_med_data)
    boluses = pd.read_csv(bolus_path, usecols=["BDSPPatientID", "TakenTime"])
    boluses["BDSPPatientID"] = norm_id(boluses["BDSPPatientID"])
    boluses["_med_ts"] = pd.to_datetime(boluses["TakenTime"], errors="coerce")
    boluses = boluses[boluses["BDSPPatientID"].notna() & boluses["_med_ts"].notna()]
    if len(boluses) == 0:
        series = pd.Series(0, index=eeg_df.index, dtype="Int64")
        return apply_na_mask(eeg_df, series, patients_with_med_data)

    inf_path = med_config.I0001_CAMS_SEDATIVE_EXPOSURE_DIR / "midazolam_intervals.csv"
    if inf_path.exists():
        intervals = pd.read_csv(inf_path, usecols=["BDSPPatientID", "StartTime", "EndTime"])
        intervals["BDSPPatientID"] = norm_id(intervals["BDSPPatientID"])
        intervals["_start"] = pd.to_datetime(intervals["StartTime"], errors="coerce")
        intervals["_end"]   = pd.to_datetime(intervals["EndTime"],   errors="coerce")
        intervals = intervals[
            intervals["BDSPPatientID"].notna()
            & intervals["_start"].notna()
            & intervals["_end"].notna()
        ]
    else:
        intervals = pd.DataFrame(columns=["BDSPPatientID", "_start", "_end"])

    con = duckdb.connect()
    con.register("bol", boluses[["BDSPPatientID", "_med_ts"]])
    con.register("inf", intervals[["BDSPPatientID", "_start", "_end"]])
    standalone = con.execute("""
        SELECT b.BDSPPatientID, b._med_ts
        FROM bol b
        LEFT JOIN inf i ON b.BDSPPatientID = i.BDSPPatientID
                       AND b._med_ts >= i._start
                       AND b._med_ts <= i._end
        WHERE i.BDSPPatientID IS NULL
    """).df()
    con.close()
    if len(standalone) == 0:
        series = pd.Series(0, index=eeg_df.index, dtype="Int64")
        return apply_na_mask(eeg_df, series, patients_with_med_data)
    return _exposure_via_duckdb(eeg_df, standalone, patients_with_med_data)


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
        print(f"CAMS EEG rows: {len(meta):,}")

        if EEG_TIMESTAMP_COL not in meta.columns:
            raise ValueError(f"'{EEG_TIMESTAMP_COL}' not found.")

        meta["_row_id"] = range(len(meta))
        meta["BDSPPatientID"] = norm_id(meta["BDSPPatientID"])
        meta["_eeg_ts"] = pd.to_datetime(meta[EEG_TIMESTAMP_COL], errors="coerce")
        meta["_window_start"] = meta["_eeg_ts"] - pd.Timedelta(hours=WINDOW_HOURS)

        # Pull ASM/antipsych from earlier output
        asm_path = med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR / "i0001_cams_per_eeg_asm_antipsych_exposure.csv"
        asm = pd.read_csv(asm_path, low_memory=False)
        asm["BDSPPatientID"] = norm_id(asm["BDSPPatientID"])
        asm["_eeg_ts"] = pd.to_datetime(asm[EEG_TIMESTAMP_COL], errors="coerce")
        asm_slim = asm[["BDSPPatientID", "_eeg_ts", "LongActingASM_Exposure", "Antipsychotic_Exposure"]] \
            .drop_duplicates(subset=["BDSPPatientID", "_eeg_ts"], keep="first")
        meta = meta.merge(asm_slim, on=["BDSPPatientID", "_eeg_ts"], how="left")
        for col in ["LongActingASM_Exposure", "Antipsychotic_Exposure"]:
            meta[col] = meta[col].astype("Int64")
            meta.loc[~meta["BDSPPatientID"].isin(patients_with_med_data), col] = pd.NA

        # Benzos
        core = compute_bolus_exposure(meta, BENZO_DRUGS, "iv", patients_with_med_data)
        midaz_standalone = compute_midazolam_bolus_without_infusion(meta, patients_with_med_data)
        combined = core.astype("Int64").combine(midaz_standalone.astype("Int64"),
                                                lambda a, b: pd.NA if (pd.isna(a) and pd.isna(b))
                                                             else (1 if (a == 1 or b == 1) else 0))
        meta["FastActing_Benzo_Exposure"] = combined
        meta["SlowActing_Benzo_Exposure"] = compute_bolus_exposure(meta, BENZO_DRUGS + ["midazolam"], "non_iv", patients_with_med_data)

        # Propofol bolus
        meta["Propofol_Exposure"] = compute_bolus_exposure(meta, ["propofol"], "iv", patients_with_med_data)

        # Opiate bolus
        meta["FastActing_Opiate_Exposure"] = compute_bolus_exposure(meta, OPIATE_DRUGS, "iv", patients_with_med_data)
        meta["SlowActing_Opiate_Exposure"] = compute_bolus_exposure(meta, OPIATE_DRUGS, "non_iv", patients_with_med_data)

        # Per-drug infusions
        for drug in SEDATIVE_INFUSION_DRUGS:
            col = f"{drug.capitalize()}_Infusion_Exposure"
            meta[col] = compute_infusion_exposure(meta, drug, med_config.I0001_CAMS_SEDATIVE_EXPOSURE_DIR, patients_with_med_data)
        for drug in OPIATE_INFUSION_DRUGS:
            col = f"{drug.capitalize()}_Infusion_Exposure"
            meta[col] = compute_infusion_exposure(meta, drug, med_config.I0001_CAMS_OPIATE_EXPOSURE_DIR, patients_with_med_data)

        # Summary
        flag_cols = [
            "LongActingASM_Exposure", "Antipsychotic_Exposure",
            "FastActing_Benzo_Exposure", "SlowActing_Benzo_Exposure",
            "Propofol_Exposure",
            "FastActing_Opiate_Exposure", "SlowActing_Opiate_Exposure",
            "Propofol_Infusion_Exposure", "Midazolam_Infusion_Exposure",
            "Ketamine_Infusion_Exposure", "Dexmedetomidine_Infusion_Exposure",
            "Fentanyl_Infusion_Exposure", "Morphine_Infusion_Exposure",
            "Hydromorphone_Infusion_Exposure",
        ]
        eligible_mask = meta["BDSPPatientID"].isin(patients_with_med_data)
        n_eegs_eligible = int(eligible_mask.sum())
        n_pts_eligible = meta.loc[eligible_mask, "BDSPPatientID"].nunique()
        n_eegs_excluded = int((~eligible_mask).sum())
        n_pts_excluded = meta.loc[~eligible_mask, "BDSPPatientID"].nunique()
        print("\n=== Summary ===")
        print(f"  Total EEG rows: {len(meta):,}")
        print(f"  EEGs eligible (patient has med data):    {n_eegs_eligible:,}")
        print(f"  EEGs not eligible (no med data, all NA): {n_eegs_excluded:,} "
              f"from {n_pts_excluded:,} patients")
        for col in flag_cols:
            n_e = int((meta[col] == 1).sum())
            n_p = meta.loc[meta[col] == 1, "BDSPPatientID"].nunique()
            pct_e = 100 * n_e / n_eegs_eligible if n_eegs_eligible else 0
            pct_p = 100 * n_p / n_pts_eligible if n_pts_eligible else 0
            print(f"  {col:<35}  EEGs: {n_e:>6,} ({pct_e:.1f}%)  "
                  f"Patients: {n_p:>5,} ({pct_p:.1f}%)")

        out = meta.drop(columns=["_row_id", "_eeg_ts", "_window_start"])
        out_path = med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR / "i0001_cams_per_eeg_master_exposure.csv"
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