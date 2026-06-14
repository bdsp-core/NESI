# build_final_exposure_tables.py
"""
Build the final patient-level and EEG-level medication exposure summary
tables for Table 1.

Includes top-3 maintenance ASMs as sub-rows under 'Maintenance ASM'.

Column order: I0001 GCS, I0002 GCS, I0001 RASS, I0001 CAMS.

Reads each cohort's master per-EEG exposure file. For per-ASM-drug breakdown,
reads each cohort's long_acting_asm_administrations.csv and recomputes per-drug
24h-prior exposure.

Filters each cohort to EEG rows with a non-null NESI value (except I0001 CAMS,
which already has NESI for all rows by construction).

Reports exposure denominators as patients with both NESI values AND
medication administration records. For I0002 GCS, infusion-rate-based
exposures are reported as '—' because reliable infusion rate data are not
available in that cohort.

Writes:
  exposure_summary_by_patient_final.csv
  exposure_summary_by_eeg_final.csv

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

# Order matters for the final table
COHORTS = [
    {
        "label": "I0001 GCS",
        "master_path": med_config.I0001_ASM_ANTIPSYCH_DIR / "i0001_per_eeg_master_exposure.csv",
        "metadata_files": med_config.I0001_GCS_METADATA_CSVS,
        "cohort_meds_parquet": med_config.COHORT_ALL_MEDS_PARQUET_NEW,
        "asm_admin_path": med_config.I0001_ASM_ANTIPSYCH_DIR / "long_acting_asm_administrations.csv",
        "ts_col": "GCSRecordedDTS",
        "filter_col": "_cohort",
        "filter_val": "GCS",
        "nesi_lookup": {
            "path": med_config.GCS_NESI_LOOKUP_CSV,
            "ts_col": "GCSRecordedDTS",
        },
    },
    {
        "label": "I0002 GCS",
        "master_path": med_config.I0002_ASM_ANTIPSYCH_DIR / "i0002_per_eeg_master_exposure.csv",
        "metadata_files": [med_config.I0002_GCS_METADATA_CSV],
        "cohort_meds_parquet": med_config.I0002_PIVOTED_MEDS_PATCHED_PARQUET,
        "asm_admin_path": med_config.I0002_ASM_ANTIPSYCH_DIR / "long_acting_asm_administrations.csv",
        "ts_col": "RecordedDTS",
        "filter_col": None,
        "filter_val": None,
        "nesi_lookup": {
            "path": med_config.GCS_NESI_LOOKUP_CSV,
            "ts_col": "GCSRecordedDTS",
        },
    },
    {
        "label": "I0001 RASS",
        "master_path": med_config.I0001_ASM_ANTIPSYCH_DIR / "i0001_per_eeg_master_exposure.csv",
        "metadata_files": med_config.I0001_RASS_METADATA_CSVS,
        "cohort_meds_parquet": med_config.COHORT_ALL_MEDS_PARQUET_NEW,
        "asm_admin_path": med_config.I0001_ASM_ANTIPSYCH_DIR / "long_acting_asm_administrations.csv",
        "ts_col": "RASSRecordedDTS",
        "filter_col": "_cohort",
        "filter_val": "RASS",
        "nesi_lookup": {
            "path": med_config.RASS_NESI_LOOKUP_CSV,
            "ts_col": "RASSRecordedDTS",
        },
    },
    {
        "label": "I0001 CAMS",
        "master_path": med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR / "i0001_cams_per_eeg_master_exposure.csv",
        "metadata_files": med_config.I0001_CAMS_METADATA_CSVS,
        "cohort_meds_parquet": med_config.I0001_CAMS_COHORT_MEDS_PARQUET,
        "asm_admin_path": med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR / "long_acting_asm_administrations.csv",
        "ts_col": "Snippet_StartDTS",
        "filter_col": None,
        "filter_val": None,
        "nesi_lookup": None,  # CAMS already has NESI for all rows
    },
]

# Categories for Table 1 rows. Sub-rows indented with leading spaces.
CATEGORIES = [
    ("Maintenance ASM",                  ["LongActingASM_Exposure"]),
    ("  Levetiracetam",                  "TOP_ASM:Levetiracetam"),
    ("  Lacosamide",                     "TOP_ASM:Lacosamide"),
    ("  Valproate",                      "TOP_ASM:Valproate"),
    ("Parenteral benzo (non-infusion)",  ["FastActing_Benzo_Exposure"]),
    ("Enteral benzo (non-ASM)",          ["SlowActing_Benzo_Exposure"]),
    ("Sedative infusion",                ["Propofol_Exposure",
                                          "Propofol_Infusion_Exposure",
                                          "Midazolam_Infusion_Exposure",
                                          "Ketamine_Infusion_Exposure",
                                          "Dexmedetomidine_Infusion_Exposure"]),
    ("  Propofol",                       ["Propofol_Exposure", "Propofol_Infusion_Exposure"]),
    ("  Midazolam",                      ["Midazolam_Infusion_Exposure"]),
    ("  Ketamine",                       ["Ketamine_Infusion_Exposure"]),
    ("  Dexmedetomidine",                ["Dexmedetomidine_Infusion_Exposure"]),
    ("Opiate infusion",                  ["Fentanyl_Infusion_Exposure",
                                          "Morphine_Infusion_Exposure",
                                          "Hydromorphone_Infusion_Exposure"]),
    ("  Fentanyl",                       ["Fentanyl_Infusion_Exposure"]),
    ("  Morphine",                       ["Morphine_Infusion_Exposure"]),
    ("  Hydromorphone",                  ["Hydromorphone_Infusion_Exposure"]),
    ("Parenteral opiate",                ["FastActing_Opiate_Exposure"]),
    ("Enteral opiate",                   ["SlowActing_Opiate_Exposure"]),
    ("Antipsychotic",                    ["Antipsychotic_Exposure"]),
]

# Categories where I0002 cannot report infusion data (rate data not available
# in that cohort's medication records). These are shown as "—" for I0002.
INFUSION_CATEGORIES_NA_FOR_I0002 = {
    "Sedative infusion",
    "  Propofol",
    "  Midazolam",
    "  Ketamine",
    "  Dexmedetomidine",
    "Opiate infusion",
    "  Fentanyl",
    "  Morphine",
    "  Hydromorphone",
}

# Per-drug patterns for top-ASM breakdown
TOP_ASM_PATTERNS = {
    "Levetiracetam": ["levetiracetam", "keppra"],
    "Lacosamide":    ["lacosamide", "vimpat"],
    "Valproate":     ["valproic", "valproate", "divalproex", "depakote", "depakene", "depacon"],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def norm_id(s):
    return (
        s.astype(str).str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .replace({"nan": None, "None": None, "": None})
    )


def fmt(n_exposed, n_total):
    if n_total == 0:
        return "—"
    pct = 100 * n_exposed / n_total
    return f"{n_exposed:,} ({pct:.1f}%)"


def count_metadata_patients(metadata_files):
    pieces = []
    for f in metadata_files:
        pieces.append(pd.read_csv(str(f), usecols=["BDSPPatientID"], low_memory=False))
    df = pd.concat(pieces, ignore_index=True)
    return norm_id(df["BDSPPatientID"]).dropna().nunique()


def get_patients_with_med_data(label, df, cohorts_list):
    """Of patients in the filtered (post-NESI) cohort, return the subset
    that also has medication records in the cohort meds parquet."""
    cohort = next(c for c in cohorts_list if c["label"] == label)
    con = duckdb.connect()
    parquet_path = str(cohort["cohort_meds_parquet"]).replace("\\", "/")
    parquet_df = con.execute(f"""
        SELECT DISTINCT CAST(BDSPPatientID AS VARCHAR) AS pid
        FROM '{parquet_path}'
    """).df()
    con.close()
    parquet_ids = set(norm_id(parquet_df["pid"]).dropna())

    cohort_ids = set(df["BDSPPatientID"].dropna().unique())
    return cohort_ids & parquet_ids


def load_cohort(cohort):
    """Load master file, apply sub-cohort filter, optionally filter to EEG rows
    with a non-null NESI value, and attach _row_id, _eeg_ts, _window_start."""
    df = pd.read_csv(cohort["master_path"], low_memory=False)
    df["BDSPPatientID"] = norm_id(df["BDSPPatientID"])
    if cohort["filter_col"] is not None:
        df = df[df[cohort["filter_col"]] == cohort["filter_val"]].copy()

    n_before = len(df)
    pts_before = df["BDSPPatientID"].nunique()

    # Filter to rows with a NESI value, if a lookup is provided
    if cohort.get("nesi_lookup"):
        lookup_path = cohort["nesi_lookup"]["path"]
        lookup_ts_col = cohort["nesi_lookup"]["ts_col"]

        lookup = pd.read_csv(lookup_path, low_memory=False,
                             usecols=["BDSPPatientID", lookup_ts_col, "NESI"])
        lookup["BDSPPatientID"] = norm_id(lookup["BDSPPatientID"])
        lookup["_lookup_ts"] = pd.to_datetime(lookup[lookup_ts_col], errors="coerce")
        lookup = lookup[lookup["NESI"].notna()].copy()

        valid_keys = set(zip(lookup["BDSPPatientID"],
                             lookup["_lookup_ts"].astype("int64")))

        df["_match_ts"] = pd.to_datetime(df[cohort["ts_col"]], errors="coerce")
        df_keys = list(zip(df["BDSPPatientID"], df["_match_ts"].astype("int64")))
        keep_mask = pd.Series([k in valid_keys for k in df_keys], index=df.index)

        dropped = df[~keep_mask].copy()
        df = df[keep_mask].copy()
        df = df.drop(columns=["_match_ts"])

        if len(dropped) > 0:
            out_path = (cohort["master_path"].parent /
                        f"{cohort['label'].replace(' ', '_')}_dropped_no_NESI.csv")
            dropped.drop(columns=["_match_ts"]).to_csv(out_path, index=False)
            print(f"  Dropped rows written to {out_path}")

        n_after = len(df)
        pts_after = df["BDSPPatientID"].nunique()
        print(f"  NESI filter for {cohort['label']}: "
              f"{n_before:,} → {n_after:,} rows "
              f"({n_before - n_after:,} dropped); "
              f"{pts_before:,} → {pts_after:,} patients")

    df["_eeg_ts"] = pd.to_datetime(df[cohort["ts_col"]], errors="coerce")
    df["_window_start"] = df["_eeg_ts"] - pd.Timedelta(hours=WINDOW_HOURS)
    df["_row_id"] = range(len(df))
    return df


def category_exposure_per_eeg(df, cols):
    """OR across columns → per-EEG Series of 0/1."""
    if len(cols) == 1:
        return df[cols[0]].fillna(0).astype(int)
    return df[cols].fillna(0).max(axis=1).astype(int)


def compute_specific_asm_exposure(eeg_df, cohort, patterns):
    """
    Recompute per-EEG exposure for a specific ASM drug by reading the cohort's
    ASM admin CSV and filtering to patterns.
    Returns Series aligned with eeg_df.index (0/1).
    """
    admin_path = cohort["asm_admin_path"]
    if not admin_path.exists():
        return pd.Series(0, index=eeg_df.index, dtype=int)

    adm = pd.read_csv(admin_path, low_memory=False)
    if len(adm) == 0:
        return pd.Series(0, index=eeg_df.index, dtype=int)

    adm["BDSPPatientID"] = norm_id(adm["BDSPPatientID"])
    adm["_med_ts"] = pd.to_datetime(adm["MedicationTakenDTS"], errors="coerce")

    name_cols = [c for c in ["MedicationDSC", "MedicationDisplayNM", "ProductDescription"]
                 if c in adm.columns]
    if not name_cols:
        return pd.Series(0, index=eeg_df.index, dtype=int)

    name_combined = pd.Series("", index=adm.index)
    for c in name_cols:
        name_combined = name_combined + " " + adm[c].fillna("").astype(str)
    name_combined = name_combined.str.lower()

    patterns_lower = [p.lower() for p in patterns]
    mask = name_combined.apply(lambda s: any(p in s for p in patterns_lower))
    drug_adm = adm[mask & adm["BDSPPatientID"].notna() & adm["_med_ts"].notna()].copy()
    if len(drug_adm) == 0:
        return pd.Series(0, index=eeg_df.index, dtype=int)

    con = duckdb.connect()
    eeg_slice = eeg_df[["_row_id", "BDSPPatientID", "_eeg_ts", "_window_start"]].copy()
    eeg_slice["BDSPPatientID"] = eeg_slice["BDSPPatientID"].astype(str)
    drug_slice = drug_adm[["BDSPPatientID", "_med_ts"]].copy()
    drug_slice["BDSPPatientID"] = drug_slice["BDSPPatientID"].astype(str)
    con.register("eegs", eeg_slice)
    con.register("adm", drug_slice)

    result = con.execute("""
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

    exposure_map = dict(zip(result["_row_id"], result["exposure"]))
    return eeg_df["_row_id"].map(exposure_map).fillna(0).astype(int)


def build_cohort_exposure_columns(cohort, df):
    """For a single cohort, compute a per-EEG exposure column for each category."""
    out = {}
    for label, spec in CATEGORIES:
        if isinstance(spec, str) and spec.startswith("TOP_ASM:"):
            drug = spec.split(":", 1)[1]
            patterns = TOP_ASM_PATTERNS[drug]
            out[label] = compute_specific_asm_exposure(df, cohort, patterns)
        else:
            out[label] = category_exposure_per_eeg(df, spec)
    return out


def summarize(df, exposure_cols, level, meddata_pts):
    """
    level = 'patient' or 'eeg'
    meddata_pts = set of patient IDs with med data — used as the analysis cohort.
    Returns {label: (n_exposed, n_total)}
    """
    df_for_summary = df[df["BDSPPatientID"].isin(meddata_pts)]
    if level == "patient":
        n_total = len(meddata_pts)
    else:
        n_total = len(df_for_summary)

    out = {"_n_total": n_total}
    for label, series in exposure_cols.items():
        series_subset = series.loc[df_for_summary.index]
        if level == "patient":
            per_pt = df_for_summary.assign(_e=series_subset).groupby("BDSPPatientID")["_e"].max()
            n_exposed = int((per_pt == 1).sum())
        else:
            n_exposed = int((series_subset == 1).sum())
        out[label] = (n_exposed, n_total)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

try:
    # Load each cohort and compute its exposure columns
    cohort_dfs = {}
    cohort_exposures = {}
    for cohort in COHORTS:
        if not cohort["master_path"].exists():
            print(f"WARNING: {cohort['label']}: master file missing")
            continue
        print(f"Loading {cohort['label']}...")
        df = load_cohort(cohort)
        cohort_dfs[cohort["label"]] = df
        cohort_exposures[cohort["label"]] = build_cohort_exposure_columns(cohort, df)
        print(f"  {len(df):,} EEGs, {df['BDSPPatientID'].nunique():,} patients (post-NESI)")

    cohort_labels = [c["label"] for c in COHORTS if c["label"] in cohort_dfs]

    # Compute med-data subset for each cohort
    cohort_meddata_pts = {}
    for c in cohort_labels:
        cohort_meddata_pts[c] = get_patients_with_med_data(c, cohort_dfs[c], COHORTS)

    # Cohort flow summary (for methods/results reporting)
    print("\n" + "=" * 84)
    print("COHORT FLOW (paste these numbers into methods/results)")
    print("=" * 84)
    print(f"{'Cohort':<20} {'Metadata':>10} {'+NESI':>10} {'+MedData':>10} {'EEGs(MedData)':>14}")
    print("-" * 84)

    cohort_flow = {}
    for c in cohort_labels:
        cohort = next(co for co in COHORTS if co["label"] == c)
        n_metadata     = count_metadata_patients(cohort["metadata_files"])
        n_with_nesi    = cohort_dfs[c]["BDSPPatientID"].nunique()
        meddata_pts    = cohort_meddata_pts[c]
        n_with_med     = len(meddata_pts)
        n_eegs_with_med = len(cohort_dfs[c][cohort_dfs[c]["BDSPPatientID"].isin(meddata_pts)])
        cohort_flow[c] = {
            "n_metadata":   n_metadata,
            "n_with_nesi":  n_with_nesi,
            "n_with_med":   n_with_med,
            "n_eegs_with_med": n_eegs_with_med,
            "n_eegs_total": len(cohort_dfs[c]),
        }
        print(f"{c:<20} {n_metadata:>10,} {n_with_nesi:>10,} "
              f"{n_with_med:>10,} {n_eegs_with_med:>14,}")
    print("=" * 84)
    print("Metadata: total patients in cohort metadata file(s)")
    print("+NESI:    patients remaining after filtering to rows with NESI values")
    print("+MedData: of +NESI patients, those with medication records in EHR")
    print("EEGs(MedData): EEG segments from patients with NESI AND med data")
    print("=" * 84 + "\n")

    # Compute summaries (denominator = patients with med data)
    print("Computing patient-level summaries...")
    pt_summaries = {
        c: summarize(cohort_dfs[c], cohort_exposures[c], "patient",
                     meddata_pts=cohort_meddata_pts[c])
        for c in cohort_labels
    }

    print("Computing EEG-level summaries...")
    eeg_summaries = {
        c: summarize(cohort_dfs[c], cohort_exposures[c], "eeg",
                     meddata_pts=cohort_meddata_pts[c])
        for c in cohort_labels
    }

    # ─── Patient-level table ─────────────────────────────────────────────
    pt_rows = []

    header = {"Category": "Patients with med data"}
    for c in cohort_labels:
        n_nesi = cohort_flow[c]["n_with_nesi"]
        n_med = cohort_flow[c]["n_with_med"]
        pct = 100 * n_med / n_nesi if n_nesi else 0
        header[c] = f"{n_med:,} / {n_nesi:,} ({pct:.1f}%)"
    pt_rows.append(header)

    for label, _ in CATEGORIES:
        row = {"Category": label}
        for c in cohort_labels:
            if c == "I0002 GCS" and label in INFUSION_CATEGORIES_NA_FOR_I0002:
                row[c] = "—"
            else:
                n_e, n_t = pt_summaries[c][label]
                row[c] = fmt(n_e, n_t)
        pt_rows.append(row)

    pt_df = pd.DataFrame(pt_rows)
    pt_out = med_config.I0001_ASM_ANTIPSYCH_DIR / "exposure_summary_by_patient_final.csv"
    pt_df.to_csv(pt_out, index=False)
    print(f"\nPatient-level summary:")
    print(pt_df.to_string(index=False))
    print(f"Saved: {pt_out}")

    # ─── EEG-level table ─────────────────────────────────────────────────
    eeg_rows = []

    header = {"Category": "EEG segments with med data"}
    for c in cohort_labels:
        n_total = cohort_flow[c]["n_eegs_total"]
        n_med = cohort_flow[c]["n_eegs_with_med"]
        pct = 100 * n_med / n_total if n_total else 0
        header[c] = f"{n_med:,} / {n_total:,} ({pct:.1f}%)"
    eeg_rows.append(header)

    for label, _ in CATEGORIES:
        row = {"Category": label}
        for c in cohort_labels:
            if c == "I0002 GCS" and label in INFUSION_CATEGORIES_NA_FOR_I0002:
                row[c] = "—"
            else:
                n_e, n_t = eeg_summaries[c][label]
                row[c] = fmt(n_e, n_t)
        eeg_rows.append(row)

    eeg_df = pd.DataFrame(eeg_rows)
    eeg_out = med_config.I0001_ASM_ANTIPSYCH_DIR / "exposure_summary_by_eeg_final.csv"
    eeg_df.to_csv(eeg_out, index=False)
    print(f"\nEEG-level summary:")
    print(eeg_df.to_string(index=False))
    print(f"Saved: {eeg_out}")

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