# build_i0002_benzo_boluses.py
"""
Bolus extraction for I0002 benzodiazepines (lorazepam, diazepam, other_benzo).
Mirrors the I0001 build_bolus_exposure.py output format so downstream
exposure-summary code can treat the two cohorts uniformly.

Key differences from build_bolus_exposure.py:
  - Reads from I0002_PIVOTED_MEDS_PATCHED_PARQUET (single wide-format parquet)
  - Uses ProductDescription as the drug-name column (MedicationDSC is empty in I0002)
  - Only processes the three benzo categories from benzo_bolus_configs.py
    (the rest of the I0001 drugs are handled separately for I0002)

For each drug, pulls bolus events, classifies by route, and writes:
  <drug>_boluses_iv.csv     — IV (including PCA-as-IV)
  <drug>_boluses_non_iv.csv — non-IV routes
Epidural/intrathecal are excluded.

Paths read from med_config.py.
"""

import duckdb
import pandas as pd

import med_config
from i0002_benzo_bolus_configs import I0002_BENZO_BOLUS_CONFIGS as BENZO_BOLUS_CONFIGS
from infusion_reconstruction import extract_boluses_for_cohort
from bolus_route_classify import add_route_bucket

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# Column in the I0002 pivoted parquet that contains drug names.
# (I0001 uses MedicationDSC; I0002 only populates ProductDescription.)
I0002_DRUG_NAME_COL = "ProductDescription"


def name_filter_sql(patterns, col):
    return "(" + " OR ".join(
        f"LOWER({col}) LIKE '%{p}%'" for p in patterns
    ) + ")"


def fetch_drug_rows(drug_name, parquet_path):
    patterns = BENZO_BOLUS_CONFIGS[drug_name]["name_patterns"]
    name_filter = name_filter_sql(patterns, I0002_DRUG_NAME_COL)
    return duckdb.query(f"""
        SELECT
            *,
            {I0002_DRUG_NAME_COL} AS MedicationDSC
        FROM read_parquet('{parquet_path}')
        WHERE {name_filter}
    """).df()


def process_drug(drug_name, parquet_path):
    print(f"\n=== {drug_name} ===")
    cfg = BENZO_BOLUS_CONFIGS[drug_name]

    rows = fetch_drug_rows(drug_name, parquet_path)
    print(f"  Total rows: {len(rows):,}")
    print(f"  Unique patients: {rows['BDSPPatientID'].nunique():,}")

    if "MedicationTakenDTS" in rows.columns:
        rows["MedicationTakenDTS"] = pd.to_datetime(
            rows["MedicationTakenDTS"], errors="coerce"
        )

    boluses = extract_boluses_for_cohort(rows, expected_units=cfg["bolus_units"])
    print(f"  Bolus events extracted: {len(boluses):,}")

    if len(boluses) == 0:
        return {"drug": drug_name, "iv": 0, "non_iv": 0, "excluded": 0}

    boluses = add_route_bucket(boluses, rows)

    iv_df     = boluses[boluses["route_bucket"] == "iv"].drop(columns=["route_bucket"])
    non_iv_df = boluses[boluses["route_bucket"] == "non_iv"].drop(columns=["route_bucket"])
    n_excl    = int((boluses["route_bucket"] == "excluded").sum())

    iv_path     = med_config.I0002_BOLUS_EXPOSURE_DIR / f"{drug_name}_boluses_iv.csv"
    non_iv_path = med_config.I0002_BOLUS_EXPOSURE_DIR / f"{drug_name}_boluses_non_iv.csv"
    iv_df.to_csv(iv_path, index=False)
    non_iv_df.to_csv(non_iv_path, index=False)

    print(f"  IV:        {len(iv_df):,}  →  {iv_path.name}")
    print(f"  Non-IV:    {len(non_iv_df):,}  →  {non_iv_path.name}")
    print(f"  Excluded:  {n_excl:,}")

    return {
        "drug": drug_name,
        "iv": len(iv_df),
        "non_iv": len(non_iv_df),
        "excluded": n_excl,
    }


try:
    med_config.I0002_BOLUS_EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = str(
        med_config.I0002_PIVOTED_MEDS_PATCHED_PARQUET
    ).replace("\\", "/")

    print(f"Reading from: {parquet_path}")
    print(f"Output dir:   {med_config.I0002_BOLUS_EXPOSURE_DIR}")

    summary = []
    for drug in BENZO_BOLUS_CONFIGS:
        summary.append(process_drug(drug, parquet_path))

    summary_df = pd.DataFrame(summary)
    summary_path = med_config.I0002_BOLUS_EXPOSURE_DIR / "bolus_extraction_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")

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