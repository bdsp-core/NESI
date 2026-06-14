# build_i0002_propofol_boluses.py
"""
Extract propofol bolus events from the I0002 GCS cohort.

Reads from I0002_PIVOTED_MEDS_PATCHED_PARQUET. Uses ProductDescription as
the drug-name field (since MedicationDSC is mostly null in I0002). Applies
the same route classification (IV/non_iv via bolus_route_classify) and
bolus identification (MARActionDSC against BOLUS_GIVEN_ACTIONS) as the
benzo extraction.

Writes:
  I0002_BOLUS_EXPOSURE_DIR/propofol_boluses_iv.csv
  I0002_BOLUS_EXPOSURE_DIR/propofol_boluses_non_iv.csv

Paths read from med_config.py.
"""

import duckdb
import pandas as pd

import med_config
from infusion_reconstruction import extract_boluses_for_cohort
from bolus_route_classify import add_route_bucket

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


NAME_PATTERNS = ["propofol", "diprivan"]
BOLUS_UNITS = ("mg",)
DRUG_NAME = "propofol"

I0002_DRUG_NAME_COL = "ProductDescription"


def name_filter_sql(patterns, col):
    return "(" + " OR ".join(
        f"LOWER({col}) LIKE '%{p}%'" for p in patterns
    ) + ")"


try:
    med_config.I0002_BOLUS_EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = str(med_config.I0002_PIVOTED_MEDS_PATCHED_PARQUET).replace("\\", "/")

    print(f"Reading from: {parquet_path}")
    print(f"Output dir:   {med_config.I0002_BOLUS_EXPOSURE_DIR}\n")

    rows = duckdb.query(f"""
        SELECT
            *,
            {I0002_DRUG_NAME_COL} AS MedicationDSC
        FROM read_parquet('{parquet_path}')
        WHERE {name_filter_sql(NAME_PATTERNS, I0002_DRUG_NAME_COL)}
    """).df()
    print(f"Total {DRUG_NAME} rows: {len(rows):,}")
    print(f"Unique patients:  {rows['BDSPPatientID'].nunique():,}")

    if "MedicationTakenDTS" in rows.columns:
        rows["MedicationTakenDTS"] = pd.to_datetime(
            rows["MedicationTakenDTS"], errors="coerce"
        )

    boluses = extract_boluses_for_cohort(rows, expected_units=BOLUS_UNITS)
    print(f"Bolus events extracted: {len(boluses):,}")

    if len(boluses) == 0:
        print("No bolus events found. Stopping.")
    else:
        boluses = add_route_bucket(boluses, rows)

        iv_df     = boluses[boluses["route_bucket"] == "iv"].drop(columns=["route_bucket"])
        non_iv_df = boluses[boluses["route_bucket"] == "non_iv"].drop(columns=["route_bucket"])
        n_excl    = int((boluses["route_bucket"] == "excluded").sum())

        iv_path     = med_config.I0002_BOLUS_EXPOSURE_DIR / f"{DRUG_NAME}_boluses_iv.csv"
        non_iv_path = med_config.I0002_BOLUS_EXPOSURE_DIR / f"{DRUG_NAME}_boluses_non_iv.csv"
        iv_df.to_csv(iv_path, index=False)
        non_iv_df.to_csv(non_iv_path, index=False)

        print(f"\nIV:        {len(iv_df):,}  →  {iv_path.name}")
        print(f"Non-IV:    {len(non_iv_df):,}  →  {non_iv_path.name}")
        print(f"Excluded:  {n_excl:,}")

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