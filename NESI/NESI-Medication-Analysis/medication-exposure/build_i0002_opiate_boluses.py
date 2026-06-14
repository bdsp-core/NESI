# build_i0002_opiate_boluses.py
"""
Extract opiate bolus events (fentanyl, morphine, hydromorphone) from the
I0002 GCS cohort.

Reads from I0002_PIVOTED_MEDS_PATCHED_PARQUET. Uses ProductDescription as
the drug-name field. Applies the standard route classification (IV/non_iv)
and bolus identification via BOLUS_GIVEN_ACTIONS in infusion_reconstruction.

Per-drug bolus units come from opiate_configs.py (mass_unit field):
  fentanyl      -> mcg
  morphine      -> mg
  hydromorphone -> mg

Writes for each drug:
  I0002_BOLUS_EXPOSURE_DIR/{drug}_boluses_iv.csv
  I0002_BOLUS_EXPOSURE_DIR/{drug}_boluses_non_iv.csv

Paths read from med_config.py.
"""

import duckdb
import pandas as pd

import med_config
from opiate_configs import OPIATE_CONFIGS
from infusion_reconstruction import extract_boluses_for_cohort
from bolus_route_classify import add_route_bucket

import bolus_route_classify
print(f"Loaded from: {bolus_route_classify.__file__}")
import inspect
print(inspect.getsource(bolus_route_classify.add_route_bucket))

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


I0002_DRUG_NAME_COL = "ProductDescription"


def name_filter_sql(patterns, col):
    return "(" + " OR ".join(
        f"LOWER({col}) LIKE '%{p}%'" for p in patterns
    ) + ")"


def process_drug(drug_name, parquet_path):
    print(f"\n=== {drug_name} ===")
    cfg = OPIATE_CONFIGS[drug_name]
    patterns = cfg["name_patterns"]
    expected_units = (cfg["mass_unit"],)

    rows = duckdb.query(f"""
        SELECT
            *,
            {I0002_DRUG_NAME_COL} AS MedicationDSC
        FROM read_parquet('{parquet_path}')
        WHERE {name_filter_sql(patterns, I0002_DRUG_NAME_COL)}
    """).df()
    print(f"  Total rows: {len(rows):,}")
    print(f"  Unique patients: {rows['BDSPPatientID'].nunique():,}")

    if "MedicationTakenDTS" in rows.columns:
        rows["MedicationTakenDTS"] = pd.to_datetime(
            rows["MedicationTakenDTS"], errors="coerce"
        )

    boluses = extract_boluses_for_cohort(rows, expected_units=expected_units)
    print(f"  Bolus events extracted: {len(boluses):,}")

    if len(boluses) == 0:
        return {"drug": drug_name, "iv": 0, "non_iv": 0, "excluded": 0}

    boluses = add_route_bucket(boluses, rows)

    iv_df     = boluses[boluses["route_bucket"] == "iv"].drop(columns=["route_bucket"])
    non_iv_df = boluses[boluses["route_bucket"] == "non_iv"].drop(columns=["route_bucket"])
    n_excl    = int((boluses["route_bucket"] == "excluded").sum())

    iv_path     = med_config.I0002_BOLUS_EXPOSURE_DIR / f"{drug_name}_boluses_iv.csv"
    non_iv_path = med_config.I0002_BOLUS_EXPOSURE_DIR / f"{drug_name}_boluses_non_iv.csv"
    print(f"DEBUG: writing IV to absolute path: {iv_path.resolve()}")
    print(f"DEBUG: writing non-IV to absolute path: {non_iv_path.resolve()}")
    print(f"DEBUG: non_iv_df has {len(non_iv_df)} rows about to be written")
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
    parquet_path = str(med_config.I0002_PIVOTED_MEDS_PATCHED_PARQUET).replace("\\", "/")

    print(f"Reading from: {parquet_path}")
    print(f"Output dir:   {med_config.I0002_BOLUS_EXPOSURE_DIR}")

    summary = []
    for drug in OPIATE_CONFIGS:
        summary.append(process_drug(drug, parquet_path))

    summary_df = pd.DataFrame(summary)
    summary_path = med_config.I0002_BOLUS_EXPOSURE_DIR / "opiate_bolus_extraction_summary.csv"
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