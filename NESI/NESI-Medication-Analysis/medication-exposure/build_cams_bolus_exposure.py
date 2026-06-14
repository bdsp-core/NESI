# build_cams_bolus_exposure.py
"""
Bolus extraction for all 10 drugs (sedatives, opiates, benzos) for the I0001
CAMS cohort. Mirrors build_bolus_exposure.py but uses CAMS paths.

Paths read from med_config.py.
"""

import duckdb
import pandas as pd

import med_config
from drug_configs import DRUG_CONFIGS
from opiate_configs import OPIATE_CONFIGS
from benzo_bolus_configs import BENZO_BOLUS_CONFIGS
from infusion_reconstruction import extract_boluses_for_cohort
from bolus_route_classify import add_route_bucket

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


ALL_BOLUS_DRUGS = {**DRUG_CONFIGS, **OPIATE_CONFIGS, **BENZO_BOLUS_CONFIGS}


def name_filter_sql(patterns):
    return "(" + " OR ".join(
        f"LOWER(MedicationDSC) LIKE '%{p}%' OR LOWER(MedicationDisplayNM) LIKE '%{p}%'"
        for p in patterns
    ) + ")"


def fetch_drug_rows(drug_name, parquet_path):
    cfg = ALL_BOLUS_DRUGS[drug_name]
    return duckdb.query(f"""
        SELECT *
        FROM read_parquet('{parquet_path}')
        WHERE {name_filter_sql(cfg['name_patterns'])}
    """).df()


def process_drug(drug_name, parquet_path):
    print(f"\n=== {drug_name} ===")
    cfg = ALL_BOLUS_DRUGS[drug_name]
    expected_units = cfg.get("bolus_units") or (cfg["mass_unit"],) if "mass_unit" in cfg else ("mg",)

    rows = fetch_drug_rows(drug_name, parquet_path)
    print(f"  Total rows: {len(rows):,}")
    print(f"  Unique patients: {rows['BDSPPatientID'].nunique():,}")

    if "MedicationTakenDTS" in rows.columns:
        rows["MedicationTakenDTS"] = pd.to_datetime(rows["MedicationTakenDTS"], errors="coerce")

    boluses = extract_boluses_for_cohort(rows, expected_units=expected_units)
    print(f"  Bolus events: {len(boluses):,}")

    if len(boluses) == 0:
        return {"drug": drug_name, "iv": 0, "non_iv": 0, "excluded": 0}

    boluses = add_route_bucket(boluses, rows)

    iv_df     = boluses[boluses["route_bucket"] == "iv"].drop(columns=["route_bucket"])
    non_iv_df = boluses[boluses["route_bucket"] == "non_iv"].drop(columns=["route_bucket"])
    n_excl    = int((boluses["route_bucket"] == "excluded").sum())

    iv_path     = med_config.I0001_CAMS_BOLUS_EXPOSURE_DIR / f"{drug_name}_boluses_iv.csv"
    non_iv_path = med_config.I0001_CAMS_BOLUS_EXPOSURE_DIR / f"{drug_name}_boluses_non_iv.csv"
    iv_df.to_csv(iv_path, index=False)
    non_iv_df.to_csv(non_iv_path, index=False)

    print(f"  IV: {len(iv_df):,}   Non-IV: {len(non_iv_df):,}   Excluded: {n_excl:,}")
    return {"drug": drug_name, "iv": len(iv_df), "non_iv": len(non_iv_df), "excluded": n_excl}


def main():
    try:
        med_config.I0001_CAMS_BOLUS_EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)
        parquet_path = str(med_config.I0001_CAMS_COHORT_MEDS_PARQUET).replace("\\", "/")
        print(f"Reading: {parquet_path}")
        print(f"Output:  {med_config.I0001_CAMS_BOLUS_EXPOSURE_DIR}")

        summary = []
        for drug in ALL_BOLUS_DRUGS:
            summary.append(process_drug(drug, parquet_path))

        summary_df = pd.DataFrame(summary)
        summary_path = med_config.I0001_CAMS_BOLUS_EXPOSURE_DIR / "bolus_extraction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\n=== Summary ===")
        print(summary_df.to_string(index=False))

        print("\nDone.")
        if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_OK)
    except Exception as e:
        print(f"\nERROR: {e}")
        if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_ICONHAND)
        raise


if __name__ == "__main__":
    main()