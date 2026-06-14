# build_cams_opiate_exposure.py
"""
Opiate infusion reconstruction (fentanyl, morphine, hydromorphone) for the
I0001 CAMS cohort. Mirrors build_opiate_exposure.py but uses CAMS paths.

Paths read from med_config.py.
"""

import duckdb
import pandas as pd

import med_config
from opiate_configs import OPIATE_CONFIGS
from infusion_reconstruction import reconstruct_infusions_for_cohort

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def name_filter_sql(patterns):
    return "(" + " OR ".join(
        f"LOWER(MedicationDSC) LIKE '%{p}%' OR LOWER(MedicationDisplayNM) LIKE '%{p}%'"
        for p in patterns
    ) + ")"


def route_exclusion_sql(patterns):
    if not patterns:
        return "1=1"
    parts = [
        f"LOWER(COALESCE(MedicationRouteDSC, '') || ' ' || COALESCE(RouteDSC, '')) NOT LIKE '%{p}%'"
        for p in patterns
    ]
    return "(" + " AND ".join(parts) + ")"


def fetch_drug_rows(drug_name, parquet_path):
    cfg = OPIATE_CONFIGS[drug_name]
    excluded = cfg.get("excluded_route_patterns", [])
    return duckdb.query(f"""
        SELECT *
        FROM read_parquet('{parquet_path}')
        WHERE {name_filter_sql(cfg['name_patterns'])}
          AND {route_exclusion_sql(excluded)}
    """).df()


def process_drug(drug_name, parquet_path):
    print(f"\n=== {drug_name} ===")
    cfg = OPIATE_CONFIGS[drug_name]

    rows = fetch_drug_rows(drug_name, parquet_path)
    print(f"  {len(rows):,} rows after route filtering")
    print(f"  {rows['BDSPPatientID'].nunique():,} unique patients")

    for col in ["MedicationTakenDTS", "OrderStartDTS", "OrderEndDTS"]:
        if col in rows.columns:
            rows[col] = pd.to_datetime(rows[col], errors="coerce")

    if "InfusionRateNBR" in rows.columns:
        rows["InfusionRateNBR"] = pd.to_numeric(rows["InfusionRateNBR"], errors="coerce")

    infusion_rows = rows[
        rows["InfusionRateNBR"].notna() & rows["MedicationTakenDTS"].notna()
    ]
    intervals = reconstruct_infusions_for_cohort(infusion_rows)
    print(f"  {len(intervals):,} reconstructed intervals")

    if len(intervals) > 0:
        intervals["StartTime"] = pd.to_datetime(intervals["StartTime"], errors="coerce")
        intervals["EndTime"]   = pd.to_datetime(intervals["EndTime"],   errors="coerce")
        n_before = len(intervals)
        intervals = intervals[intervals["EndTime"] > intervals["StartTime"]].copy()
        n_dropped = n_before - len(intervals)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped:,} zero/negative-duration intervals")

    if len(intervals) > 0:
        merge_key = (
            infusion_rows[["BDSPPatientID", "MedicationTakenDTS", "MedicationDSC"]]
            .rename(columns={"MedicationTakenDTS": "StartTime"})
        )
        intervals = intervals.merge(merge_key, on=["BDSPPatientID", "StartTime"], how="left")
        intervals["ConcentrationPerML"] = intervals["MedicationDSC"].apply(cfg["parse_concentration"])
        intervals["Rate"] = pd.to_numeric(intervals["Rate"], errors="coerce")
        intervals["ConcentrationPerML"] = pd.to_numeric(intervals["ConcentrationPerML"], errors="coerce")
        intervals["Rate_converted_per_hr"] = intervals["Rate"] * intervals["ConcentrationPerML"]

    out_path = med_config.I0001_CAMS_OPIATE_EXPOSURE_DIR / f"{drug_name}_intervals.csv"
    intervals.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


def main():
    try:
        med_config.I0001_CAMS_OPIATE_EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)
        parquet_path = str(med_config.I0001_CAMS_COHORT_MEDS_PARQUET).replace("\\", "/")
        print(f"Reading: {parquet_path}")
        print(f"Output:  {med_config.I0001_CAMS_OPIATE_EXPOSURE_DIR}")

        for drug in OPIATE_CONFIGS:
            process_drug(drug, parquet_path)

        print("\nDone.")
        if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_OK)
    except Exception as e:
        print(f"\nERROR: {e}")
        if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_ICONHAND)
        raise


if __name__ == "__main__":
    main()