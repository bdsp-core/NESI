# build_opiate_exposure.py
"""
Reconstruct continuous infusion intervals for opiates (fentanyl, morphine,
hydromorphone).

For each drug, pulls infusion-relevant rows from the cohort medication parquets
(applying route and name exclusions defined in opiate_configs.py),
reconstructs intervals via the state machine in infusion_reconstruction.py,
parses concentration from MedicationDSC, and converts rate (mL/hr) to mass/hr.

Zero/negative-duration intervals (state-machine artifacts) are dropped before
writing. Implausible-rate intervals are retained — binary exposure summaries
don't care about rate magnitude, and future weight-based plausibility filtering
will be a separate step.

Bolus extraction (including PRN doses, non-IV opiate routes, and PCA boluses)
is handled separately by build_bolus_exposure.py. PCA is excluded from
infusion extraction because PCA basal rates are not reliably captured in
MAR data for this cohort.

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


MEDS_GLOB = str(med_config.EXTRACT_PER_PART_DIR / "filtered_*.parquet").replace("\\", "/")


def name_filter_sql(patterns):
    return "(" + " OR ".join(
        f"LOWER(MedicationDSC) LIKE '%{p}%' OR LOWER(MedicationDisplayNM) LIKE '%{p}%'"
        for p in patterns
    ) + ")"


def name_exclusion_sql(patterns):
    if not patterns:
        return "1=1"
    parts = [f"LOWER(MedicationDSC) NOT LIKE '%{p}%'" for p in patterns]
    return "(" + " AND ".join(parts) + ")"


def route_exclusion_sql(patterns):
    if not patterns:
        return "1=1"
    parts = [
        f"LOWER(COALESCE(MedicationRouteDSC, '') || ' ' || COALESCE(RouteDSC, '')) NOT LIKE '%{p}%'"
        for p in patterns
    ]
    return "(" + " AND ".join(parts) + ")"


def fetch_drug_rows(drug_name):
    cfg = OPIATE_CONFIGS[drug_name]
    return duckdb.query(f"""
        SELECT *
        FROM read_parquet('{MEDS_GLOB}')
        WHERE {name_filter_sql(cfg['name_patterns'])}
          AND {name_exclusion_sql(cfg.get('excluded_name_patterns', []))}
          AND {route_exclusion_sql(cfg['excluded_route_patterns'])}
          AND InfusionRateNBR IS NOT NULL
    """).df()


def process_drug(drug_name):
    print(f"\n=== {drug_name} ===")
    cfg = OPIATE_CONFIGS[drug_name]

    print("  Fetching infusion rows...")
    rows = fetch_drug_rows(drug_name)
    print(f"    {len(rows):,} infusion rows after route/name filtering")
    print(f"    {rows['BDSPPatientID'].nunique():,} unique patients")

    if len(rows) == 0:
        print("  No matching rows. Skipping.")
        return

    for col in ["MedicationTakenDTS", "OrderStartDTS", "OrderEndDTS"]:
        if col in rows.columns:
            rows[col] = pd.to_datetime(rows[col], errors="coerce")

    print("  Reconstructing infusion intervals...")
    infusion_rows = rows[
        rows["InfusionRateNBR"].notna()
        & rows["MedicationTakenDTS"].notna()
    ]
    intervals = reconstruct_infusions_for_cohort(infusion_rows)
    n_int_pts = intervals["BDSPPatientID"].nunique() if len(intervals) > 0 else 0
    print(f"    {len(intervals):,} reconstructed intervals from {n_int_pts:,} patients")

    if len(intervals) > 0:
        # Drop zero/negative-duration intervals (state-machine artifacts —
        # back-to-back events at identical timestamps). These don't represent
        # real exposure.
        n_before = len(intervals)
        intervals["StartTime"] = pd.to_datetime(intervals["StartTime"], errors="coerce")
        intervals["EndTime"]   = pd.to_datetime(intervals["EndTime"], errors="coerce")
        intervals = intervals[intervals["EndTime"] > intervals["StartTime"]].copy()
        n_dropped = n_before - len(intervals)
        if n_dropped > 0:
            print(f"    Dropped {n_dropped:,} zero/negative-duration intervals")

        # Attach MedicationDSC for concentration parsing
        merge_key = (
            infusion_rows[["BDSPPatientID", "MedicationTakenDTS", "MedicationDSC"]]
            .rename(columns={"MedicationTakenDTS": "StartTime"})
        )
        intervals = intervals.merge(merge_key, on=["BDSPPatientID", "StartTime"], how="left")

        intervals["ConcentrationPerML"] = intervals["MedicationDSC"].apply(cfg["parse_concentration"])
        intervals["Rate"] = pd.to_numeric(intervals["Rate"], errors="coerce")
        intervals["ConcentrationPerML"] = pd.to_numeric(intervals["ConcentrationPerML"], errors="coerce")

        n_no_conc = intervals["ConcentrationPerML"].isna().sum()
        if n_no_conc > 0:
            print(f"    ⚠ {n_no_conc:,} intervals had no parseable concentration — DROPPING")
            failed = intervals[intervals["ConcentrationPerML"].isna()]["MedicationDSC"].value_counts()
            print(f"      Top unparseable strings:")
            for name, n in failed.head(10).items():
                print(f"        {n:>5}: {name}")
            intervals = intervals.dropna(subset=["ConcentrationPerML"])

        intervals[f"Rate_{cfg['mass_unit']}_per_hr"] = (
            intervals["Rate"] * intervals["ConcentrationPerML"]
        )

    out_path = med_config.OPIATE_EXPOSURE_DIR / f"{drug_name}_intervals.csv"
    intervals.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


try:
    med_config.OPIATE_EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading from: {MEDS_GLOB}")
    print(f"Output dir:   {med_config.OPIATE_EXPOSURE_DIR}")

    for drug in OPIATE_CONFIGS:
        process_drug(drug)

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