# build_bolus_exposure.py
"""
Unified bolus extraction across sedatives, opiates, and benzodiazepines.

For each drug, pulls bolus events from the cohort medication parquets,
classifies each by route, and writes one CSV per route bucket:
  <drug>_boluses_iv.csv      — IV (including PCA-as-IV)
  <drug>_boluses_non_iv.csv  — PO, IM, SQ, SL, transdermal, intranasal, etc.
Epidural and intrathecal routes are dropped entirely.

Drugs covered:
  Sedatives: propofol, midazolam, ketamine, dexmedetomidine
  Opiates:   fentanyl, morphine, hydromorphone
  Benzos:    lorazepam, diazepam, other_benzo (alprazolam, chlordiazepoxide,
             oxazepam, temazepam combined)

Reads source rows directly from EXTRACT_PER_PART_DIR (no route exclusions
applied at the query level — we need all routes here so we can split by
bucket). Bolus identification (DiscreteDoseAMT numeric, MARActionDSC in
{Given, Bolus from Bag}, InfusionRateNBR null) is delegated to
extract_boluses_for_cohort.

Paths read from med_config.py.
"""

import os
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


# Combine all drug configs into one lookup. Each entry needs:
#   name_patterns: list[str]
#   bolus_units: tuple[str]
ALL_BOLUS_DRUGS = {}
for d, cfg in DRUG_CONFIGS.items():
    ALL_BOLUS_DRUGS[d] = {
        "name_patterns": cfg["name_patterns"],
        "bolus_units": cfg["bolus_units"],
    }
for d, cfg in OPIATE_CONFIGS.items():
    # Opiate configs use mass_unit; assume boluses are charted in that unit.
    ALL_BOLUS_DRUGS[d] = {
        "name_patterns": cfg["name_patterns"],
        "bolus_units": (cfg["mass_unit"],),
    }
for d, cfg in BENZO_BOLUS_CONFIGS.items():
    ALL_BOLUS_DRUGS[d] = cfg


def name_filter_sql(patterns):
    return "(" + " OR ".join(
        f"LOWER(MedicationDSC) LIKE '%{p}%'" for p in patterns
    ) + ")"


def fetch_drug_rows(drug_name, meds_glob):
    """Pull all rows for this drug. No route exclusion — we split by route later."""
    patterns = ALL_BOLUS_DRUGS[drug_name]["name_patterns"]
    name_filter = name_filter_sql(patterns)
    return duckdb.query(f"""
        SELECT *
        FROM read_parquet('{meds_glob}')
        WHERE {name_filter}
    """).df()


def process_drug(drug_name, meds_glob):
    print(f"\n=== {drug_name} ===")
    cfg = ALL_BOLUS_DRUGS[drug_name]

    rows = fetch_drug_rows(drug_name, meds_glob)
    print(f"  Total rows: {len(rows):,}")
    print(f"  Unique patients: {rows['BDSPPatientID'].nunique():,}")

    for col in ["MedicationTakenDTS"]:
        if col in rows.columns:
            rows[col] = pd.to_datetime(rows[col], errors="coerce")

    boluses = extract_boluses_for_cohort(rows, expected_units=cfg["bolus_units"])
    print(f"  Bolus events extracted: {len(boluses):,}")

    if len(boluses) == 0:
        return {"drug": drug_name, "iv": 0, "non_iv": 0, "excluded": 0}

    boluses = add_route_bucket(boluses, rows)

    iv_df     = boluses[boluses["route_bucket"] == "iv"].drop(columns=["route_bucket"])
    non_iv_df = boluses[boluses["route_bucket"] == "non_iv"].drop(columns=["route_bucket"])
    n_excl    = (boluses["route_bucket"] == "excluded").sum()

    iv_path     = med_config.BOLUS_EXPOSURE_DIR / f"{drug_name}_boluses_iv.csv"
    non_iv_path = med_config.BOLUS_EXPOSURE_DIR / f"{drug_name}_boluses_non_iv.csv"
    iv_df.to_csv(iv_path, index=False)
    non_iv_df.to_csv(non_iv_path, index=False)

    print(f"  IV:        {len(iv_df):,}  →  {iv_path.name}")
    print(f"  Non-IV:    {len(non_iv_df):,}  →  {non_iv_path.name}")
    print(f"  Excluded:  {n_excl:,}")

    return {
        "drug": drug_name,
        "iv": len(iv_df),
        "non_iv": len(non_iv_df),
        "excluded": int(n_excl),
    }


try:
    med_config.BOLUS_EXPOSURE_DIR.mkdir(parents=True, exist_ok=True)

    meds_glob = str(med_config.EXTRACT_PER_PART_DIR / "filtered_*.parquet").replace("\\", "/")
    print(f"Reading from: {meds_glob}")
    print(f"Output dir:   {med_config.BOLUS_EXPOSURE_DIR}")

    summary = []
    for drug in ALL_BOLUS_DRUGS:
        summary.append(process_drug(drug, meds_glob))

    summary_df = pd.DataFrame(summary)
    summary_path = med_config.BOLUS_EXPOSURE_DIR / "bolus_extraction_summary.csv"
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