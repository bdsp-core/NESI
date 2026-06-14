# extract_i0002_asm_antipsych_administrations.py
"""
Extract all per-administration events for long-acting ASMs and antipsychotics
from the I0002 GCS cohort.

Reads from I0002_PIVOTED_MEDS_PATCHED_PARQUET. Uses ProductDescription as
the drug-name field. Includes all administration events for the cohort (no
time-window filtering).

Writes one CSV per category:
  I0002_ASM_ANTIPSYCH_DIR/{category}_administrations.csv

Paths read from med_config.py.
"""

import duckdb
import pandas as pd

import med_config
from i0002_asm_antipsych_configs import CATEGORY_MAP

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


try:
    med_config.I0002_ASM_ANTIPSYCH_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = str(med_config.I0002_PIVOTED_MEDS_PATCHED_PARQUET).replace("\\", "/")

    print(f"Reading from: {parquet_path}")
    print(f"Output dir:   {med_config.I0002_ASM_ANTIPSYCH_DIR}\n")

    summary = []
    cols_to_keep = [
        "BDSPPatientID", "MedicationTakenDTS", "ProductDescription",
        "MedicationDSC", "MedicationRouteDSC", "DiscreteDoseAMT",
        "DoseUnitDSC", "InfusionRateNBR", "InfusionRateUnitDSC",
        "MARActionDSC", "AdministrationTypesDSC",
    ]

    for category, patterns in CATEGORY_MAP.items():
        name_filter = name_filter_sql(patterns, I0002_DRUG_NAME_COL)
        df = duckdb.query(f"""
            SELECT {', '.join(cols_to_keep)}
            FROM read_parquet('{parquet_path}')
            WHERE {name_filter}
        """).df()

        out_path = med_config.I0002_ASM_ANTIPSYCH_DIR / f"{category}_administrations.csv"
        df.to_csv(out_path, index=False)

        n_rows = len(df)
        n_patients = df["BDSPPatientID"].nunique() if n_rows else 0
        print(f"{category:<20}  rows: {n_rows:>7,}   patients: {n_patients:>4,}   →  {out_path.name}")

        summary.append({
            "category": category,
            "n_rows": n_rows,
            "n_patients": n_patients,
        })

    summary_df = pd.DataFrame(summary)
    summary_path = med_config.I0002_ASM_ANTIPSYCH_DIR / "extraction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
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