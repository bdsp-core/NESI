# extract_i0001_asm_antipsych_administrations.py
"""
Extract all per-administration events for long-acting ASMs and antipsychotics
from the I0001 cohort (GCS + RASS combined).

Reads from COHORT_ALL_MEDS_PARQUET_NEW. Primary drug-name field is
MedicationDSC (Epic-style I0001 data, well populated); also searches
MedicationDisplayNM as a fallback for any rows where MedicationDSC is sparse.

Writes one CSV per category:
  I0001_ASM_ANTIPSYCH_DIR/{category}_administrations.csv

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


def name_filter_sql(patterns):
    return "(" + " OR ".join(
        f"LOWER(MedicationDSC) LIKE '%{p}%' OR LOWER(MedicationDisplayNM) LIKE '%{p}%'"
        for p in patterns
    ) + ")"


try:
    med_config.I0001_ASM_ANTIPSYCH_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = str(med_config.COHORT_ALL_MEDS_PARQUET_NEW).replace("\\", "/")

    print(f"Reading from: {parquet_path}")
    print(f"Output dir:   {med_config.I0001_ASM_ANTIPSYCH_DIR}\n")

    cols_to_keep = [
        "BDSPPatientID", "MedicationTakenDTS", "MedicationDSC", "MedicationDisplayNM",
        "MedicationRouteDSC", "DiscreteDoseAMT", "DoseUnitDSC",
        "InfusionRateNBR", "InfusionRateUnitDSC", "MARActionDSC",
    ]

    summary = []
    for category, patterns in CATEGORY_MAP.items():
        name_filter = name_filter_sql(patterns)
        df = duckdb.query(f"""
            SELECT {', '.join(cols_to_keep)}
            FROM read_parquet('{parquet_path}')
            WHERE {name_filter}
        """).df()

        out_path = med_config.I0001_ASM_ANTIPSYCH_DIR / f"{category}_administrations.csv"
        df.to_csv(out_path, index=False)

        n_rows = len(df)
        n_patients = df["BDSPPatientID"].nunique() if n_rows else 0
        print(f"{category:<20}  rows: {n_rows:>7,}   patients: {n_patients:>5,}   →  {out_path.name}")

        summary.append({
            "category": category,
            "n_rows": n_rows,
            "n_patients": n_patients,
        })

    summary_df = pd.DataFrame(summary)
    summary_path = med_config.I0001_ASM_ANTIPSYCH_DIR / "extraction_summary.csv"
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