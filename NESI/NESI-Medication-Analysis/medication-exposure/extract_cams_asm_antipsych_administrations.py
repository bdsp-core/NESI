# extract_cams_asm_antipsych_administrations.py
"""
Extract ASM and antipsychotic admin events from the I0001 CAMS cohort.

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


def main():
    try:
        med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR.mkdir(parents=True, exist_ok=True)
        parquet_path = str(med_config.I0001_CAMS_COHORT_MEDS_PARQUET).replace("\\", "/")

        print(f"Reading: {parquet_path}")
        print(f"Output:  {med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR}\n")

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
            out_path = med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR / f"{category}_administrations.csv"
            df.to_csv(out_path, index=False)

            n_rows = len(df)
            n_pts = df["BDSPPatientID"].nunique() if n_rows else 0
            print(f"{category:<20}  rows: {n_rows:>7,}   patients: {n_pts:>5,}  →  {out_path.name}")

            summary.append({"category": category, "n_rows": n_rows, "n_patients": n_pts})

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(med_config.I0001_CAMS_ASM_ANTIPSYCH_DIR / "extraction_summary.csv", index=False)

        print("\nDone.")
        if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_OK)
    except Exception as e:
        print(f"\nERROR: {e}")
        if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_ICONHAND)
        raise


if __name__ == "__main__":
    main()