# combine_cams_per_part_meds.py
"""
Combine all per-part filtered parquets in I0001_CAMS_EXTRACT_PER_PART_DIR
into a single I0001_CAMS_COHORT_MEDS_PARQUET.

Same pattern as combine_per_part_meds.py (used for I0001 GCS/RASS) — uses
DuckDB glob read with bounded memory and a custom temp dir.

Paths read from med_config.py.
"""

import duckdb
import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def main():
    try:
        glob_pattern = str(
            med_config.I0001_CAMS_EXTRACT_PER_PART_DIR / "filtered_*.parquet"
        ).replace("\\", "/")
        out_path = str(med_config.I0001_CAMS_COHORT_MEDS_PARQUET).replace("\\", "/")

        med_config.I0001_CAMS_COHORT_MEDS_PARQUET.parent.mkdir(parents=True, exist_ok=True)

        print(f"Reading: {glob_pattern}")
        print(f"Writing: {out_path}\n")

        con = duckdb.connect()
        temp_dir = str(med_config.DUCKDB_TEMP_DIR).replace("\\", "/")
        con.execute(f"SET temp_directory = '{temp_dir}'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET memory_limit = '8GB'")
        con.execute("SET threads = 4")

        con.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{glob_pattern}', union_by_name=true)
            ) TO '{out_path}' (FORMAT PARQUET)
        """)

        n_rows = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
        n_pts  = con.execute(f"SELECT COUNT(DISTINCT BDSPPatientID) FROM '{out_path}'").fetchone()[0]
        print(f"Final parquet: {n_rows:,} rows, {n_pts:,} patients")
        con.close()
        print("\nDone.")
        if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_OK)
        else:
            try: print("\a")
            except Exception: pass
    except Exception as e:
        print(f"\nERROR: {e}")
        if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_ICONHAND)
        raise


if __name__ == "__main__":
    main()