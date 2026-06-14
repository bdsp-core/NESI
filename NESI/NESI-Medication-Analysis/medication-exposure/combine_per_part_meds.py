# combine_per_part_meds.py
"""
Combine per-part filtered medication parquets into a single final parquet.
Reads the per-part outputs that extract_meds_for_cohort_v2.py already wrote
and merges them into COHORT_ALL_MEDS_PARQUET_NEW.

Configures DuckDB with a custom temp directory and reduced memory pressure
to avoid the out-of-memory error from the original UNION ALL combine step.

Paths read from med_config.py.
"""

import duckdb

import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


try:
    med_config.DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    part_files = sorted(med_config.EXTRACT_PER_PART_DIR.glob("filtered_*.parquet"))
    print(f"Combining {len(part_files)} per-part files")
    for p in part_files:
        size_gb = p.stat().st_size / 1e9
        print(f"  {p.name}: {size_gb:.2f} GB")

    final_out = str(med_config.COHORT_ALL_MEDS_PARQUET_NEW).replace("\\", "/")
    temp_dir = str(med_config.DUCKDB_TEMP_DIR).replace("\\", "/")

    con = duckdb.connect()

    # Configure DuckDB to use the chosen temp dir and reduce memory pressure
    con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")

    # Use read_parquet with a glob; this is more memory-efficient than UNION ALL
    glob_pattern = str(med_config.EXTRACT_PER_PART_DIR / "filtered_*.parquet").replace("\\", "/")
    print(f"\nWriting final parquet to: {final_out}")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{glob_pattern}', union_by_name=true)
        ) TO '{final_out}' (FORMAT PARQUET)
    """)

    n_total = con.execute(f"SELECT COUNT(*) FROM '{final_out}'").fetchone()[0]
    n_patients = con.execute(
        f"SELECT COUNT(DISTINCT BDSPPatientID) FROM '{final_out}'"
    ).fetchone()[0]
    con.close()

    print(f"\nFinal parquet written:")
    print(f"  Rows: {n_total:,}")
    print(f"  Unique patients: {n_patients:,}")

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