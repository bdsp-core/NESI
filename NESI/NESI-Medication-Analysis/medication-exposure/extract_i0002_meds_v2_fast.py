# extract_i0002_meds_v2_fast.py
"""
Faster extraction for I0002 MarID-keyed tables that have many tiny part files.

Uses 'aws s3 sync' to parallelize downloads (vs the per-part 'aws s3 cp'
approach in extract_i0002_meds.py which is hopelessly slow for tables with
hundreds of small part files). Then filters locally with one DuckDB query.

This script only handles the MarID-keyed tables (mar_ext, mar_product_details).
Run extract_i0002_meds.py first to populate the patient-keyed tables (most
importantly mar_nax, which we need to get the cohort's MarIDs).

Paths read from med_config.py.
"""

import os
import shutil
import subprocess
import duckdb
import pandas as pd

import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


MARID_KEYED_TABLES = [
    "mar_ext_nax_2025_parquet",
    "mar_product_details_nax_2025_parquet",
]


def configure_duckdb(con):
    temp_dir = str(med_config.DUCKDB_TEMP_DIR).replace("\\", "/")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")


def sync_table_from_s3(table_name):
    """Sync entire S3 table folder to local temp dir using aws s3 sync (parallel)."""
    local_dir = med_config.I0002_EXTRACT_TEMP_DIR / table_name
    local_dir.mkdir(parents=True, exist_ok=True)
    s3_folder = f"{med_config.I0002_S3_BASE}{table_name}/"
    print(f"  Syncing {s3_folder}")
    print(f"  → {local_dir}")
    result = subprocess.run(
        f'aws s3 sync "{s3_folder}" "{local_dir}"',
        shell=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 sync failed for {table_name}")
    n_files = len(list(local_dir.glob("*.parquet")))
    print(f"  Synced {n_files} parquet files")
    return local_dir


def filter_local_table(local_dir, id_col, id_values, out_path, con):
    """Filter all local parts in one DuckDB query and write a single output parquet."""
    glob_path = str(local_dir / "*.parquet").replace("\\", "/")

    con.execute("CREATE OR REPLACE TEMP TABLE _filter_ids (id VARCHAR)")
    con.executemany(
        "INSERT INTO _filter_ids VALUES (?)",
        [(str(v),) for v in id_values],
    )

    out = str(out_path).replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT t.*
            FROM read_parquet('{glob_path}', union_by_name=true) t
            INNER JOIN _filter_ids f
              ON CAST(t.{id_col} AS VARCHAR) = f.id
        ) TO '{out}' (FORMAT PARQUET)
    """)
    n_rows = con.execute(f"SELECT COUNT(*) FROM '{out}'").fetchone()[0]
    return n_rows


try:
    med_config.DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    med_config.I0002_EXTRACT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    med_config.I0002_COHORT_MEDS_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    configure_duckdb(con)

    # Get cohort MarIDs from already-extracted mar_nax
    mar_nax_final = (
        med_config.I0002_COHORT_MEDS_DIR / "mar_nax_2025_parquet.parquet"
    )
    if not mar_nax_final.exists():
        raise FileNotFoundError(
            f"mar_nax final parquet not found at {mar_nax_final}. "
            "Run extract_i0002_meds.py first to populate patient-keyed tables."
        )

    mar_nax_path = str(mar_nax_final).replace("\\", "/")
    mar_ids = con.execute(f"""
        SELECT DISTINCT CAST(MarID AS VARCHAR) AS MarID
        FROM '{mar_nax_path}'
        WHERE MarID IS NOT NULL
    """).df()["MarID"].tolist()
    print(f"Got {len(mar_ids):,} unique cohort MarIDs from mar_nax\n")

    for table in MARID_KEYED_TABLES:
        print(f"\n── {table} ─────────────────────────")
        out_path = med_config.I0002_COHORT_MEDS_DIR / f"{table}.parquet"
        if out_path.exists():
            print(f"  Final parquet already exists at {out_path}, skipping")
            continue

        local_dir = sync_table_from_s3(table)

        print(f"  Filtering by MarID...")
        n_rows = filter_local_table(local_dir, "MarID", mar_ids, out_path, con)
        print(f"  Saved {n_rows:,} cohort rows → {out_path.name}")

        # Clean up synced temp folder to save disk
        print(f"  Cleaning up temp folder...")
        shutil.rmtree(local_dir)

    con.close()
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