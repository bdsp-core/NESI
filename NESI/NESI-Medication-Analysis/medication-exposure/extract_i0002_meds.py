# extract_i0002_meds.py
"""
Resumable medication extraction for the I0002 GCS cohort.

Two-pass approach:
  Pass 1 — patient-keyed tables (have BDSPPatientID):
    For each table, for each S3 part file:
      - Skip if per-part output already exists
      - Download
      - Filter to cohort patients
      - Save filtered parquet
      - Delete temp file

  Pass 2 — MarID-keyed tables (only have MarID):
    After Pass 1 completes, read all MarIDs from the cohort-filtered mar_nax,
    then for each MarID-keyed table do the same per-part loop, filtering by
    MarID instead of patient ID.

Final step: combine per-part outputs into one parquet per table.

Resumable: rerun if interrupted; existing per-part outputs are skipped.

Paths read from med_config.py.
"""

import os
import subprocess
import duckdb
import pandas as pd

import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


PATIENT_KEYED_TABLES = [
    "mar_nax_2025_parquet",
    "mar_details_nax_2025_parquet",
    "mar_product_nax_2025_parquet",
    "medication_nax_2024_parquet",
    "poe_order_med_nax_2025_parquet",
    "poe_order_nax_2025_parquet",
    "poe_order_details_nax_2025_parquet",
    "inpt_paml_meds_nax_2025_parquet",
]

MARID_KEYED_TABLES = [
    "mar_ext_nax_2025_parquet",
    "mar_product_details_nax_2025_parquet",
]


def norm_id(s):
    return (
        s.astype(str).str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .replace({"nan": None, "None": None, "": None})
    )


def load_cohort_ids():
    """Load BDSPPatientIDs from the I0002 GCS metadata CSV."""
    df = pd.read_csv(
        med_config.I0002_GCS_METADATA_CSV,
        low_memory=False, usecols=["BDSPPatientID"],
    )
    ids = set(norm_id(df["BDSPPatientID"]).dropna())
    return ids


def list_s3_part_files(s3_folder):
    """List parquet files in S3 folder, smallest first."""
    result = subprocess.run(
        f'aws s3 ls "{s3_folder}"',
        capture_output=True, text=True, shell=True,
    )
    files = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith(".parquet"):
            files.append((parts[-1], int(parts[2])))
    files.sort(key=lambda x: x[1])
    return files


def configure_duckdb(con):
    """Configure DuckDB with sane temp dir and memory limits."""
    temp_dir = str(med_config.DUCKDB_TEMP_DIR).replace("\\", "/")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")


def filter_part_by_id(part_path, out_path, id_col, id_values, con):
    """
    Filter one downloaded part file by an ID column using a DuckDB temp table.
    Faster than IN-list with thousands of literals.
    """
    # Build temp table of IDs
    con.execute("CREATE OR REPLACE TEMP TABLE _filter_ids (id VARCHAR)")
    con.executemany(
        "INSERT INTO _filter_ids VALUES (?)",
        [(str(v),) for v in id_values],
    )
    src = str(part_path).replace("\\", "/")
    dst = str(out_path).replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT t.*
            FROM '{src}' t
            INNER JOIN _filter_ids f
              ON CAST(t.{id_col} AS VARCHAR) = f.id
        ) TO '{dst}' (FORMAT PARQUET)
    """)


def process_table(table_name, id_col, id_values, con):
    """Run the per-part download/filter loop for one table."""
    print(f"\n── {table_name} (filtering by {id_col}) ─────────────────────────")

    per_part_dir = med_config.I0002_EXTRACT_PER_PART_DIR / table_name
    per_part_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = med_config.I0002_EXTRACT_TEMP_DIR / table_name
    temp_dir.mkdir(parents=True, exist_ok=True)

    s3_folder = f"{med_config.I0002_S3_BASE}{table_name}/"
    parts = list_s3_part_files(s3_folder)
    print(f"  {len(parts)} parquet files in S3")

    for i, (fname, size) in enumerate(parts, start=1):
        per_part_out = per_part_dir / f"filtered_{fname}"
        if per_part_out.exists():
            print(f"  [{i}/{len(parts)}] {fname}: already processed, skipping")
            continue

        temp_path = temp_dir / fname
        s3_url = f"{s3_folder}{fname}"
        print(f"  [{i}/{len(parts)}] {fname} ({size/1e9:.2f} GB)")

        # Download
        print(f"    Downloading...")
        result = subprocess.run(
            f'aws s3 cp "{s3_url}" "{temp_path}"',
            shell=True, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    Download failed: {result.stderr}")
            continue

        # Filter
        print(f"    Filtering...")
        try:
            filter_part_by_id(temp_path, per_part_out, id_col, id_values, con)
        except Exception as e:
            print(f"    Filter failed: {e}")
            os.remove(temp_path)
            continue

        # Row count
        n_rows = con.execute(
            f"SELECT COUNT(*) FROM '{str(per_part_out).replace(chr(92), '/')}'"
        ).fetchone()[0]
        print(f"    Saved {n_rows:,} rows")

        # Cleanup
        os.remove(temp_path)


def combine_per_part(table_name, con):
    """Combine per-part outputs into one final parquet per table."""
    per_part_dir = med_config.I0002_EXTRACT_PER_PART_DIR / table_name
    parts = sorted(per_part_dir.glob("filtered_*.parquet"))
    if not parts:
        print(f"  {table_name}: no per-part files, skipping combine")
        return None

    final_path = med_config.I0002_COHORT_MEDS_DIR / f"{table_name}.parquet"
    glob_path = str(per_part_dir / "filtered_*.parquet").replace("\\", "/")
    out = str(final_path).replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{glob_path}', union_by_name=true)
        ) TO '{out}' (FORMAT PARQUET)
    """)
    n_total = con.execute(f"SELECT COUNT(*) FROM '{out}'").fetchone()[0]
    print(f"  {table_name}: combined → {final_path.name} ({n_total:,} rows)")
    return final_path


try:
    med_config.DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    med_config.I0002_EXTRACT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    med_config.I0002_EXTRACT_PER_PART_DIR.mkdir(parents=True, exist_ok=True)
    med_config.I0002_COHORT_MEDS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Load I0002 GCS cohort ===")
    cohort_ids = load_cohort_ids()
    print(f"{len(cohort_ids):,} unique GCS patients")

    con = duckdb.connect()
    configure_duckdb(con)

    # ── Pass 1: Patient-keyed tables ──────────────────────────────────────────
    print("\n=== Step 2: Extract patient-keyed tables ===")
    for table in PATIENT_KEYED_TABLES:
        process_table(table, "BDSPPatientID", cohort_ids, con)

    # ── Combine patient-keyed tables (needed for Pass 2) ─────────────────────
    print("\n=== Step 3: Combine patient-keyed tables ===")
    for table in PATIENT_KEYED_TABLES:
        combine_per_part(table, con)

    # ── Pass 2: collect MarIDs from cohort-filtered mar_nax ──────────────────
    print("\n=== Step 4: Collect MarIDs from mar_nax ===")
    mar_nax_path = str(
        med_config.I0002_COHORT_MEDS_DIR / "mar_nax_2025_parquet.parquet"
    ).replace("\\", "/")
    mar_ids = con.execute(f"""
        SELECT DISTINCT CAST(MarID AS VARCHAR) AS MarID
        FROM '{mar_nax_path}'
        WHERE MarID IS NOT NULL
    """).df()["MarID"].tolist()
    print(f"{len(mar_ids):,} unique MarIDs for cohort")

    # ── Pass 2: MarID-keyed tables ───────────────────────────────────────────
    print("\n=== Step 5: Extract MarID-keyed tables ===")
    for table in MARID_KEYED_TABLES:
        # mar_product_details uses MarID too
        process_table(table, "MarID", mar_ids, con)

    print("\n=== Step 6: Combine MarID-keyed tables ===")
    for table in MARID_KEYED_TABLES:
        combine_per_part(table, con)

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