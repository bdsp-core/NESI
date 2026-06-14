# extract_meds_for_cohort_v2.py
"""
Resumable medication extraction for the current GCS ∪ RASS cohort (I0001).

For each part file in the I0001 Medications S3 folder:
1. Skip if a per-part output file already exists (resume from interruption)
2. Download the part file to temp dir
3. Filter to current cohort patients using DuckDB
4. Save filtered result to a per-part parquet
5. Delete the source temp file

After all parts are processed, combine the per-part outputs into a single
final parquet. The old cohort_all_medications.parquet is left untouched.

Paths read from med_config.py.
"""

import os
import glob
import subprocess
import duckdb
import pandas as pd

import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def norm_id(s):
    return (
        s.astype(str).str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .replace({"nan": None, "None": None, "": None})
    )


def load_cohort_ids():
    """Build unified GCS ∪ RASS patient list from current metadata CSVs."""
    ids = set()
    for label, files in [
        ("GCS", med_config.I0001_GCS_METADATA_CSVS),
        ("RASS", med_config.I0001_RASS_METADATA_CSVS),
    ]:
        for f in files:
            if not f.exists():
                print(f"  WARNING: missing {label} metadata file: {f}")
                continue
            df = pd.read_csv(f, low_memory=False, usecols=["BDSPPatientID"])
            n = norm_id(df["BDSPPatientID"]).dropna()
            print(f"  {f.name}: {n.nunique():,} unique patients")
            ids.update(n)
    return ids


def list_s3_parquet_files(s3_base):
    """List parquet files in the S3 medications folder, smallest first."""
    print(f"Listing S3 folder: {s3_base}")
    result = subprocess.run(
        f'aws s3 ls "{s3_base}"',
        capture_output=True, text=True, shell=True,
    )
    files = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith(".parquet"):
            files.append((parts[-1], int(parts[2])))
    files.sort(key=lambda x: x[1])
    return files


try:
    print("=== Step 1: Build cohort patient list ===")
    patient_ids = load_cohort_ids()
    print(f"\nTotal unique GCS ∪ RASS patients: {len(patient_ids):,}")

    print("\n=== Step 2: List S3 medications folder ===")
    files_with_size = list_s3_parquet_files(med_config.I0001_MEDS_S3_BASE)
    print(f"Found {len(files_with_size)} parquet files in S3")

    print("\n=== Step 3: Process each part (resumable) ===")
    med_config.EXTRACT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    med_config.EXTRACT_PER_PART_DIR.mkdir(parents=True, exist_ok=True)

    # DuckDB IN-list for cohort filter
    id_list_str = ",".join(f"'{pid}'" for pid in patient_ids)

    for i, (fname, size) in enumerate(files_with_size, start=1):
        per_part_out = med_config.EXTRACT_PER_PART_DIR / f"filtered_{fname}"
        if per_part_out.exists():
            print(f"  [{i}/{len(files_with_size)}] {fname}: already processed, skipping")
            continue

        temp_path = med_config.EXTRACT_TEMP_DIR / fname
        s3_url = med_config.I0001_MEDS_S3_BASE + fname

        print(f"  [{i}/{len(files_with_size)}] {fname} ({size/1e9:.2f} GB)")
        print(f"    Downloading...")
        subprocess.run(
            f'aws s3 cp "{s3_url}" "{temp_path}"',
            shell=True, check=True,
        )

        print(f"    Filtering...")
        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT *
                FROM '{str(temp_path).replace(chr(92), '/')}'
                WHERE CAST(BDSPPatientID AS VARCHAR) IN ({id_list_str})
            ) TO '{str(per_part_out).replace(chr(92), '/')}' (FORMAT PARQUET)
        """)
        con.close()

        n_rows = duckdb.connect().execute(
            f"SELECT COUNT(*) FROM '{str(per_part_out).replace(chr(92), '/')}'"
        ).fetchone()[0]
        print(f"    Saved {n_rows:,} filtered rows to {per_part_out.name}")

        os.remove(temp_path)
        print(f"    Deleted temp file")

    print("\n=== Step 4: Combine per-part outputs ===")
    part_files = sorted(med_config.EXTRACT_PER_PART_DIR.glob("filtered_*.parquet"))
    print(f"Combining {len(part_files)} per-part files into final parquet...")

    union_sql = " UNION ALL ".join(
        f"SELECT * FROM '{str(p).replace(chr(92), '/')}'" for p in part_files
    )
    final_out = str(med_config.COHORT_ALL_MEDS_PARQUET_NEW).replace("\\", "/")
    con = duckdb.connect()
    con.execute(f"COPY ({union_sql}) TO '{final_out}' (FORMAT PARQUET)")
    n_total = con.execute(f"SELECT COUNT(*) FROM '{final_out}'").fetchone()[0]
    n_patients = con.execute(
        f"SELECT COUNT(DISTINCT BDSPPatientID) FROM '{final_out}'"
    ).fetchone()[0]
    con.close()

    print(f"\nFinal parquet: {med_config.COHORT_ALL_MEDS_PARQUET_NEW}")
    print(f"  Rows: {n_total:,}")
    print(f"  Unique patients: {n_patients:,}")
    print(f"  Expected patients (cohort): {len(patient_ids):,}")

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