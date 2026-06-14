# extract_meds_for_cams_cohort.py
"""
Extract cohort-filtered medication rows from I0001 medications S3 for the CAMS
cohort, using aws s3 sync for parallel downloads and parallel filtering.

Two phases:
  1. Download all S3 part files locally in one shot via aws s3 sync
     (parallelized by aws CLI, ~10 concurrent connections)
  2. Filter each local part file in parallel via DuckDB, writing one filtered
     parquet per part

Resumable: skips parts whose filtered output already exists.

Paths read from med_config.py.
"""

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


N_PARALLEL_FILTERS = 3   # tuned for 32GB system with other apps running


def norm_id(s):
    return (
        s.astype(str).str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .replace({"nan": None, "None": None, "": None})
    )


def filter_one_part(args):
    """Worker: filter one local parquet by cohort IDs."""
    local_path, out_path, id_list_sql, duckdb_temp = args
    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory = '{duckdb_temp}'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET memory_limit = '6GB'")
        con.execute("SET threads = 2")
        con.execute(f"""
            COPY (
                SELECT *
                FROM read_parquet('{str(local_path).replace(chr(92), '/')}')
                WHERE CAST(BDSPPatientID AS VARCHAR) IN ({id_list_sql})
            ) TO '{str(out_path).replace(chr(92), '/')}' (FORMAT PARQUET)
        """)
        n_rows = con.execute(
            f"SELECT COUNT(*) FROM '{str(out_path).replace(chr(92), '/')}'"
        ).fetchone()[0]
        return out_path.name, n_rows, None
    except Exception as e:
        return out_path.name, 0, str(e)
    finally:
        con.close()


def main():
    try:
        # ── Load CAMS cohort patient IDs ──────────────────────────────────────
        print("Loading CAMS cohort patient IDs...")
        pieces = []
        for f in med_config.I0001_CAMS_METADATA_CSVS:
            pieces.append(pd.read_csv(str(f), usecols=["BDSPPatientID"], low_memory=False))
        meta = pd.concat(pieces, ignore_index=True)
        cohort_ids = set(norm_id(meta["BDSPPatientID"]).dropna())
        print(f"  Cohort patients: {len(cohort_ids):,}\n")
        id_list_sql = ",".join(f"'{p}'" for p in cohort_ids)

        # ── Prepare directories ───────────────────────────────────────────────
        med_config.I0001_CAMS_EXTRACT_PER_PART_DIR.mkdir(parents=True, exist_ok=True)
        download_dir = med_config.I0001_CAMS_EXTRACT_PER_PART_DIR / "_downloads"
        download_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: aws s3 sync (parallel download) ──────────────────────────
        print(f"Phase 1: Syncing from {med_config.I0001_MEDS_S3_BASE}")
        print(f"  to {download_dir}")
        print("  (this may take a while for first run; subsequent runs only sync new files)")
        subprocess.run(
            [
                "aws", "s3", "sync",
                str(med_config.I0001_MEDS_S3_BASE),
                str(download_dir),
                "--exclude", "*",
                "--include", "*.parquet",
            ],
            check=True,
        )
        parts = sorted(download_dir.glob("*.parquet"))
        print(f"  {len(parts)} parquet files local\n")

        # ── Phase 2: parallel filtering ───────────────────────────────────────
        print(f"Phase 2: Filtering {len(parts)} files with {N_PARALLEL_FILTERS} parallel workers")

        duckdb_temp = str(med_config.DUCKDB_TEMP_DIR).replace("\\", "/")

        jobs = []
        n_skipped = 0
        for part in parts:
            out_path = med_config.I0001_CAMS_EXTRACT_PER_PART_DIR / f"filtered_{part.name}"
            if out_path.exists():
                n_skipped += 1
                continue
            jobs.append((part, out_path, id_list_sql, duckdb_temp))

        if n_skipped:
            print(f"  {n_skipped} parts already filtered, skipping")
        print(f"  {len(jobs)} parts to process\n")

        n_done = 0
        if jobs:
            with ProcessPoolExecutor(max_workers=N_PARALLEL_FILTERS) as ex:
                futures = {ex.submit(filter_one_part, job): job for job in jobs}
                for fut in as_completed(futures):
                    name, n_rows, err = fut.result()
                    n_done += 1
                    if err:
                        print(f"  [{n_done}/{len(jobs)}] {name}: ERROR — {err}")
                    else:
                        print(f"  [{n_done}/{len(jobs)}] {name}: {n_rows:,} cohort rows")

        print(f"\n=== Summary ===")
        print(f"  Filtered: {n_done}")
        print(f"  Skipped (already done): {n_skipped}")
        print(f"\nLocal copies of source files remain in:")
        print(f"  {download_dir}")
        print(f"  (~50 GB; delete manually when finished if no longer needed)")
        print(f"\nNext step: combine per-part filtered files into "
              f"I0001_CAMS_COHORT_MEDS_PARQUET via combine_per_part_meds.py pattern.")

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


if __name__ == '__main__':
    main()