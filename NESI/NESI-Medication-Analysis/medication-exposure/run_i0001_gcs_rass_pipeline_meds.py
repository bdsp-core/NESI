# run_i0001_gcs_rass_pipeline.py
"""
Reproduce the I0001 GCS+RASS medication exposure pipeline.

REQUIRED INPUTS:
  - I0001 GCS metadata CSVs at I0001_GCS_METADATA_CSVS
  - I0001 RASS metadata CSVs at I0001_RASS_METADATA_CSVS
  - I0001 medications data on S3 at I0001_MEDS_S3_BASE

PIPELINE STAGES:
  Stage 0 (ONE-TIME, COMMENTED OUT): Extract cohort medications from S3.
    Takes several hours; produces COHORT_ALL_MEDS_PARQUET_NEW.
    Uncomment to re-extract.
  Stage 1+: Downstream processing on the cohort medications parquet.

To reproduce from scratch:
  1. Fill in paths in med_config.py
  2. Uncomment and run Stage 0 (one-time, hours)
  3. Run this script — it executes Stages 1+ (minutes)

Paths read from med_config.py.
"""

import subprocess
import sys
import time
from pathlib import Path

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# ─── Stage 0: S3 extraction (one-time; uncomment to re-extract) ──────────────
# SCRIPTS_STAGE_0 = [
#     "extract_meds_for_cohort_v2.py",       # parallel S3 download + filter
#     "combine_per_part_meds.py",            # combine per-part filtered files
# ]

# ─── Stage 1+: Downstream processing ─────────────────────────────────────────
SCRIPTS = [
    "build_sedative_exposure.py",
    "build_opiate_exposure.py",
    "build_bolus_exposure.py",
    "extract_i0001_asm_antipsych_administrations.py",
    "build_i0001_asm_antipsych_exposure.py",
    "build_i0001_master_exposure.py",
]


def main():
    script_dir = Path(__file__).parent
    overall_start = time.time()

    for i, script in enumerate(SCRIPTS, 1):
        script_path = script_dir / script
        print(f"\n{'='*70}")
        print(f"[{i}/{len(SCRIPTS)}] Running: {script}")
        print(f"{'='*70}\n")
        start = time.time()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=False,
        )
        elapsed = time.time() - start
        print(f"\n[{script}] finished in {elapsed:.1f} seconds (exit code {result.returncode})")

        if result.returncode != 0:
            print(f"\n!!! Script failed: {script}")
            print("Stopping pipeline.")
            if HAS_WINSOUND:
                winsound.MessageBeep(winsound.MB_ICONHAND)
            sys.exit(result.returncode)

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"All scripts completed successfully in {overall_elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    if HAS_WINSOUND:
        winsound.MessageBeep(winsound.MB_OK)


if __name__ == "__main__":
    main()