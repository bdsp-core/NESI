# run_all_pipelines.py
"""
Reproduce the full medication exposure analysis across all four cohorts and
generate the final Table 1 outputs.

Cohorts:
  - I0001 GCS+RASS
  - I0001 CAMS
  - I0002 GCS

Final outputs:
  - exposure_summary_by_patient_final.csv
  - exposure_summary_by_eeg_final.csv
  - Table_1_medication_exposure.docx

This script DOES NOT include any S3 extraction. To reproduce from scratch
(including S3 extraction), see the Stage 0 commented sections in each
per-cohort pipeline runner.

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


# ─── Per-cohort pipelines + final reporting ──────────────────────────────────
SCRIPTS = [
    "run_i0001_gcs_rass_pipeline_meds.py",
    "run_cams_pipeline_meds.py",
    "run_i0002_pipeline_meds.py",
    "build_final_exposure_tables.py",
    "build_table_1_docx.py",
]


def main():
    script_dir = Path(__file__).parent
    overall_start = time.time()

    for i, script in enumerate(SCRIPTS, 1):
        script_path = script_dir / script
        print(f"\n{'#'*70}")
        print(f"# [{i}/{len(SCRIPTS)}] Running: {script}")
        print(f"{'#'*70}\n")
        start = time.time()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=False,
        )
        elapsed = time.time() - start
        print(f"\n[{script}] finished in {elapsed:.1f} seconds (exit code {result.returncode})")

        if result.returncode != 0:
            print(f"\n!!! Script failed: {script}")
            print("Stopping master pipeline.")
            if HAS_WINSOUND:
                winsound.MessageBeep(winsound.MB_ICONHAND)
            sys.exit(result.returncode)

    overall_elapsed = time.time() - overall_start
    print(f"\n{'#'*70}")
    print(f"# All pipelines completed in {overall_elapsed/60:.1f} minutes")
    print(f"# Final Table 1 deliverables in I0001_ASM_ANTIPSYCH_DIR")
    print(f"{'#'*70}")
    if HAS_WINSOUND:
        winsound.MessageBeep(winsound.MB_OK)


if __name__ == "__main__":
    main()