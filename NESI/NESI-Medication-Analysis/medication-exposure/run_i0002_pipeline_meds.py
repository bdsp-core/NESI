# run_i0002_pipeline.py
"""
Reproduce the I0002 GCS medication exposure pipeline.

REQUIRED INPUTS:
  - I0002 GCS metadata CSV at I0002_GCS_METADATA_CSV
  - I0002 medications data on S3 (mar_details, mar_nax)

PIPELINE STAGES:
  Stage 0 (ONE-TIME, COMMENTED OUT): Extract MAR data from S3, pivot the EAV
    table, patch MARActionDSC values. Produces I0002_PIVOTED_MEDS_PATCHED_PARQUET.
    Uncomment to re-extract.
  Stage 1+: Downstream processing on the patched parquet.

To reproduce from scratch:
  1. Fill in paths in med_config.py
  2. Uncomment and run Stage 0 (one-time, hours due to large mar_details table)
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


# ─── Stage 0: S3 extraction + EAV pivot (one-time; uncomment to re-extract) ──
# SCRIPTS_STAGE_0 = [
#     "extract_i0002_meds.py",               # initial S3 sync
#     "extract_i0002_meds_v2_fast.py",       # parallel S3 sync of mar_ext parts
#     "build_i0002_pivoted_meds.py",         # EAV → wide pivot, join mar_nax
#     "patch_i0002_pivoted_meds.py",         # fill null MARActionDSC from codes
# ]

# ─── Stage 1+: Downstream processing ─────────────────────────────────────────
SCRIPTS = [
    "build_i0002_benzo_boluses.py",
    "build_i0002_propofol_boluses.py",
    "build_i0002_opiate_boluses.py",
    "extract_i0002_asm_antipsych_administrations.py",
    "build_i0002_asm_antipsych_exposure.py",
    "build_i0002_master_exposure.py",
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