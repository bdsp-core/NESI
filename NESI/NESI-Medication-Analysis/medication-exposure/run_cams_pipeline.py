# run_cams_pipeline.py
"""
Run the complete CAMS downstream pipeline (assumes combine has already run
to produce I0001_CAMS_COHORT_MEDS_PARQUET).

Runs scripts in order, stops on any error.
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


SCRIPTS = [
    "build_cams_sedative_exposure.py",
    "build_cams_opiate_exposure.py",
    "build_cams_bolus_exposure.py",
    "extract_cams_asm_antipsych_administrations.py",
    "build_cams_asm_antipsych_exposure.py",
    "build_cams_master_exposure.py",
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
            if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_ICONHAND)
            sys.exit(result.returncode)

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"All scripts completed successfully in {overall_elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    if HAS_WINSOUND: winsound.MessageBeep(winsound.MB_OK)


if __name__ == "__main__":
    main()