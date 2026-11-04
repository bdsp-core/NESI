# run_variance_all.py
"""
Propofol exposure-response analyses for the revision, in order.

  1. Primary variance decomposition, propofol-only cohort (Ce and pump rate)
  2. Sensitivity: excluding intermittent benzodiazepine or opioid doses on board
  3. Within- vs between-patient Ce slopes (propofol-only cohort)
  4. Expanded cohort, propofol with or without opioids (Ce only, + opioid, + Ce x opioid)
  5. Spearman rho of Ce with NESI and RASS, propofol-only cohort (same rows as step 1)
  6. Spearman rho of Ce with NESI and RASS, cohort with opioid infusions allowed (same rows as step 4)

Each step is the same script with a flag; no settings are edited. Outputs are the
markdown reports written by each script (paths printed as they run).

Usage (from eleveld-pk-analysis/):
    python run_variance_all.py               # full run, 1000 resamples each (several hours)
    python run_variance_all.py --n-boot 2    # smoke test, a few minutes
    python run_variance_all.py --only 4      # one step
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("Primary: Ce and pump rate",              "variance_decomposition_eleveld_ce_vs_pump.py", []),
    ("Excluding intermittent doses",           "variance_decomposition_eleveld_ce_vs_pump.py", ["--exclude-intermittent"]),
    ("Within- vs between-patient Ce",          "variance_decomposition_eleveld_ce_vs_pump.py", ["--within-between"]),
    ("Expanded cohort with opioids",           "variance_decomposition_expanded_opioid.py",    []),
    ("Rank correlation: propofol only",        "variance_decomposition_eleveld_ce_vs_pump.py", ["--correlations"]),
    ("Rank correlation: opioids allowed",      "variance_decomposition_expanded_opioid.py",    ["--correlations"]),
]

ap = argparse.ArgumentParser()
ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples per step (default 1000)")
ap.add_argument("--only", type=int, nargs="*", help="step numbers to run (1-6); default all")
args = ap.parse_args()

for i, (name, script, flags) in enumerate(STEPS, start=1):
    if args.only and i not in args.only:
        continue
    cmd = [sys.executable, str(HERE / script)] + flags + \
          (["--n-boot", str(args.n_boot)] if args.n_boot else [])
    print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {name}\n  {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"Step {i} ({name}) failed; stopping.")
    print(f"[{i}] done in {(time.time() - t0) / 60:.1f} min", flush=True)

print("\nAll variance analyses done.")
