# run_stratified_all.py
"""
Sedation-stratified analyses for the revision, in order.

  1. build_strata.py           NESI + exposure flags -> one row per RASS assessment with its stratum
  2. build_strata_labels.py    sedation labels for every RASS, GCS and CAM-S assessment
                               (input for the performance and mortality analyses)
  3. agent_stratified_model.py primary (uncapped, primary strata): models, report, all figures
  4.   ... --cap p95           sensitivity: at most K obs per patient per stratum (K = p95 in reference)
  5.   ... --strata strict     sensitivity: intermittent opioid doses also exclude
  6. build_tables.py           main-text propofol table + Supplementary Tables S1-S6 (Word + CSV)
  7. md_to_html.py             HTML copies of the model reports

Run eleveld-pk-analysis/run_variance_all.py first: build_tables.py reads its reports.
Each step is the same script with a flag; no settings are edited.

Usage (from sedation-strata/):
    python run_stratified_all.py               # full run, 1000 resamples per model run (hours)
    python run_stratified_all.py --n-boot 2    # smoke test
    python run_stratified_all.py --only 6 7    # e.g. rebuild tables and HTML only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOT = "boot"   # marker: step takes --n-boot
STEPS = [
    ("Build strata",                         "build_strata.py",           []),
    ("Sedation labels (RASS, GCS, CAM-S)",   "build_strata_labels.py",    []),
    ("Stratified model: primary",            "agent_stratified_model.py", [BOOT]),
    ("Stratified model: cap p95",            "agent_stratified_model.py", ["--cap", "p95", BOOT]),
    ("Stratified model: strict rule",        "agent_stratified_model.py", ["--strata", "strict", BOOT]),
    ("Tables (main text + supplement)",      "build_tables.py",           []),
    ("HTML reports",                         "md_to_html.py",             []),
]

ap = argparse.ArgumentParser()
ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples per model run (default 1000)")
ap.add_argument("--only", type=int, nargs="*", help="step numbers to run (1-7); default all")
args = ap.parse_args()

for i, (name, script, flags) in enumerate(STEPS, start=1):
    if args.only and i not in args.only:
        continue
    extra = [f for f in flags if f != BOOT]
    if BOOT in flags and args.n_boot:
        extra += ["--n-boot", str(args.n_boot)]
    cmd = [sys.executable, str(HERE / script)] + extra
    print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {name}\n  {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"Step {i} ({name}) failed; stopping.")
    print(f"[{i}] done in {(time.time() - t0) / 60:.1f} min", flush=True)

print("\nAll stratified analyses done.")
