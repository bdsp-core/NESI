#!/usr/bin/env python3
"""Run the cohort result scripts behind Supplementary Figs. 3 and 10-13 in run_all.py's working copy.

Uses the workspace run_all.py builds at $NESI_BUILD_ROOT/NESI.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build"))
OUT = BUILD / "output" / "figures" / "cohort"
OUT.mkdir(parents=True, exist_ok=True)
SCRIPTS = [
    "cohort_models/RASS/Results5Fld/RASS_Swimmers_plot.py",
    "cohort_models/GCS/Results5Fld/GCSSwimmersPlot.py",
    "cohort_models/RASS/Results5Fld/RASS_Results_Compare.py",
    "cohort_models/RASS/Results5Fld/RASS_results_MAE_1levelACC_Corr.py",
    "cohort_models/CAMS/Results5Fld/CAMS_Result_Compare_3class.py",
    "cohort_models/CAMS/Results5Fld/CAMS_Result_Compare_8class.py",
    "cohort_models/ICANS/Results5Fld/ICANS_Results_Compare.py",
]
# RASS_Results_Compare.py finds its root by a RASS/ folder, which now stops one level too deep
loop = BUILD / "NESI/cohort_models/cohort_models"
if not loop.exists():
    loop.symlink_to(BUILD / "NESI/cohort_models")
for rel in SCRIPTS:
    p = BUILD / "NESI" / rel
    try:
        r = subprocess.run([sys.executable, str(HERE / "_run_script.py"), str(p), str(OUT)], cwd=p.parent, capture_output=True, text=True, timeout=300, env=dict(os.environ, PYTHONPATH=str(BUILD / "stubs")))
        err = r.stderr.strip().splitlines()[-1][:150] if r.returncode and r.stderr.strip() else ""
        print(p.name, "ok" if r.returncode == 0 else "FAILED", err, flush=True)
    except subprocess.TimeoutExpired:
        print(p.name, "timed out", flush=True)
