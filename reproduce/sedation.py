#!/usr/bin/env python3
"""Christine Eckhardt's medication analyses, run from the repo's code.

    python reproduce/sedation.py                # redraw figures and rebuild tables from saved results (minutes)
    python reproduce/sedation.py --from-source  # refit every model with 1,000 resamples (many hours)

Her data and saved results are in NESI_DATA_ROOT/sedation_repo, with the layout of
NESI/NESI-Medication-Analysis. The repo itself is not written to.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build"))
REPO = Path(__file__).resolve().parent.parent
PY = os.environ.get("NESI_SEDATION_PYTHON", sys.executable)
DATA = ROOT / "sedation_repo"
PARTS = ["medication-exposure", "eleveld-pk-analysis", "sedation-strata"]


def workspace():
    """Copy her code and link her data folders, so every output lands beside the data."""
    work = BUILD / "sedation"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(REPO / "NESI/NESI-Medication-Analysis", work, ignore=shutil.ignore_patterns("__pycache__"))
    for part in PARTS:
        (work / part / "data").symlink_to(DATA / part / "data")
    m, e = work / "medication-exposure/data", work / "eleveld-pk-analysis/data"
    s = (work / "medication-exposure/med_config_template.py").read_text()
    paths = {
        "I0001_ASM_ANTIPSYCH_DIR": m, "I0001_CAMS_ASM_ANTIPSYCH_DIR": m, "I0002_ASM_ANTIPSYCH_DIR": m,
        "OUTPUT_CSV": e, "OUTPUT_PARQUET": e, "ELEVELD_VALIDATION_DIR": work / "eleveld-pk-analysis/validation",
        "GCS_NESI_LOOKUP_CSV": m / "GCS_NESI_lookup.csv", "RASS_NESI_LOOKUP_CSV": m / "RASS_NESI_lookup.csv",
        "DUCKDB_TEMP_DIR": BUILD / "duckdb_tmp",
    }
    for key, val in paths.items():
        s, n = re.subn(rf'(?m)^{key}\s*=\s*Path\(r"[^"]*"\)', f'{key} = Path(r"{val}")', s)
        if n != 1:
            sys.exit(f"{key} not found in med_config_template.py")
    (work / "medication-exposure/med_config.py").write_text(s)
    return work


def run(cwd, *args):
    print(" ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-source", action="store_true")
    a = ap.parse_args()
    work = workspace()
    if a.from_source:
        run(work / "eleveld-pk-analysis", "run_variance_all.py")
        run(work / "sedation-strata", "run_stratified_all.py")
        return
    strata = work / "sedation-strata"
    run(strata, "agent_stratified_model.py", "--plots-only")
    run(strata, "build_tables.py")
    out = BUILD / "output" / "figures" / "sedation"
    out.mkdir(parents=True, exist_ok=True)
    for f in (DATA / "medication-exposure/data/stratified_feasibility/model").glob("*.png"):
        shutil.copy2(f, out)


if __name__ == "__main__":
    main()
