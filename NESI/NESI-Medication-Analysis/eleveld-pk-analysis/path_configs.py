# path_configs.py
"""
Path configuration for the Eleveld PK analysis pipeline.

All paths resolve relative to the location of this file. Clone the repo,
run the scripts — no edits needed.

Repo layout:
    repo-root/
    +-- path_configs.py        <- this file
    +-- scripts/               <- analysis .py files
    +-- data/                  <- shipped inputs (de-identified)
    |   +-- wt_ht_per_hosp_FULL_with_eleveld.csv
    |   +-- sedative_exposures/
    |       +-- propofol_intervals.csv
    |       +-- propofol_boluses.csv
    +-- outputs/               <- generated artifacts (created on first run)
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR    = REPO_ROOT / 'data'
OUTPUTS_DIR = REPO_ROOT / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Inputs (shipped in data/)
# ============================================================
PROPOFOL_INTERVALS_CSV = str(DATA_DIR / 'sedative_exposures' / 'propofol_intervals.csv')
PROPOFOL_BOLUSES_CSV   = str(DATA_DIR / 'sedative_exposures' / 'propofol_boluses.csv')


# ============================================================
# Folders the scripts use
# ============================================================
# OUTPUT_CSV: where the wt_ht_per_hosp_FULL_with_eleveld.csv input lives.
# Analysis scripts read it from here.
OUTPUT_CSV     = str(DATA_DIR)

# Where generated artifacts (markdown reports, plots, parquets) go.
OUTPUT_PARQUET = str(OUTPUTS_DIR)
OUTPUT_MD      = str(OUTPUTS_DIR)
OUTPUT_PNG     = str(OUTPUTS_DIR)


# ============================================================
# Diagnostic (run `python path_configs.py` to verify paths)
# ============================================================
if __name__ == '__main__':
    print(f"REPO_ROOT:    {REPO_ROOT}")
    print(f"DATA_DIR:     {DATA_DIR}   (exists: {DATA_DIR.exists()})")
    print(f"OUTPUTS_DIR:  {OUTPUTS_DIR} (exists: {OUTPUTS_DIR.exists()})")
    print()
    print(f"PROPOFOL_INTERVALS_CSV: {PROPOFOL_INTERVALS_CSV}")
    print(f"  exists: {Path(PROPOFOL_INTERVALS_CSV).exists()}")
    print(f"PROPOFOL_BOLUSES_CSV:   {PROPOFOL_BOLUSES_CSV}")
    print(f"  exists: {Path(PROPOFOL_BOLUSES_CSV).exists()}")
    wtht = Path(OUTPUT_CSV) / 'wt_ht_per_hosp_FULL_with_eleveld.csv'
    print(f"WTHT with Eleveld:      {wtht}")
    print(f"  exists: {wtht.exists()}")