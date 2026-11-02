# propofol_benzo_count.py
"""
Propofol cohort: how many observations have intermittent sedative/opioid doses on board?
Counts only (no NESI modeling). Decides whether to exclude them in the propofol analyses,
matching the agent strata.

Cohort: same filters as the primary propofol analysis
  propofol_only_recomputed == True, PropofolSimStatus == "simulated",
  Ce + NESI + RASS present, RASS -5 to 0, >= MIN_OBS_PER_PATIENT obs per patient.
Flags come from the exposure windows file, joined on patient + RASS time:
  benzo_intermittent  = benzo_on_board == 1 and midazolam_inf_now == 0
  opioid_intermittent = opioid_on_board == 1 and opioid_inf_now == 0
  either              = combined "no intermittent sedative or opioid doses" rule

Edit IN_CSV if the cohort lives in the other file; the script lists candidate columns
if anything is missing. Run with Play. Paths are printed.
"""

import os
import sys

import pandas as pd

# Repo layout: med_config.py lives in ../medication-exposure (no effect if already importable)
import sys as _sys
from pathlib import Path as _Path
_MC = _Path(__file__).resolve().parent.parent / "medication-exposure"
if _MC.is_dir() and str(_MC) not in _sys.path:
    _sys.path.insert(0, str(_MC))
from med_config import OUTPUT_CSV, I0001_ASM_ANTIPSYCH_DIR

# ============ PATHS ============
IN_CSV = os.path.join(OUTPUT_CSV, "wt_ht_per_hosp_FULL_with_eleveld_expanded.csv")
WINDOWS_CSV = os.path.join(I0001_ASM_ANTIPSYCH_DIR, "i0001_per_eeg_master_exposure_windows_opiates.csv")

# ============ CONFIG ============
PATIENT_COL = "BDSPPatientID"
TIME_COL = "RASSRecordedDTS"
COHORT_FLAG = "propofol_only_recomputed"
SIM_STATUS_COL = "PropofolSimStatus"
CE_COL = "Ce_ug_per_mL"
NESI_COL = "PredictedNESI"
RASS_COL = "R PHS IP RASS"
MIN_OBS_PER_PATIENT = 5
WIN_FLAGS = ["benzo_on_board", "midazolam_inf_now", "opioid_on_board", "opioid_inf_now"]


def norm_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def need(path, cols):
    head = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in cols if c not in head]
    if missing:
        print(f"\n{os.path.basename(path)}: columns not found -> {missing}")
        for t in ["propofol", "nesi", "ce_", "rass", "dts", "board", "inf_now", "status"]:
            hits = [c for c in head if t in c.lower()]
            if hits:
                print(f"  '{t}': {hits[:10]}")
        sys.exit("Edit the column names at the top and run again.")


print("INPUTS")
print(f"  cohort  : {os.path.abspath(IN_CSV)}")
print(f"  windows : {os.path.abspath(WINDOWS_CSV)}\n")
for p in (IN_CSV, WINDOWS_CSV):
    if not os.path.exists(p):
        sys.exit(f"File not found: {os.path.abspath(p)}")

cohort_cols = [PATIENT_COL, TIME_COL, COHORT_FLAG, SIM_STATUS_COL, CE_COL, NESI_COL, RASS_COL]
need(IN_CSV, cohort_cols)
need(WINDOWS_CSV, [PATIENT_COL, TIME_COL] + WIN_FLAGS)

# ---- cohort, same filters as the primary analysis
d = pd.read_csv(IN_CSV, usecols=cohort_cols, low_memory=False)
flag = d[COHORT_FLAG].astype(str).str.lower().isin(["true", "1"])
d = d[flag & (d[SIM_STATUS_COL] == "simulated")].copy()
for c in (CE_COL, NESI_COL, RASS_COL):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=[CE_COL, NESI_COL, RASS_COL])
d = d[d[RASS_COL].between(-5, 0)]
n_per = d.groupby(PATIENT_COL).size()
d = d[d[PATIENT_COL].isin(n_per[n_per >= MIN_OBS_PER_PATIENT].index)].copy()
d["_pid"] = norm_id(d[PATIENT_COL])
d["_t"] = pd.to_datetime(d[TIME_COL], errors="coerce")
print(f"Propofol cohort (primary filters): {len(d):,} observations, {d['_pid'].nunique():,} patients")

# ---- exposure flags at the RASS time
w = pd.read_csv(WINDOWS_CSV, usecols=[PATIENT_COL, TIME_COL] + WIN_FLAGS, low_memory=False)
w["_pid"] = norm_id(w[PATIENT_COL])
w["_t"] = pd.to_datetime(w[TIME_COL], errors="coerce")
w = w.dropna(subset=["_t"]).drop_duplicates(["_pid", "_t"])
m = d.merge(w[["_pid", "_t"] + WIN_FLAGS], on=["_pid", "_t"], how="left", indicator=True)
matched = m["_merge"] == "both"
print(f"Matched to exposure flags: {matched.sum():,} of {len(m):,} ({100 * matched.mean():.1f}%)")
m = m[matched].copy()
unknown = m[WIN_FLAGS].isna().any(axis=1)
print(f"Unknown medication status (no MAR data): {unknown.sum():,}")
m = m[~unknown]
f = m[WIN_FLAGS].astype(int)

groups = {
    "Intermittent benzo dose (no midazolam infusion)": (f["benzo_on_board"] == 1) & (f["midazolam_inf_now"] == 0),
    "Midazolam infusion running": f["midazolam_inf_now"] == 1,
    "Intermittent opioid dose (no opioid infusion)": (f["opioid_on_board"] == 1) & (f["opioid_inf_now"] == 0),
    "Opioid infusion running": f["opioid_inf_now"] == 1,
}
groups["Either intermittent dose (combined rule excludes)"] = groups[list(groups)[0]] | groups[list(groups)[2]]

rows = []
for name, mask in groups.items():
    mask = mask.values
    rows.append({"group": name, "observations": int(mask.sum()),
                 "% of observations": round(100 * mask.mean(), 1),
                 "patients": m.loc[mask, "_pid"].nunique(),
                 "% of patients": round(100 * m.loc[mask, "_pid"].nunique() / m["_pid"].nunique(), 1)})
tab = pd.DataFrame(rows)
print(f"\nAmong {len(m):,} observations / {m['_pid'].nunique():,} patients with known medication status:")
print(tab.to_string(index=False))

keep = ~groups["Either intermittent dose (combined rule excludes)"].values
print(f"\nCombined-rule cohort would keep {keep.sum():,} observations, "
      f"{m.loc[keep, '_pid'].nunique():,} patients")
out = os.path.join(os.path.dirname(os.path.abspath(IN_CSV)), "propofol_cohort_intermittent_dose_counts.csv")
tab.to_csv(out, index=False)
print(f"\nSaved {out}")
