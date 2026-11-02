"""
build_strata.py

Feasibility counts for the agent-stratified analysis:
    NESI ~ RASS x agent + (1|patient)

Counts only; nothing is modeled.

Exposure rule (same "on board" logic for every class, at the RASS time):
  propofol, dexmedetomidine, ketamine : infusion running (<drug>_inf_now)
  benzodiazepine : midazolam infusion running, OR fast-acting dose in 6h, OR slow-acting in 12h
                   (benzo_on_board; infusion part = midazolam_inf_now)
  opioid         : opioid infusion running, OR fast-acting dose in 6h, OR slow-acting in 12h
                   (opioid_on_board; infusion part = opioid_inf_now)
  Vasopressors are ignored for strata; used only as a negative control in the reference group.

Strata (one per observation):
  no sedative or opioid           nothing on board from any class (reference)
  <agent> only / <agent> + opioid infusion-based combinations
  midazolam + propofol (+/- opioid), midazolam + dexmedetomidine (+/- opioid)
  ketamine (+/- others)           any ketamine infusion
  other combinations              any other mix of infusions
  intermittent dose involved      primary: a benzo dose on board without a midazolam infusion -> excluded
                                  (intermittent opioid doses ignored; opioids classified by infusion)
                                  strict sensitivity: a benzo or opioid dose without its infusion
  unknown medication status       missing flags -> excluded

Inputs:
  NESI file    : full-cohort NESI file from colleague (column NESI: a value, or a text
                 reason it could not be calculated). Matched on patient + RASS time.
                 If two EEG windows share one RASS time, the first NESI in file order
                 is kept (temporary rule, pending colleague's explanation).
  Windows file : <master>_windows_opiates.csv (rerun with benzo windows first)

Run with Play. If a column name is wrong, the script stops before loading data
and prints candidate names from each file.
"""

import sys
from pathlib import Path

import pandas as pd

# Repo layout: med_config.py lives in ../medication-exposure (no effect if already importable)
import sys as _sys
from pathlib import Path as _Path
_MC = _Path(__file__).resolve().parent.parent / "medication-exposure"
if _MC.is_dir() and str(_MC) not in _sys.path:
    _sys.path.insert(0, str(_MC))
from med_config import OUTPUT_CSV, I0001_ASM_ANTIPSYCH_DIR

# EDIT: full path to the colleague's NESI file (.csv or .xlsx)
NESI_CSV = Path(__file__).resolve().parent / "data" / "i0001_RASS_Meds_NESI.csv"
# Output of add_sedation_windows_opiates.py (GCS+RASS cohort), same folder as the master file
WINDOWS_CSV = Path(I0001_ASM_ANTIPSYCH_DIR) / "i0001_per_eeg_master_exposure_windows_opiates.csv"

# ----------------------------------------------------------------------
# CONFIG: column names (edit if the check fails)
# ----------------------------------------------------------------------
NESI_COLS = {
    "patient": "BDSPPatientID",
    "time":    "RASSRecordedDTS",
    "nesi":    "NESI",
}
SESSION_COL = "SessionID"
# RASS times covered by two overlapping EEG files (NESI values disagree):
#   "drop"           -> exclude those RASS times (primary)
#   "higher_session" -> keep the NESI from the higher SessionID
DUP_RULE = "drop"
NESI_GCS_TIME = "GCSRecordedDTS"   # optional; used only to count GCS assessments with NESI
WIN_COLS = {
    "patient":   "BDSPPatientID",
    "time":      "RASSRecordedDTS",
    "rass":      "R PHS IP RASS",
    "propofol":  "propofol_inf_now",
    "dex":       "dexmedetomidine_inf_now",
    "ketamine":  "ketamine_inf_now",
    "midaz_inf": "midazolam_inf_now",
    "benzo_obb": "benzo_on_board",
    "opioid_inf": "opioid_inf_now",
    "opioid_obb": "opioid_on_board",
    "vaso":      "any_vasopressor_inf_now",
}

# Extra columns for the midazolam/ketamine reconciliation (used if present)
RECON = {
    "midazolam": {"table": "Midazolam_Infusion_Exposure", "inf_24h": "midazolam_inf_24h",
                  "inf_now": "midazolam_inf_now", "on_board": "benzo_on_board"},
    "ketamine":  {"table": "Ketamine_Infusion_Exposure", "inf_24h": "ketamine_inf_24h",
                  "inf_now": "ketamine_inf_now"},
}

MIN_OBS = 100        # pre-specified
MIN_PATIENTS = 20

OUT_DIR = Path(WINDOWS_CSV).parent / "stratified_feasibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("INPUTS")
print(f"  NESI file    : {Path(NESI_CSV).resolve()}")
print(f"  windows file : {Path(WINDOWS_CSV).resolve()}")
print("OUTPUT")
print(f"  {OUT_DIR.resolve()}\n")
for p in (NESI_CSV, WINDOWS_CSV):
    if not Path(p).exists():
        sys.exit(f"File not found: {Path(p).resolve()}")


# ----------------------------------------------------------------------
# 1. Column check (headers only)
# ----------------------------------------------------------------------
def read_table(path, **kw):
    """CSV or Excel, by file extension."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        kw.pop("low_memory", None)
        if "nrows" in kw and kw["nrows"] == 0:
            kw["nrows"] = 1
        return pd.read_excel(path, **kw)
    return pd.read_csv(path, **kw)


def check(path, cols, terms):
    head = read_table(path, nrows=0).columns.tolist()
    missing = [c for c in cols.values() if c not in head]
    if missing:
        print(f"\n{Path(path).name}: not found -> {missing}")
        for t in terms:
            hits = [c for c in head if t in c.lower()]
            if hits:
                print(f"  {t:10s} -> {hits[:12]}")
    return not missing


ok1 = check(NESI_CSV, NESI_COLS, ["nesi", "patient", "dts", "time", "rass"])
ok2 = check(WINDOWS_CSV, WIN_COLS, ["patient", "rass", "_inf_now", "on_board"])
if not (ok1 and ok2):
    print("\nEdit NESI_COLS / WIN_COLS at the top and run again."
          "\n(benzo_on_board missing = rerun add_sedation_windows_opiates.py first.)")
    sys.exit(1)


# ----------------------------------------------------------------------
# 2. Load and join on patient + RASS time
# ----------------------------------------------------------------------
nesi_head = read_table(NESI_CSV, nrows=0).columns.tolist()
if SESSION_COL not in nesi_head:
    sys.exit(f"{SESSION_COL} not found in the NESI file; edit SESSION_COL at the top.")
nesi_use = list(NESI_COLS.values()) + [SESSION_COL] + \
    ([NESI_GCS_TIME] if NESI_GCS_TIME in nesi_head else [])
nesi = read_table(NESI_CSV, usecols=nesi_use, low_memory=False)
win_head = read_table(WINDOWS_CSV, nrows=0).columns.tolist()
recon_cols = [c for drug in RECON.values() for c in drug.values() if c in win_head]
win = read_table(WINDOWS_CSV, usecols=list(dict.fromkeys(list(WIN_COLS.values()) + recon_cols)),
                 low_memory=False)

for d, c in [(nesi, NESI_COLS), (win, WIN_COLS)]:
    d["_pid"] = d[c["patient"]].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    d["_t"] = pd.to_datetime(d[c["time"]], errors="coerce")

# NESI: numeric value, or a text reason it could not be calculated
raw_nesi = nesi[NESI_COLS["nesi"]]
nesi["nesi_value"] = pd.to_numeric(raw_nesi, errors="coerce")
reasons = (raw_nesi[nesi["nesi_value"].isna() & raw_nesi.notna()]
           .astype(str).str.strip().value_counts())
print(f"NESI file: {len(nesi):,} rows; numeric NESI {nesi['nesi_value'].notna().sum():,}; "
      f"text reason {int(reasons.sum()):,}; blank {int(raw_nesi.isna().sum()):,}")
if len(reasons):
    print("Reasons NESI could not be calculated (top 10):")
    print(reasons.head(10).to_string())

# Count GCS assessments with NESI (tells us whether a GCS analysis is possible)
if NESI_GCS_TIME in nesi.columns:
    n_gcs = int((nesi["nesi_value"].notna() & nesi[NESI_GCS_TIME].notna()).sum())
    print(f"Rows with numeric NESI and a GCS time: {n_gcs:,}")

# One NESI per patient + RASS time. Some RASS times are covered by two overlapping EEG
# files; rule (from the data team): keep the NESI from the higher SessionID.
nesi = nesi.dropna(subset=["_t", "nesi_value"])
nesi["_sess"] = pd.to_numeric(nesi[SESSION_COL].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
dup = nesi.duplicated(["_pid", "_t"], keep=False)
n_dup_keys = nesi.loc[dup, ["_pid", "_t"]].drop_duplicates().shape[0]
# Patients with overlapping EEG files (duplicate keys): flagged so the model can be rerun
# without them (all their rows), since differing NESI for the same 10 minutes may mean a
# timestamp shift or file problem that also affects their non-duplicated rows.
dup_pids = set(nesi.loc[dup, "_pid"])
if n_dup_keys:
    spread = nesi[dup].groupby(["_pid", "_t"])["nesi_value"].agg(lambda x: x.max() - x.min())
    sd_all = nesi["nesi_value"].std()
    print(f"NOTE: {n_dup_keys:,} patient + RASS times have two EEG windows "
          f"({len(dup_pids):,} patients); rule: {DUP_RULE}.")
    print(f"  NESI difference within duplicates: median {spread.median():.2f}, "
          f"90th pct {spread.quantile(.9):.2f}, max {spread.max():.2f} "
          f"(overall NESI SD {sd_all:.2f})")
else:
    print("One NESI value per patient + RASS time.")
if DUP_RULE == "drop":
    nesi = nesi[~dup][["_pid", "_t", "nesi_value"]]
elif DUP_RULE == "higher_session":
    nesi = (nesi.sort_values(["_pid", "_t", "_sess"], ascending=[True, True, False], na_position="last")
            .drop_duplicates(["_pid", "_t"], keep="first")[["_pid", "_t", "nesi_value"]])
else:
    sys.exit(f"Unknown DUP_RULE: {DUP_RULE}")
NESI_COLS["nesi"] = "nesi_value"
win = win.dropna(subset=["_t"]).drop_duplicates(["_pid", "_t"])

df = nesi[["_pid", "_t", NESI_COLS["nesi"]]].merge(
    win.drop(columns=[WIN_COLS["patient"], WIN_COLS["time"]], errors="ignore"),
    on=["_pid", "_t"], how="left", indicator=True)

n_join = (df["_merge"] == "both").sum()
print(f"NESI rows: {len(nesi):,}  matched to windows file: {n_join:,} "
      f"({n_join / len(nesi) * 100:.1f}%)")
df = df[df["_merge"] == "both"].dropna(subset=[WIN_COLS["rass"]])

# Sanity: is the NESI file the full RASS cohort, or only propofol patients?
pct_no_prop = (df[WIN_COLS["propofol"]] == 0).mean() * 100
print(f"Rows without a propofol infusion: {pct_no_prop:.1f}%"
      + ("   <-- WARNING: file may contain only propofol patients" if pct_no_prop < 5 else ""))


# ----------------------------------------------------------------------
# 3. Strata
# ----------------------------------------------------------------------
flag_cols = [WIN_COLS[k] for k in ["propofol", "dex", "ketamine", "midaz_inf",
                                   "benzo_obb", "opioid_inf", "opioid_obb"]]
unknown = df[flag_cols].isna().any(axis=1)
f = df[flag_cols].fillna(0).astype(int)
f.columns = ["prop", "dex", "ket", "midaz", "benzo", "op_inf", "op"]

# PRIMARY rule: opioids by infusion only (intermittent opioid doses ignored, as in the
#   propofol cohort; expanded-cohort analysis showed opioids add little).
#   Intermittent BENZO doses still exclude (they would contaminate the reference group).
# SENSITIVITY rule ("strict"): intermittent benzo OR opioid doses exclude.
benzo_intermittent = (f["benzo"] == 1) & (f["midaz"] == 0)
opioid_intermittent = (f["op"] == 1) & (f["op_inf"] == 0)
INTERMITTENT = {"primary": benzo_intermittent,
                "strict": benzo_intermittent | opioid_intermittent}

NAMES = {"prop": "propofol", "dex": "dexmedetomidine", "midaz": "midazolam"}


def assign(i, rule="primary"):
    if unknown.iat[i]:
        return "unknown medication status"
    if INTERMITTENT[rule].iat[i]:
        return "intermittent dose involved"
    on = [k for k in ["prop", "dex", "midaz", "ket"] if f[k].iat[i] == 1]
    op = f["op_inf"].iat[i] == 1
    if not on and not op:
        return "no sedative or opioid"
    if "ket" in on:
        return "ketamine (+/- others)"
    if not on and op:
        return "opioid only"
    if len(on) == 1:
        return f"{NAMES[on[0]]} + opioid" if op else f"{NAMES[on[0]]} only"
    if set(on) == {"midaz", "prop"}:
        return "midazolam + propofol + opioid" if op else "midazolam + propofol"
    if set(on) == {"midaz", "dex"}:
        return "midazolam + dexmedetomidine + opioid" if op else "midazolam + dexmedetomidine"
    return "other combinations"


df["stratum"] = [assign(i, "primary") for i in range(len(df))]
df["stratum_strict"] = [assign(i, "strict") for i in range(len(df))]

ORDER = ["no sedative or opioid", "propofol only", "dexmedetomidine only",
         "midazolam only", "opioid only", "propofol + opioid",
         "dexmedetomidine + opioid", "midazolam + opioid",
         "midazolam + propofol", "midazolam + propofol + opioid",
         "midazolam + dexmedetomidine", "midazolam + dexmedetomidine + opioid",
         "ketamine (+/- others)", "other combinations",
         "intermittent dose involved", "unknown medication status"]
EXCLUDED = {"intermittent dose involved", "unknown medication status"}


# ----------------------------------------------------------------------
# 4. Counts and RASS spread
# ----------------------------------------------------------------------
rass = WIN_COLS["rass"]


def count_table(col):
    rows = []
    for s in ORDER:
        g = df[df[col] == s]
        n_obs, n_pat = len(g), g["_pid"].nunique()
        x = g[rass]
        rows.append({
            "stratum": s,
            "n_obs": n_obs,
            "n_patients": n_pat,
            "passes_rule": (n_obs >= MIN_OBS and n_pat >= MIN_PATIENTS and s not in EXCLUDED),
            "rass_median": x.median() if n_obs else None,
            "rass_IQR": f"{x.quantile(.25):g} to {x.quantile(.75):g}" if n_obs else "",
            "rass_levels_>=10_obs": int((x.value_counts() >= 10).sum()) if n_obs else 0,
            "pct_rass_<=-4": round((x <= -4).mean() * 100, 1) if n_obs else None,
        })
    return pd.DataFrame(rows)


counts = count_table("stratum")
counts_strict = count_table("stratum_strict")

dist = (pd.crosstab(df["stratum"], df[rass], normalize="index") * 100).round(1)
dist = dist.reindex([s for s in ORDER if s in dist.index])

# Counts by RASS level: observations and distinct patients (is each level anchored by enough patients?)
lvl_obs = pd.crosstab(df["stratum"], df[rass]).reindex([s for s in ORDER if s in dist.index])
lvl_pat = (df.groupby(["stratum", rass])["_pid"].nunique().unstack(fill_value=0)
           .reindex([s for s in ORDER if s in dist.index]))
print("\nPatients at each RASS level, by stratum:")
print(lvl_pat.to_string())


# ----------------------------------------------------------------------
# 5. Negative control: any vasopressor, within the reference group
# ----------------------------------------------------------------------
ref = df[(df["stratum"] == "no sedative or opioid")].dropna(subset=[WIN_COLS["vaso"]])
vrows = []
for v, name in [(1, "vasopressor"), (0, "no vasopressor")]:
    g = ref[ref[WIN_COLS["vaso"]] == v]
    vrows.append({"group": name, "n_obs": len(g), "n_patients": g["_pid"].nunique(),
                  "passes_rule": len(g) >= MIN_OBS and g["_pid"].nunique() >= MIN_PATIENTS,
                  "rass_median": g[rass].median() if len(g) else None})
vaso = pd.DataFrame(vrows)


# ----------------------------------------------------------------------
# 5b. Reconciliation: midazolam and ketamine under different definitions
#     (same matched cohort; denominator = all patients in it)
# ----------------------------------------------------------------------
n_pat_all = df["_pid"].nunique()
DEF_LABEL = {"table": "Exposure table column (*_Infusion_Exposure)",
             "inf_24h": "Infusion in prior 24 h (any RASS time)",
             "inf_now": "Infusion running at any RASS time",
             "on_board": "Benzo on board at any RASS time (incl. doses)"}
recon_rows = []
for drug, defs in RECON.items():
    for k, col in defs.items():
        if col not in df.columns:
            recon_rows.append({"drug": drug, "definition": DEF_LABEL[k], "column": col,
                               "patients": None, "pct_patients": None, "pct_obs": None})
            continue
        x = pd.to_numeric(df[col], errors="coerce") == 1
        n_p = df.loc[x, "_pid"].nunique()
        recon_rows.append({"drug": drug, "definition": DEF_LABEL[k], "column": col,
                           "patients": n_p, "pct_patients": round(100 * n_p / n_pat_all, 1),
                           "pct_obs": round(100 * x.mean(), 1)})
recon = pd.DataFrame(recon_rows)

# where do observations with a midazolam infusion running end up?
mid_now = pd.to_numeric(df[WIN_COLS["midaz_inf"]], errors="coerce") == 1
mid_strata = (df.loc[mid_now, "stratum"].value_counts()
              .rename("obs").to_frame()
              .assign(patients=df.loc[mid_now].groupby("stratum")["_pid"].nunique()))

print(f"\nReconciliation (denominator: {n_pat_all:,} patients in matched cohort)")
print(recon.to_string(index=False))
print("\nObservations with a midazolam infusion running, by stratum:")
print(mid_strata.to_string())


# ----------------------------------------------------------------------
# 6. Save
# ----------------------------------------------------------------------
# Analysis rows for agent_stratified_model.py (strata defined here only)
ANALYSIS_ROWS = OUT_DIR / "strata_analysis_rows.csv"
df["dup_patient"] = df["_pid"].isin(dup_pids).astype(int)
print(f"Rows from patients with overlapping EEG files: {int(df['dup_patient'].sum()):,} "
      f"of {len(df):,}")
df[["_pid", "_t", NESI_COLS["nesi"], rass, "stratum", "stratum_strict", WIN_COLS["vaso"],
    "dup_patient"]].rename(
    columns={NESI_COLS["nesi"]: "nesi", rass: "rass", WIN_COLS["vaso"]: "vaso"}
).to_csv(ANALYSIS_ROWS, index=False)
print(f"\nAnalysis rows saved for the model script:\n  {ANALYSIS_ROWS.resolve()}")

counts.to_csv(OUT_DIR / "strata_counts_rass.csv", index=False)
counts_strict.to_csv(OUT_DIR / "strata_counts_rass_strict.csv", index=False)
dist.to_csv(OUT_DIR / "strata_rass_distribution_pct.csv")
lvl_obs.to_csv(OUT_DIR / "strata_rass_level_obs.csv")
lvl_pat.to_csv(OUT_DIR / "strata_rass_level_patients.csv")
vaso.to_csv(OUT_DIR / "vasopressor_negative_control_counts.csv", index=False)

with open(OUT_DIR / "stratified_feasibility_report.md", "w") as fh:
    fh.write("# Agent-stratified analysis: feasibility (NESI-RASS)\n\n")
    fh.write(f"NESI rows matched to exposure flags: {n_join:,} of {len(nesi):,}.\n")
    fh.write(f"RASS times with two EEG windows (rule: {DUP_RULE}): {n_dup_keys:,}.\n\n")
    if len(reasons):
        fh.write("Reasons NESI could not be calculated:\n\n"
                 + reasons.rename("n").to_frame().to_markdown() + "\n\n")
    fh.write(f"Rule (pre-specified): >= {MIN_OBS} observations and >= {MIN_PATIENTS} patients.\n\n")
    fh.write("## Primary rule: opioids by infusion only; intermittent benzo doses exclude\n\n")
    fh.write(counts.to_markdown(index=False) + "\n\n")
    fh.write("## Sensitivity (strict): intermittent benzo or opioid doses exclude\n\n")
    fh.write(counts_strict.to_markdown(index=False) + "\n\n")
    fh.write("## RASS distribution within stratum (% of obs)\n\n" + dist.to_markdown() + "\n\n")
    fh.write("## Observations at each RASS level\n\n" + lvl_obs.to_markdown() + "\n\n")
    fh.write("## Distinct patients at each RASS level\n\n" + lvl_pat.to_markdown() + "\n\n")
    fh.write("## Negative control: any vasopressor, reference group only\n\n")
    fh.write(vaso.to_markdown(index=False) + "\n\n")
    fh.write(f"## Reconciliation: midazolam and ketamine (denominator {n_pat_all:,} patients)\n\n")
    fh.write(recon.to_markdown(index=False) + "\n\n")
    fh.write("Observations with a midazolam infusion running, by stratum:\n\n")
    fh.write(mid_strata.to_markdown() + "\n")

pd.set_option("display.width", 220)
print("\nPRIMARY (opioids by infusion only; intermittent benzo doses exclude)")
print(counts.to_string(index=False))
print("\nSENSITIVITY (strict: intermittent benzo or opioid doses exclude)")
print(counts_strict[["stratum", "n_obs", "n_patients", "passes_rule"]].to_string(index=False))
print("\n" + vaso.to_string(index=False))
print(f"\nSaved to {OUT_DIR}")
print("\a")
