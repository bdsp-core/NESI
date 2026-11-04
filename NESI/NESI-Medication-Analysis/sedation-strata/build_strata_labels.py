# build_strata_labels.py
"""
Sedation group labels for each GCS, RASS and CAM-S assessment, for the performance-by-sedation
analysis. No NESI needed: the output is a label file to join to NESI/performance data on
patient ID + assessment time.

Same rules as build_strata.py (locked Oct 29):
  - flags at the assessment time: infusion running now or stopped within 6 h; benzo/opioid "on board"
  - primary rule: opioids by infusion only; intermittent benzo doses -> "intermittent dose involved"
  - strict rule: intermittent benzo OR opioid doses -> excluded
  - mutually exclusive strata; ketamine only / ketamine + opioid (ketamine with another sedative = other combinations)

Each row of the windows file is one assessment. Its time is GCSRecordedDTS or RASSRecordedDTS.
Rows with BOTH times are ambiguous (flags were anchored to the GCS time) and are counted and
dropped from the RASS labels.

CAM-S: read from the CAM-S windows file (same flags, same rules). Set CAMS_TIME / CAMS_SCORE
below; if they are not found the script lists candidate columns and skips CAM-S.

Outputs (in stratified_feasibility/labels/):
  strata_labels_gcs.csv, strata_labels_rass.csv, strata_labels_cams.csv
     BDSPPatientID, assessment_time, score, one 0/1 column per group,
     Included under strict rule (0/1), Vasopressor (0/1)
     one 0/1 column per group, named by the group (5 performance groups first)
  strata_label_counts.csv : patients and assessments per group, all cohorts
  performance_group_counts.csv : same, for the 5 performance groups
Counts only are printed. Run with Play. Paths are printed at start.
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
import med_config as mc
from med_config import I0001_ASM_ANTIPSYCH_DIR

WINDOWS_CSV = Path(I0001_ASM_ANTIPSYCH_DIR) / "i0001_per_eeg_master_exposure_windows_opiates.csv"
OUT_DIR = Path(I0001_ASM_ANTIPSYCH_DIR) / "stratified_feasibility" / "labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CAM-S cohort (EDIT the two column names if the script says they are not found)
CAMS_WINDOWS_CSV = Path(getattr(mc, "I0001_CAMS_ASM_ANTIPSYCH_DIR", ".")) / \
    "i0001_cams_per_eeg_master_exposure_windows_opiates.csv"
CAMS_TIME  = ["CAMSRecordedDTS", "RecordedDTS", "Snippet_StartDTS"]   # first one found is used
CAMS_SCORE = None    # e.g. "R CAM-S SHORT FORM SCORE"; None = leave score blank

PATIENT = "BDSPPatientID"
TIMES = {"gcs": "GCSRecordedDTS", "rass": "RASSRecordedDTS"}
SCORES = {"gcs": "R CPN GLASGOW COMA SCALE SCORE", "rass": "R PHS IP RASS"}
# Infusions count if running at the assessment OR stopped within the prior 6 h (same rule as
# build_strata.py). Intermittent doses: benzo/opioid on board (fast-acting 6 h, slow-acting 12 h).
INFUSION_LOOKBACK = "6h"
FLAGS = {"prop": f"propofol_inf_{INFUSION_LOOKBACK}", "dex": f"dexmedetomidine_inf_{INFUSION_LOOKBACK}",
         "ket": f"ketamine_inf_{INFUSION_LOOKBACK}", "midaz": f"midazolam_inf_{INFUSION_LOOKBACK}",
         "benzo": "benzo_on_board", "op_inf": f"opioid_inf_{INFUSION_LOOKBACK}", "op": "opioid_on_board"}
VASO = "any_vasopressor_inf_now"
MIN_OBS, MIN_PATIENTS = 100, 20

# analysis groups (match the main table)
GROUP = {
    "no sedative or opioid": "Reference (no sedative or opioid infusion)",
    "propofol only": "Propofol only",
    "propofol + opioid": "Propofol + opioid",
    "midazolam only": "Midazolam only",
    "midazolam + opioid": "Midazolam + opioid",
    "dexmedetomidine only": "Dexmedetomidine only",
    "dexmedetomidine + opioid": "Dexmedetomidine + opioid",
    "midazolam + propofol": "Midazolam + propofol (± opioid)",
    "midazolam + propofol + opioid": "Midazolam + propofol (± opioid)",
    "ketamine only": "Ketamine (± opioid)",
    "ketamine + opioid": "Ketamine (± opioid)",
    "opioid only": "Opioid only",
}
# simpler groups for the performance-by-subgroup table (each sedative pooled with/without
# opioid). Midazolam + propofol, ketamine and midazolam + dex are left out (blank).
GROUP_PERF = {
    "no sedative or opioid": "Reference (no sedative or opioid infusion)",
    "propofol only": "Propofol (± opioid)",
    "propofol + opioid": "Propofol (± opioid)",
    "dexmedetomidine only": "Dexmedetomidine (± opioid)",
    "dexmedetomidine + opioid": "Dexmedetomidine (± opioid)",
    "midazolam only": "Midazolam (± opioid)",
    "midazolam + opioid": "Midazolam (± opioid)",
    "opioid only": "Opioid only",
}
# 0/1 indicator columns, named by group. The 5 performance groups first, then the
# split groups from the main table (Reference and Opioid only are not repeated: same rows).
INDICATORS = {
    "Reference (no sedative or opioid infusion)": ("group_performance", "Reference (no sedative or opioid infusion)"),
    "Propofol (+/- opioid)": ("group_performance", "Propofol (± opioid)"),
    "Dexmedetomidine (+/- opioid)": ("group_performance", "Dexmedetomidine (± opioid)"),
    "Midazolam (+/- opioid)": ("group_performance", "Midazolam (± opioid)"),
    "Opioid only": ("group_performance", "Opioid only"),
    "Propofol only": ("group", "Propofol only"),
    "Propofol + opioid": ("group", "Propofol + opioid"),
    "Midazolam only": ("group", "Midazolam only"),
    "Midazolam + opioid": ("group", "Midazolam + opioid"),
    "Dexmedetomidine only": ("group", "Dexmedetomidine only"),
    "Dexmedetomidine + opioid": ("group", "Dexmedetomidine + opioid"),
    "Midazolam + propofol (+/- opioid)": ("group", "Midazolam + propofol (± opioid)"),
    "Ketamine (+/- opioid)": ("group", "Ketamine (± opioid)"),
}
NAMES = {"prop": "propofol", "dex": "dexmedetomidine", "midaz": "midazolam", "ket": "ketamine"}

print("INPUTS")
print(f"  GCS + RASS: {WINDOWS_CSV.resolve()}")
print(f"  CAM-S     : {CAMS_WINDOWS_CSV.resolve()}")
print("OUTPUT")
print(f"  {OUT_DIR.resolve()}\n")
if not WINDOWS_CSV.exists():
    sys.exit(f"File not found: {WINDOWS_CSV.resolve()}")

need = [PATIENT, *TIMES.values(), *SCORES.values(), *FLAGS.values(), VASO]
head = pd.read_csv(WINDOWS_CSV, nrows=0).columns.tolist()
missing = [c for c in need if c not in head]
if missing:
    sys.exit(f"Columns not found: {missing}\n(benzo_on_board missing = rerun add_sedation_windows_opiates.py)")
w = pd.read_csv(WINDOWS_CSV, usecols=need, low_memory=False)
w[PATIENT] = w[PATIENT].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
for c in TIMES.values():
    w[c] = pd.to_datetime(w[c], errors="coerce")

g, r = w[TIMES["gcs"]].notna(), w[TIMES["rass"]].notna()
print(f"Rows: {len(w):,}   GCS only: {(g & ~r).sum():,}   RASS only: {(r & ~g).sum():,}   "
      f"both: {(g & r).sum():,}   neither: {(~g & ~r).sum():,}")
if (g & r).any():
    print("  NOTE: rows with both times have flags anchored to the GCS time; "
          "they are used for GCS and dropped from RASS.")


def make_out(d, tcol, score):
    """Label table: ID, time, score, 0/1 indicators (performance groups first), then text labels."""
    out = pd.DataFrame({PATIENT: d[PATIENT].values, "assessment_time": d[tcol].values,
                        "score": score})
    grp, perf = d["stratum"].map(GROUP).values, d["stratum"].map(GROUP_PERF).values
    for col, (kind, val) in INDICATORS.items():
        src = perf if kind == "group_performance" else grp
        out[col] = (src == val).astype(int)
    # 1 = kept under the strict rule (no intermittent opioid dose either); the group is unchanged
    in_any = out[list(INDICATORS)].max(axis=1).values == 1
    out["Included under strict rule"] = (in_any & (d["stratum_strict"].values == d["stratum"].values)).astype(int)
    out["Vasopressor"] = d[VASO].values
    out["stratum"] = d["stratum"].values                 # used for the counts; not written
    out["group"], out["group_performance"] = grp, perf   # used for the counts; not written
    return out


def perf_table(out, cohort):
    p = (out.dropna(subset=["group_performance"]).groupby("group_performance")
         .agg(assessments=(PATIENT, "size"), patients=(PATIENT, "nunique")).reset_index())
    ok = (p["assessments"] >= MIN_OBS) & (p["patients"] >= MIN_PATIENTS)
    p["passes_rule"] = ok.map({True: "yes", False: "NO"})
    p.insert(0, "cohort", cohort)
    print(f"\n  Performance groups ({cohort}):")
    print("  " + p.drop(columns="cohort").to_string(index=False).replace("\n", "\n  "))
    return p


perf_counts = []


def label(df):
    unknown = df[list(FLAGS.values())].isna().any(axis=1).values
    f = df[list(FLAGS.values())].fillna(0).astype(int)
    f.columns = list(FLAGS)
    benzo_int = ((f["benzo"] == 1) & (f["midaz"] == 0)).values
    op_int = ((f["op"] == 1) & (f["op_inf"] == 0)).values
    rows = f.to_dict("records")

    def assign(i, strict):
        if unknown[i]:
            return "unknown medication status"
        if benzo_int[i] or (strict and op_int[i]):
            return "intermittent dose involved"
        x = rows[i]
        on = [k for k in ["prop", "dex", "midaz", "ket"] if x[k] == 1]
        op = x["op_inf"] == 1
        if not on and not op:
            return "no sedative or opioid"
        if not on:
            return "opioid only"
        if len(on) == 1:
            return f"{NAMES[on[0]]} + opioid" if op else f"{NAMES[on[0]]} only"
        if set(on) == {"midaz", "prop"}:
            return "midazolam + propofol + opioid" if op else "midazolam + propofol"
        if set(on) == {"midaz", "dex"}:
            return "midazolam + dexmedetomidine + opioid" if op else "midazolam + dexmedetomidine"
        return "other combinations"

    return [assign(i, False) for i in range(len(df))], [assign(i, True) for i in range(len(df))]


counts = []
for kind in ["gcs", "rass"]:
    sel = g if kind == "gcs" else (r & ~g)
    d = w[sel].copy()
    d = d.drop_duplicates([PATIENT, TIMES[kind]])            # one row per assessment
    d["stratum"], d["stratum_strict"] = label(d)
    d["group"] = d["stratum"].map(GROUP)
    out = make_out(d, TIMES[kind], d[SCORES[kind]].values)
    out.drop(columns=["group", "group_performance", "stratum"]).to_csv(OUT_DIR / f"strata_labels_{kind}.csv", index=False)

    c = (out.groupby(out["group"].fillna("(excluded) " + out["stratum"]))
         .agg(assessments=(PATIENT, "size"), patients=(PATIENT, "nunique")).reset_index()
         .rename(columns={"group": "group"}))
    ok = (c["assessments"] >= MIN_OBS) & (c["patients"] >= MIN_PATIENTS)
    c["passes_rule"] = ok.map({True: "yes", False: "NO"})
    c.loc[c["group"].str.startswith("(excluded)"), "passes_rule"] = ""
    c.insert(0, "cohort", kind.upper())
    counts.append(c.sort_values("assessments", ascending=False))
    print(f"\n{kind.upper()}: {len(out):,} assessments, {out[PATIENT].nunique():,} patients")
    print(c.drop(columns="cohort").to_string(index=False))
    perf_counts.append(perf_table(out, kind.upper()))

# ---------------- CAM-S ----------------
print("\n=== CAM-S ===")
if not CAMS_WINDOWS_CSV.exists():
    print(f"File not found, CAM-S skipped: {CAMS_WINDOWS_CSV.resolve()}")
else:
    ch = pd.read_csv(CAMS_WINDOWS_CSV, nrows=0).columns.tolist()
    tcol = next((c for c in CAMS_TIME if c in ch), None)
    miss = [c for c in [PATIENT, *FLAGS.values(), VASO] if c not in ch]
    if CAMS_SCORE and CAMS_SCORE not in ch:
        miss.append(CAMS_SCORE)
    if tcol is None or miss:
        print(f"CAM-S skipped. Time column found: {tcol}; missing: {miss}")
        print("Candidate columns: " + str([c for c in ch if any(k in c.upper() for k in ["CAM", "DTS", "TIME"])]))
    else:
        use = [PATIENT, tcol, *FLAGS.values(), VASO] + ([CAMS_SCORE] if CAMS_SCORE else [])
        d = pd.read_csv(CAMS_WINDOWS_CSV, usecols=use, low_memory=False)
        d[PATIENT] = d[PATIENT].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
        d = d.dropna(subset=[tcol]).drop_duplicates([PATIENT, tcol])
        d["stratum"], d["stratum_strict"] = label(d)
        out = make_out(d, tcol, d[CAMS_SCORE].values if CAMS_SCORE else pd.NA)
        out.drop(columns=["group", "group_performance", "stratum"]).to_csv(OUT_DIR / "strata_labels_cams.csv", index=False)
        c = (out.groupby(out["group"].fillna("(excluded) " + out["stratum"]))
             .agg(assessments=(PATIENT, "size"), patients=(PATIENT, "nunique")).reset_index())
        ok = (c["assessments"] >= MIN_OBS) & (c["patients"] >= MIN_PATIENTS)
        c["passes_rule"] = ok.map({True: "yes", False: "NO"})
        c.loc[c["group"].str.startswith("(excluded)"), "passes_rule"] = ""
        c.insert(0, "cohort", "CAMS")
        counts.append(c.sort_values("assessments", ascending=False))
        print(f"Time column: {tcol}. {len(out):,} assessments, {out[PATIENT].nunique():,} patients")
        print(c.drop(columns="cohort").to_string(index=False))
        perf_counts.append(perf_table(out, "CAMS"))

pd.concat(counts).to_csv(OUT_DIR / "strata_label_counts.csv", index=False)
pd.concat(perf_counts).to_csv(OUT_DIR / "performance_group_counts.csv", index=False)
print(f"\nSaved to {OUT_DIR.resolve()}")
