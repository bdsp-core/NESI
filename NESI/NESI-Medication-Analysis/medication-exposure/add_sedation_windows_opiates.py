# add_sedation_windows_opiates.py
"""
Add multi-window sedative, opioid, and vasopressor exposure flags to the
per-EEG master exposure files (I0001 GCS+RASS, I0001 CAMS).

Windows, anchored to the EEG/score timestamp t:
  now : infusion running at t          (StartTime <= t <= EndTime)
  6h  : any exposure in [t-6h, t]
  12h : any exposure in [t-12h, t]
  24h : any exposure in [t-24h, t]

INFUSIONS (from <drug>_intervals.csv), all four windows:
  sedative          : propofol, midazolam, dexmedetomidine, ketamine
  opioid            : fentanyl, hydromorphone, morphine
  phenylephrine     : per-drug column is the covariate
  other_vasopressor : norepinephrine, epinephrine, vasopressin, dopamine, angiotensin II
  any_vasopressor   : phenylephrine + other_vasopressor
  Zero-rate intervals dropped if FILTER_ZERO_RATE (titrated off != running).

OPIATE BOLUSES (from build_bolus_exposure.py output), 6h / 12h / 24h only
(a bolus is a point event, so "running now" does not apply):
  opiate_fast_bolus : <drug>_boluses_iv.csv      ("fast-acting parenteral": IV, PCA, IM,
                      intranasal; see bolus_route_classify.py) for fentanyl,
                      morphine, hydromorphone
  opiate_slow_bolus : <drug>_boluses_non_iv.csv  (PO, SL, SQ, rectal, transdermal,
                      buccal) for the same drugs
  The 24h columns are checked against the existing FastActing_Opiate_Exposure /
  SlowActing_Opiate_Exposure columns. High agreement confirms fast = IV file and
  slow = non-IV file; low agreement means the master build defines them differently.

COMBINED:
  opioid_any_{6h,12h,24h} : opioid infusion OR fast bolus OR slow bolus

  opioid_on_board         : covariate for the expanded-cohort analysis =
                            opioid infusion running at t (opioid_inf_now)
                            OR fast-acting dose in prior 6h
                            OR slow-acting dose in prior 12h

BENZODIAZEPINE BOLUSES (same bolus files and route classes as opiates), 6h / 12h / 24h:
  benzo_fast_bolus : <drug>_boluses_iv.csv      for drugs in BENZOS
  benzo_slow_bolus : <drug>_boluses_non_iv.csv  for drugs in BENZOS
  24h columns checked against FastActing_Benzo_Exposure / SlowActing_Benzo_Exposure.
  Missing per-drug files are treated as no events, so check that validation line.

  benzo_on_board          : same rule as opioid_on_board =
                            midazolam infusion running at t (midazolam_inf_now)
                            OR fast-acting benzo dose in prior 6h
                            OR slow-acting benzo dose in prior 12h

Prior exposure only (nothing after t). Rows from patients with no MAR data
stay NA, matching the master files. I0002 is skipped: no reliable infusion data.

Read-only on inputs. Writes <master>_windows_opiates.csv next to each master file.
"""

import os
import duckdb
import pandas as pd

try:
    import winsound
except ImportError:
    winsound = None

import med_config as mc

WINDOWS = {"now": 0, "6h": 6, "12h": 12, "24h": 24}
BOLUS_WINDOWS = ["6h", "12h", "24h"]
FILTER_ZERO_RATE = True

SEDATIVES = ["propofol", "midazolam", "dexmedetomidine", "ketamine"]
OPIOIDS = ["fentanyl", "hydromorphone", "morphine"]
OTHER_PRESSORS = ["norepinephrine", "epinephrine", "vasopressin", "dopamine", "angiotensin_ii"]
PRESSORS = ["phenylephrine"] + OTHER_PRESSORS

TS_CANDIDATES = ["GCSRecordedDTS", "RASSRecordedDTS", "Snippet_StartDTS", "RecordedDTS"]
BOLUS_TIME_CANDIDATES = ["TakenTime", "MedicationTakenDTS", "BolusTime", "AdministrationDTS", "Time", "StartTime"]

# Benzodiazepines with bolus files from build_bolus_exposure.py.
# EDIT to match the <drug>_boluses_iv.csv / _non_iv.csv files in BOLUS_EXPOSURE_DIR.
# Matches benzo_bolus_configs.py buckets plus midazolam (bolus files from the sedative pipeline).
# Clonazepam/clobazam are excluded on purpose: they live in the long-acting ASM pipeline.
BENZOS = ["midazolam", "lorazepam", "diazepam", "other_benzo"]

# bolus class -> (file suffix, existing 24h master column to validate against)
OPIATE_BOLUS_CLASSES = {
    "opiate_fast_bolus": ("iv", "FastActing_Opiate_Exposure"),
    "opiate_slow_bolus": ("non_iv", "SlowActing_Opiate_Exposure"),
}
BENZO_BOLUS_CLASSES = {
    "benzo_fast_bolus": ("iv", "FastActing_Benzo_Exposure"),
    "benzo_slow_bolus": ("non_iv", "SlowActing_Benzo_Exposure"),
}
# (class dict, drug list) pairs processed by the bolus loop
BOLUS_GROUPS = [(OPIATE_BOLUS_CLASSES, OPIOIDS), (BENZO_BOLUS_CLASSES, BENZOS)]

COHORTS = {
    "i0001_gcs_rass": dict(
        master=os.path.join(mc.I0001_ASM_ANTIPSYCH_DIR, "i0001_per_eeg_master_exposure.csv"),
        sed_dir=mc.SEDATIVE_EXPOSURE_DIR,
        opi_dir=mc.OPIATE_EXPOSURE_DIR,
        vaso_dir=getattr(mc, "VASOPRESSOR_EXPOSURE_DIR", None),
        bolus_dir=getattr(mc, "BOLUS_EXPOSURE_DIR", None),
    ),
    "i0001_cams": dict(
        master=os.path.join(mc.I0001_CAMS_ASM_ANTIPSYCH_DIR, "i0001_cams_per_eeg_master_exposure.csv"),
        sed_dir=mc.I0001_CAMS_SEDATIVE_EXPOSURE_DIR,
        opi_dir=mc.I0001_CAMS_OPIATE_EXPOSURE_DIR,
        vaso_dir=getattr(mc, "I0001_CAMS_VASOPRESSOR_EXPOSURE_DIR", None),
        bolus_dir=getattr(mc, "I0001_CAMS_BOLUS_EXPOSURE_DIR", None),
    ),
}

CLASS_AGGREGATES = [
    ("sedative", SEDATIVES),
    ("opioid", OPIOIDS),
    ("other_vasopressor", OTHER_PRESSORS),
    ("any_vasopressor", PRESSORS),
]


def norm_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_intervals(path):
    """Returns (intervals, stats) or None if the file is missing."""
    if not path or not os.path.exists(path):
        return None
    head = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = ["BDSPPatientID", "StartTime", "EndTime"] + (["Rate"] if "Rate" in head else [])
    iv = pd.read_csv(path, usecols=usecols)
    iv["BDSPPatientID"] = norm_id(iv["BDSPPatientID"])
    iv["StartTime"] = pd.to_datetime(iv["StartTime"], errors="coerce")
    iv["EndTime"] = pd.to_datetime(iv["EndTime"], errors="coerce")

    stats = dict(
        n_raw=len(iv),
        bad_start=int(iv["StartTime"].isna().sum()),
        open_end=int(iv["EndTime"].isna().sum()),
        zero_rate=0,
    )
    iv = iv.dropna(subset=["StartTime"]).copy()
    iv["EndTime"] = iv["EndTime"].fillna(iv["StartTime"])  # point exposure

    if "Rate" in iv.columns:
        rate = pd.to_numeric(iv["Rate"], errors="coerce")
        zero = rate.eq(0)
        stats["zero_rate"] = int(zero.sum())
        if FILTER_ZERO_RATE:
            iv = iv.loc[~zero]

    dur_h = (iv["EndTime"] - iv["StartTime"]).dt.total_seconds() / 3600
    stats["dur_h_max"] = round(float(dur_h.max()), 1) if len(dur_h) else float("nan")
    return iv[["BDSPPatientID", "StartTime", "EndTime"]], stats


def load_boluses(path):
    """Bolus events as zero-length intervals (StartTime = EndTime = dose time).
    Returns (events, time_col) or None if the file is missing."""
    if not path or not os.path.exists(path):
        return None
    head = pd.read_csv(path, nrows=0).columns.tolist()
    time_col = next((c for c in BOLUS_TIME_CANDIDATES if c in head), None)
    if time_col is None or "BDSPPatientID" not in head:
        raise KeyError(f"{os.path.basename(path)}: need BDSPPatientID and one of "
                       f"{BOLUS_TIME_CANDIDATES}. Columns: {head}")
    b = pd.read_csv(path, usecols=["BDSPPatientID", time_col])
    b["BDSPPatientID"] = norm_id(b["BDSPPatientID"])
    b["StartTime"] = pd.to_datetime(b[time_col], errors="coerce")
    b = b.dropna(subset=["StartTime"])
    b["EndTime"] = b["StartTime"]
    return b[["BDSPPatientID", "StartTime", "EndTime"]], time_col


def flag_drug(con, eeg, iv):
    con.register("eeg_df", eeg)
    con.register("iv_df", iv)
    cases = ",\n            ".join(
        f"MAX(CASE WHEN i.EndTime >= e.t - INTERVAL {h} HOUR THEN 1 ELSE 0 END) AS w_{lab}"
        for lab, h in WINDOWS.items()
    )
    max_h = max(WINDOWS.values())
    out = con.execute(f"""
        SELECT e.row_id,
            {cases}
        FROM (SELECT row_id, BDSPPatientID, CAST(t AS TIMESTAMP) AS t FROM eeg_df) e
        JOIN (SELECT BDSPPatientID,
                     CAST(StartTime AS TIMESTAMP) AS StartTime,
                     CAST(EndTime   AS TIMESTAMP) AS EndTime
              FROM iv_df) i
          ON i.BDSPPatientID = e.BDSPPatientID
         AND i.StartTime <= e.t
         AND i.EndTime   >= e.t - INTERVAL {max_h} HOUR
        GROUP BY e.row_id
    """).df()
    con.unregister("eeg_df")
    con.unregister("iv_df")
    return out.set_index("row_id")


def flags_to_cols(f, index, no_med, labels):
    """Turn flag_drug output into Int64 columns over the full master index."""
    cols = {}
    for lab in labels:
        col = pd.Series(0, index=index, dtype="Int64")
        if len(f):
            col.loc[f.index] = f[f"w_{lab}"].astype("int64").values
        col[no_med] = pd.NA
        cols[lab] = col
    return cols


def existing_cols_for(drug, cols):
    hits = [c for c in cols if drug in c.lower()]
    if drug == "morphine":
        hits = [c for c in hits if "hydromorphone" not in c.lower()]
    return hits


def agreement(existing, new):
    both = existing.notna() & new.notna()
    if not both.any():
        return None, 0
    try:
        agree = (existing[both].astype(int).values == new[both].astype(int).values).mean()
    except (ValueError, TypeError):
        return None, int(both.sum())
    return agree, int(both.sum())


def na_series(index):
    return pd.Series(pd.NA, index=index, dtype="Int64")


def prevalence_line(new, prefix, labels):
    return "  ".join(f"{lab}={new[f'{prefix}_{lab}'].mean() * 100:5.1f}%" for lab in labels)


def process(con, label, cfg):
    path = cfg["master"]
    print(f"\n=== {label} ===")
    print("  INPUTS")
    print(f"    master file        : {os.path.abspath(path)}")
    for key, name in [("sed_dir", "sedative intervals"), ("opi_dir", "opioid intervals"),
                      ("vaso_dir", "vasopressor intervals"), ("bolus_dir", "bolus files")]:
        d = cfg.get(key)
        print(f"    {name:<19}: {os.path.abspath(d) if d else 'NOT SET in med_config'}")
    print("  OUTPUT")
    print(f"    {os.path.abspath(os.path.splitext(path)[0] + '_windows_opiates.csv')}\n")
    m = pd.read_csv(path, low_memory=False)
    m["BDSPPatientID"] = norm_id(m["BDSPPatientID"])

    # anchor timestamp (coalesce whichever columns exist)
    ts_cols = [c for c in TS_CANDIDATES if c in m.columns]
    if not ts_cols:
        raise KeyError(f"No timestamp column among {TS_CANDIDATES}. Columns: {list(m.columns)}")
    t = pd.Series(pd.NaT, index=m.index, dtype="datetime64[ns]")
    for c in ts_cols:
        t = t.fillna(pd.to_datetime(m[c], errors="coerce"))
    print(f"  timestamp from {ts_cols}; {int(t.isna().sum()):,} rows unparseable")

    # NA mask: rows where every existing sedative/opioid column is NA (no MAR data)
    drug_cols = [c for c in m.columns if any(d in c.lower() for d in SEDATIVES + OPIOIDS)]
    if drug_cols:
        no_med = m[drug_cols].isna().all(axis=1)
    else:
        print("  WARNING: no existing drug columns found; NA mask not applied")
        no_med = pd.Series(False, index=m.index)
    no_med = no_med | t.isna()
    print(f"  {len(m):,} EEG rows; {int(no_med.sum()):,} set to NA")

    eeg = pd.DataFrame({"row_id": m.index, "BDSPPatientID": m["BDSPPatientID"], "t": t})
    eeg = eeg.loc[~no_med]

    if not cfg["vaso_dir"]:
        print("  WARNING: vasopressor exposure dir not set in med_config; pressor flags will be NA")

    new = {}

    # ---------------- infusions ----------------
    groups = [
        (SEDATIVES, cfg["sed_dir"]),
        (OPIOIDS, cfg["opi_dir"]),
        (PRESSORS, cfg["vaso_dir"]),
    ]
    for drugs, d in groups:
        for drug in drugs:
            fpath = os.path.join(d, f"{drug}_intervals.csv") if d else None
            loaded = load_intervals(fpath)
            if loaded is None:
                print(f"  {drug:<16} intervals file MISSING ({fpath}); flags left NA")
                for lab in WINDOWS:
                    new[f"{drug}_inf_{lab}"] = na_series(m.index)
                continue
            iv, st = loaded
            cols = flags_to_cols(flag_drug(con, eeg, iv), m.index, no_med, WINDOWS)
            for lab in WINDOWS:
                new[f"{drug}_inf_{lab}"] = cols[lab]
            print(f"  {drug:<16} {prevalence_line(new, f'{drug}_inf', WINDOWS)}   "
                  f"(zero-rate {st['zero_rate']:,}{' dropped' if FILTER_ZERO_RATE else ''}, "
                  f"open end {st['open_end']}, max dur {st['dur_h_max']}h)")

            # validate 24h against existing master columns (sedatives/opioids only)
            for c in existing_cols_for(drug, drug_cols):
                agree, n = agreement(m[c], new[f"{drug}_inf_24h"])
                if agree is not None:
                    print(f"      vs existing '{c}': {agree * 100:.2f}% agreement (n={n:,})")

    # infusion class aggregates
    for cls, drugs in CLASS_AGGREGATES:
        for lab in WINDOWS:
            agg = pd.concat([new[f"{d}_inf_{lab}"] for d in drugs], axis=1).max(axis=1).astype("Int64")
            agg[no_med] = pd.NA
            new[f"{cls}_inf_{lab}"] = agg
        print(f"  {cls.upper():<18} {prevalence_line(new, f'{cls}_inf', WINDOWS)}")

    # ---------------- opiate and benzodiazepine boluses ----------------
    bdir = cfg.get("bolus_dir")
    if not bdir:
        print("  WARNING: bolus exposure dir not set in med_config; bolus flags will be NA")
    bolus_items = [(cls, spec, drugs) for classes, drugs in BOLUS_GROUPS
                   for cls, spec in classes.items()]
    for cls, (suffix, existing_col), bolus_drugs in bolus_items:
        per_drug = []
        for drug in bolus_drugs:
            fpath = os.path.join(bdir, f"{drug}_boluses_{suffix}.csv") if bdir else None
            loaded = load_boluses(fpath)
            if loaded is None:
                print(f"  {drug:<16} {suffix} bolus file MISSING ({fpath}); treated as no events")
                continue
            ev, time_col = loaded
            cols = flags_to_cols(flag_drug(con, eeg, ev), m.index, no_med, BOLUS_WINDOWS)
            per_drug.append(cols)
            print(f"  {drug:<16} {suffix:<6} boluses: {len(ev):,} events (time col '{time_col}'); "
                  + "  ".join(f"{lab}={cols[lab].mean() * 100:5.1f}%" for lab in BOLUS_WINDOWS))
        for lab in BOLUS_WINDOWS:
            if per_drug:
                agg = pd.concat([c[lab] for c in per_drug], axis=1).max(axis=1).astype("Int64")
                agg[no_med] = pd.NA
            elif bdir:
                agg = pd.Series(0, index=m.index, dtype="Int64")
                agg[no_med] = pd.NA
            else:
                agg = na_series(m.index)
            new[f"{cls}_{lab}"] = agg
        print(f"  {cls.upper():<18} {prevalence_line(new, cls, BOLUS_WINDOWS)}")
        if existing_col in m.columns:
            agree, n = agreement(m[existing_col], new[f"{cls}_24h"])
            if agree is not None:
                flag = "" if agree >= 0.98 else "   <-- CHECK: definitions may differ"
                print(f"      vs existing '{existing_col}': {agree * 100:.2f}% agreement (n={n:,}){flag}")
        else:
            print(f"      (no existing '{existing_col}' column to validate against)")

    # ---------------- any opioid (infusion or bolus) ----------------
    for lab in BOLUS_WINDOWS:
        agg = pd.concat([new[f"opioid_inf_{lab}"],
                         new[f"opiate_fast_bolus_{lab}"],
                         new[f"opiate_slow_bolus_{lab}"]], axis=1).max(axis=1).astype("Int64")
        agg[no_med] = pd.NA
        new[f"opioid_any_{lab}"] = agg
    print(f"  {'OPIOID_ANY':<18} {prevalence_line(new, 'opioid_any', BOLUS_WINDOWS)}")

    # ---------------- opioid on board (covariate) ----------------
    # infusion running at t OR fast-acting dose in prior 6h OR slow-acting dose in prior 12h
    obb = pd.concat([new["opioid_inf_now"],
                     new["opiate_fast_bolus_6h"],
                     new["opiate_slow_bolus_12h"]], axis=1).max(axis=1).astype("Int64")
    obb[no_med] = pd.NA
    new["opioid_on_board"] = obb
    inf_only = (new["opioid_inf_now"] == 1)
    print(f"  {'OPIOID_ON_BOARD':<18} {obb.mean() * 100:5.1f}% of rows "
          f"(infusion running: {inf_only.mean() * 100:5.1f}%; "
          f"intermittent only: {((obb == 1) & ~inf_only.fillna(False)).mean() * 100:5.1f}%)")

    # ---------------- benzodiazepine on board (same rule) ----------------
    # midazolam infusion running at t OR fast-acting benzo dose in prior 6h OR slow-acting in prior 12h
    bbb = pd.concat([new["midazolam_inf_now"],
                     new["benzo_fast_bolus_6h"],
                     new["benzo_slow_bolus_12h"]], axis=1).max(axis=1).astype("Int64")
    bbb[no_med] = pd.NA
    new["benzo_on_board"] = bbb
    binf = (new["midazolam_inf_now"] == 1)
    print(f"  {'BENZO_ON_BOARD':<18} {bbb.mean() * 100:5.1f}% of rows "
          f"(infusion running: {binf.mean() * 100:5.1f}%; "
          f"intermittent only: {((bbb == 1) & ~binf.fillna(False)).mean() * 100:5.1f}%)")

    # sanity: windows must be nested
    labs = list(WINDOWS)
    for cls, _ in CLASS_AGGREGATES:
        for a, b in zip(labs, labs[1:]):
            viol = (new[f"{cls}_inf_{a}"] > new[f"{cls}_inf_{b}"]).fillna(False).sum()
            if viol:
                print(f"  WARNING: {cls} {a} > {b} in {int(viol)} rows")
    for cls in list(OPIATE_BOLUS_CLASSES) + list(BENZO_BOLUS_CLASSES) + ["opioid_any"]:
        for a, b in zip(BOLUS_WINDOWS, BOLUS_WINDOWS[1:]):
            viol = (new[f"{cls}_{a}"] > new[f"{cls}_{b}"]).fillna(False).sum()
            if viol:
                print(f"  WARNING: {cls} {a} > {b} in {int(viol)} rows")

    out = pd.concat([m, pd.DataFrame(new)], axis=1)
    out_path = os.path.splitext(path)[0] + "_windows_opiates.csv"
    tmp = out_path + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    print(f"  wrote {out_path}")


def main():
    con = duckdb.connect()
    if getattr(mc, "DUCKDB_TEMP_DIR", None):
        con.execute(f"SET temp_directory='{mc.DUCKDB_TEMP_DIR}'")
    for label, cfg in COHORTS.items():
        process(con, label, cfg)


if __name__ == "__main__":
    try:
        main()
        if winsound:
            winsound.MessageBeep(winsound.MB_OK)
        else:
            print("\a")
    except Exception:
        if winsound:
            winsound.MessageBeep(winsound.MB_ICONHAND)
        else:
            print("\a")
        raise