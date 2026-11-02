#!/usr/bin/env python3
"""EEG type by cohort from the HEEDB ServiceName field (R4.6).

Needs NESI_DATA_ROOT/intermediates/reports/*_eeg_metadata_*.csv, copied from
s3://bdsp-opendata-repository/EEG/eeg-metadata/. Writes a small per-recording file
under intermediates, so the table regenerates without the larger inputs.
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ["NESI_DATA_ROOT"])
INTER = ROOT / "intermediates"
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

meta = pd.concat(
    [pd.read_csv(f, low_memory=False, usecols=lambda c: c in {"SiteID", "BDSPPatientID", "SessionID", "StartTime", "EndTime", "DurationInSeconds", "ServiceName", "EEGFolder"}) for f in sorted((INTER / "reports").glob("*_eeg_metadata_*.csv"))],
    ignore_index=True,
)
meta["BDSPPatientID"] = meta.BDSPPatientID.astype(str).str.replace(r"\.0$", "", regex=True)
meta["SessionID"] = pd.to_numeric(meta.SessionID, errors="coerce")
meta = meta.dropna(subset=["SessionID"]).drop_duplicates(["BDSPPatientID", "SessionID", "SiteID"])

idx = pd.read_csv(ROOT / "s3/segment_index.csv", low_memory=False, usecols=["cohort", "subject_id", "SessionID", "DurationInSeconds", "resolved_BDSPPatientID", "source_EEGBeginDTS"])
idx["pid"] = idx.resolved_BDSPPatientID.fillna(idx.subject_id).astype(str).str.replace(r"\.0$", "", regex=True)
rec = idx.drop_duplicates(["cohort", "pid", "SessionID", "source_EEGBeginDTS"]).copy()
rec["SessionID"] = pd.to_numeric(rec.SessionID, errors="coerce")

# Session numbers repeat across sites, so accept a match only when the start time agrees
m = rec.merge(meta, left_on=["pid", "SessionID"], right_on=["BDSPPatientID", "SessionID"], how="left", suffixes=("", "_m"))
m["dt"] = (pd.to_datetime(m.source_EEGBeginDTS, errors="coerce") - pd.to_datetime(m.StartTime, errors="coerce")).abs()
m = m.sort_values("dt").drop_duplicates(["cohort", "pid", "SessionID", "source_EEGBeginDTS"])
m.loc[~(m.dt <= pd.Timedelta(hours=1)), "ServiceName"] = pd.NA
m["Type"] = m.ServiceName.fillna("Not recorded")
# CAM-S has no session numbers in the index, so match on patient and start time
c = pd.read_csv(ROOT / "s3/CAMS_i0001_HarvardEEG_metadata.csv")
c["pid"] = c.BDSPPatientID.astype(str)
c = c.merge(meta, left_on="pid", right_on="BDSPPatientID", how="left", suffixes=("", "_m"))
c["dt"] = (pd.to_datetime(c.SourceEEG_StartDTS, errors="coerce") - pd.to_datetime(c.StartTime, errors="coerce")).abs()
c = c.sort_values("dt").drop_duplicates(["pid", "SourceEEG_StartDTS"])
c.loc[~(c.dt <= pd.Timedelta(hours=1)), "ServiceName"] = pd.NA
c = c.assign(cohort="CAMS", source_EEGBeginDTS=c.SourceEEG_StartDTS, Type=c.ServiceName.fillna("Not recorded"))
m = pd.concat([m[m.cohort != "CAMS"], c], ignore_index=True)
m[["cohort", "pid", "SessionID", "source_EEGBeginDTS", "DurationInSeconds", "SiteID", "Type"]].to_csv(INTER / "eeg_type_per_recording.csv", index=False)

rows = []
for c, g in m.groupby("cohort"):
    v = g.Type.value_counts()
    known = g[g.Type != "Not recorded"]
    rows.append(dict(cohort=c, recordings=len(g), patients=g.pid.nunique(), **{k: int(v.get(k, 0)) for k in ["LTM", "Routine", "EMU", "OR", "Not recorded"]},
                     pct_LTM_of_known=round(100 * (known.Type == "LTM").mean(), 1) if len(known) else None,
                     pct_routine_of_known=round(100 * (known.Type == "Routine").mean(), 1) if len(known) else None,
                     median_h_LTM=round(known[known.Type == "LTM"].DurationInSeconds.median() / 3600, 1) if len(known) else None,
                     median_h_routine=round(known[known.Type == "Routine"].DurationInSeconds.median() / 3600, 2) if len(known) else None,
                     median_h_all=round(g.DurationInSeconds.median() / 3600, 1)))
r = pd.DataFrame(rows)
r.to_csv(OUT / "eeg_type_by_cohort.csv", index=False)
print(r.to_string(index=False))
