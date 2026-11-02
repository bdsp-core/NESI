#!/usr/bin/env python3
"""Classify EEG indications from report text with documented keyword rules (R1.1, Table 1, Supp Fig 27).

The published labels came from an LLM (Qwen3; column IndicationNeuroreport_LLM in
NESI/PacMAP/NESI_composite_pacmap.py), whose output files are not available. This is
a transparent replacement that can be compared with them. Categories are not exclusive.

Needs report zips in NESI_DATA_ROOT/intermediates/reports/neuro and the
*_EEG_*reports_findings.csv links in intermediates/reports. Saves per-report labels
under intermediates; reports themselves are not copied anywhere else.
"""
import glob
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
REP = ROOT / "intermediates" / "reports"
OUT = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "intermediates" / "indication_labels.csv"

SECTION = re.compile(
    r"(?:REASON FOR (?:STUDY|EXAM|EEG)|INDICATIONS?(?: FOR (?:STUDY|EEG))?|CLINICAL HISTORY|HISTORY)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z /]{3,40}:|\Z)",
    re.S,
)
RULES = {
    "AMS": r"altered mental|\bams\b|mental status|encephalopath|unrespons|obtund|letharg|somnolen|confus|delirium|\bcoma|comatose|not waking|fail(?:ure)? to wake|poor(?:ly)? respons",
    "Seizure": r"seizure|\bsz\b|szs\b|spell|convuls|twitch|jerk|staring|shaking|myoclon|events?\b",
    "Arrest": r"cardiac arrest|\barrest\b|post[- ]arrest|\bpea\b|\bvf\b|v[- ]?fib|anoxic|hypoxic[- ]ischemic|\brosc\b|cooling|hypothermia protocol|\bttm\b",
    "Epilepsy": r"epilep|status epilepticus|\bse\b|breakthrough|\baed\b|\basm\b",
    "Stroke": r"stroke|\bcva\b|\bich\b|\bsah\b|\bsdh\b|\bivh\b|hemorrhage|haemorrhage|infarct|ischemi|\bmca\b|\baca\b|\bpca\b|aneurysm|thrombect",
}
RULES = {k: re.compile(v, re.I) for k, v in RULES.items()}


def labels(text):
    sec = " ".join(m.group(1) for m in SECTION.finditer(text))
    return {"has_section": bool(sec.strip()), **{k: bool(p.search(sec)) for k, p in RULES.items()}}


def build_cache():
    names = pd.concat(
        [pd.read_csv(f, usecols=["DeidentifiedName(Reports)"], low_memory=False) for f in glob.glob(str(REP / "*reports_findings.csv"))]
    )["DeidentifiedName(Reports)"].dropna()
    want, rows = set(names), []
    for z in sorted(glob.glob(str(REP / "neuro" / "*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                b = os.path.basename(n)
                if b in want:
                    rows.append(dict(report=b, **labels(zf.read(n).decode("latin1"))))
    d = pd.DataFrame(rows).drop_duplicates("report")
    d.to_csv(CACHE, index=False)
    return d


lab = pd.read_csv(CACHE) if CACHE.exists() else build_cache()
print(len(lab), "reports labeled;", int(lab.has_section.sum()), "with an indication or history section")

link = pd.concat(
    [pd.read_csv(f, usecols=["BDSPPatientID", "SessionID", "DeidentifiedName(Reports)"], low_memory=False) for f in glob.glob(str(REP / "*reports_findings.csv"))]
).rename(columns={"DeidentifiedName(Reports)": "report"})
link["pid"] = link.BDSPPatientID.astype(str).str.replace(r"\.0$", "", regex=True)
link = link.merge(lab, on="report")
cat = list(RULES)
per = link.groupby("pid")[cat + ["has_section"]].max()

f = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
f["pid"] = f.BDSPPatientID.astype(str).str.replace(r"\.0$", "", regex=True)
PUBLISHED = {"RASS": dict(AMS=5551, Seizure=4280, Arrest=1035, Epilepsy=761, Stroke=2221, Other=76), "GCS": dict(AMS=6204, Seizure=4859, Arrest=843, Epilepsy=969, Stroke=2708, Other=154), "CAMS": dict(AMS=290, Seizure=186, Arrest=46, Epilepsy=39, Stroke=85, Other=137)}
rows = []
for ds in ["RASS", "GCS", "CAMS"]:
    u = pd.Index(f[f.Dataset == ds].pid.unique())
    p = per.reindex(u)
    row = dict(Scale=ds, Patients=len(u), Matched=int(p.has_section.notna().sum()))
    for k in cat:
        row[k] = int(p[k].fillna(False).astype(bool).sum())
        row[k + "_published"] = PUBLISHED[ds][k]
    row["Other"] = int((~p[cat].fillna(False).astype(bool).any(axis=1)).sum())
    row["Other_published"] = PUBLISHED[ds]["Other"]
    rows.append(row)
c = pd.DataFrame(rows)
c.to_csv(OUT / "indication_counts.csv", index=False)
print(c.T.to_string())


def rho_ci(d, boot=1000):
    rng = np.random.default_rng(2026)
    ids = {k: np.asarray(v) for k, v in d.groupby("pid").indices.items()}
    keys, y, x, b = list(ids), d.TransformedScore.values, d.NESI.values, []
    for _ in range(boot):
        i = np.concatenate([ids[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        if len(np.unique(y[i])) > 1:
            b.append(abs(stats.spearmanr(y[i], x[i])[0]))
    return abs(stats.spearmanr(y, x)[0]), np.percentile(b, 2.5), np.percentile(b, 97.5)


t = f[f.Split == "Test"].join(per, on="pid")
rows = []
for ds in ["RASS", "GCS", "CAMS"]:
    g = t[t.Dataset == ds]
    for k in cat:
        v = g[g[k].eq(True)]
        if v.pid.nunique() >= 10:
            a, lo, hi = rho_ci(v)
            rows.append(dict(Scale=ds, Indication=k, Segments=len(v), Patients=v.pid.nunique(), rho=round(a, 2), lo=round(lo, 2), hi=round(hi, 2)))
r = pd.DataFrame(rows)
r.to_csv(OUT / "indication_rho_test.csv", index=False)
print(r.to_string(index=False))
