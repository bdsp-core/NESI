#!/usr/bin/env python3
"""Date-shift and rename the Northwestern (BDSP site I0025) NESI files for release.

Every date belonging to a patient moves by that patient's whole-week shift; times of day, durations
and every interval between two of the patient's timestamps are unchanged. Northwestern's recording
codes (RA0017E8, DA001008, ...) are replaced by sub-<bdsp_subject_id>_ses-<k>, k numbering the
patient's recordings in time order, so no Northwestern identifier survives in a name.

The key comes from db_phi_aurora's loom/mint_northwestern_ids.py --key-out: one row per recording,
`recording,bdsp_subject_id,shift_days`, the same id and shift stored for the patient in the PHI OMOP
crosswalk. The key, names.csv and every input here are PHI-linked. Run this only where the PHI is
(the PHI account or the encrypted SSD), and never commit or upload anything but the outputs marked
release.

    northwestern_dates.py tables --key KEY --src nesi-erika.xlsx --out OUT
        OUT/release/assessments.csv   the table, shifted and renamed (what external_validation.py reads)
        OUT/release/deid/recordings.parquet, assessments.parquet   for the de-identified OMOP handoff
        OUT/private/names.csv         recording -> new name (PHI-linked; stays with the key)
    northwestern_dates.py rename --names OUT/private/names.csv --src DIR --out DIR
        copies every file whose name starts with a recording code under its new name
        (*_ErikaSegment_*.mat, NESI predictions, MORGOTH activations); contents untouched
    northwestern_dates.py edf --key KEY --names OUT/private/names.csv --src DIR --out DIR
        EDF/EDF+ headers: start date shifted, patient and recording fields replaced; signal bytes
        copied unchanged; EDF+ annotation texts emptied, onsets kept (--annotations keep to keep them)
    northwestern_dates.py column --names OUT/private/names.csv --src FILE.csv --col Filename --out FILE.csv
        renames the recording codes in one column (recordings.csv, demographics.csv)

Each command checks its own output and stops on any mismatch.
"""
import argparse
import csv
import hashlib
import re
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

EDF_YEARS = (1985, 2084)   # EDF's two-digit start-date field cannot hold any other year


def code_of(name):
    """Northwestern recording code at the start of a file or segment name."""
    return re.split(r"[_.]", Path(str(name)).name, maxsplit=1)[0]


def read_key(path):
    k = pd.read_csv(path, dtype={"recording": str})
    if k.recording.duplicated().any():
        sys.exit("key lists a recording twice")
    bad = k[(k.shift_days % 7 != 0) | (k.shift_days == 0) | (k.shift_days.abs() > 364)]
    if len(bad):
        sys.exit(f"{len(bad)} shifts are not nonzero whole weeks within a year")
    if k.groupby("bdsp_subject_id").shift_days.nunique().max() > 1:
        sys.exit("a patient has two different shifts in the key")
    return k.set_index("recording")


def read_names(path):
    return dict(csv.reader(open(path)))


def tables(a):
    key = read_key(a.key)
    x = pd.read_excel(a.src) if a.src.endswith(("xlsx", "xls")) else pd.read_csv(a.src)
    x["code"] = x.Filename.astype(str).map(code_of)
    missing = set(x.code) - set(key.index)
    if missing:
        sys.exit(f"{len(missing)} recordings have no key row")
    when = [c for c in x.columns if pd.api.types.is_datetime64_any_dtype(x[c])]
    for c in x.columns:
        if c not in when and x[c].dtype == object and x[c].astype(str).str.contains(r"\b(?:19|20)\d\d[-/.]\d\d?[-/.]\d\d?\b|\b\d\d?[/.]\d\d?[/.](?:19|20)\d\d\b").any():
            sys.exit(f"column {c} holds dates as text; parse it before shifting")
    x["bdsp_subject_id"] = x.code.map(key.bdsp_subject_id)
    shift = pd.to_timedelta(x.code.map(key.shift_days), unit="D")
    # sessions numbered by start time within each patient (the shift is the same for all of them)
    x = x.sort_values(["bdsp_subject_id", "EEG_StartDateTime", "code"])
    x["session"] = x.groupby("bdsp_subject_id").cumcount() + 1
    x["new"] = "sub-" + x.bdsp_subject_id.astype(str) + "_ses-" + x.session.astype(str)
    y = x.copy()
    for c in when:
        y[c] = x[c] + shift
    y["Filename"] = y.new
    y["segment_name"] = [f"{n}_{s.split('_', 2)[2]}" if s.count("_") >= 2 else n for n, s in zip(y.new, y.segment_name.astype(str))]
    # checks: each date moved by the patient's shift; clock times and every interval unchanged
    for c in when:
        ok = x[c].isna() & y[c].isna() | ((y[c] - x[c]) == shift)
        if not ok.all() or not (x[c].dt.time.fillna(0) == y[c].dt.time.fillna(0)).all():
            sys.exit(f"shift check failed in {c}")
    for i, c in enumerate(when):
        for d in when[i + 1:]:
            if not ((x[c] - x[d]).fillna(pd.Timedelta(0)) == (y[c] - y[d]).fillna(pd.Timedelta(0))).all():
                sys.exit(f"interval {c} - {d} changed")
    out = Path(a.out)
    (out / "release" / "deid").mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(parents=True, exist_ok=True)
    y.drop(columns=["code", "new", "session", "bdsp_subject_id"]).to_csv(out / "release" / "assessments.csv", index=False)
    with open(out / "private" / "names.csv", "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(sorted(zip(x.code, x.new)))
    # de-identified OMOP handoff: one row per recording, one per assessment
    rec = y[["bdsp_subject_id", "session", "Filename", "EEG_StartDateTime", "EEG_segment_StartdateTime",
             "Segment duration_s", "Segmentduration_sec_AfterAutoreject"]].rename(columns={"Filename": "recording"})
    rec.to_parquet(out / "release" / "deid" / "recordings.parquet", index=False)
    long = []
    for scale, t, v in [("GCS", "GCS_closest_DateTime", "GCS_closest"), ("RASS", "RASS_closest_Datetime", "RASS_numeric_closest"),
                        ("CAM-ICU", "AnyCAM_closest_Datetime", "AnyCAM_closestScore")]:
        long.append(pd.DataFrame({"bdsp_subject_id": y.bdsp_subject_id, "recording": y.Filename, "scale": scale,
                                  "assessment_datetime": y[t], "value": y[v].astype(str)}))
    pd.concat(long).to_parquet(out / "release" / "deid" / "assessments.parquet", index=False)
    print(f"{len(y)} recordings, {y.bdsp_subject_id.nunique()} patients; shifted {len(when)} datetime columns; checks passed")


def rename(a):
    names = read_names(a.names)
    src, out = Path(a.src), Path(a.out)
    n = skipped = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        c = code_of(p.name)
        if c not in names:
            skipped += 1
            continue
        q = out / p.relative_to(src).parent / (names[c] + p.name[len(c):])
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    print(f"{n} files renamed into {out}; {skipped} files without a recording code left out")
    if skipped:
        sys.exit("some files carry no known recording code; check them before release")


def edf_date(h):
    """Start date of an EDF/EDF+ header. EDF+ keeps the 4-digit year in the recording field."""
    m = re.match(r"Startdate (\d\d)-([A-Z]{3})-(\d{4})", h[88:168].decode("ascii", "replace"))
    if h[192:197] in (b"EDF+C", b"EDF+D") and m:
        return datetime.strptime(f"{m[1]}-{m[2].title()}-{m[3]}", "%d-%b-%Y").date()
    dd, mm, yy = (int(v) for v in h[168:176].decode().split("."))
    return date(1900 + yy if yy >= 85 else 2000 + yy, mm, dd)


DATE_IN_TEXT = re.compile(rb"\b(?:19|20)\d\d[-/.]\d\d?[-/.]\d\d?\b|\b\d\d?[-/.]\d\d?[-/.](?:19|20)?\d\d\b|"
                          rb"\b\d\d?[- ](?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[- ]\d{2,4}\b", re.I)


def blank_annotations(raw, nhead):
    """EDF+ annotation texts replaced by "X" in place; onsets, durations and record sizes unchanged.

    A TAL is onset[\\x15duration]\\x14text\\x14...\\x14\\x00. One with text becomes
    onset[\\x15duration]\\x14X\\x14 (readers skip annotations with empty text, so "X" keeps the
    event visible); the time-keeping TAL, which has none, stays onset\\x14\\x14. Each record is
    padded with \\x00 to its old length, which "X" always fits.
    """
    ns = int(raw[252:256])
    labels = [raw[256 + i * 16:256 + (i + 1) * 16].strip() for i in range(ns)]
    if b"EDF Annotations" not in labels:
        return raw, 0
    nsamp = [int(raw[256 + ns * 216 + i * 8:256 + ns * 216 + (i + 1) * 8]) for i in range(ns)]
    nrec, rec = int(raw[236:244]), sum(nsamp) * 2
    out, n = bytearray(raw), 0
    for i, lab in enumerate(labels):
        if lab != b"EDF Annotations":
            continue
        off = sum(nsamp[:i]) * 2
        for r in range(nrec):
            s = nhead + r * rec + off
            block = bytes(out[s:s + nsamp[i] * 2])
            new = b""
            for tal in block.split(b"\x00"):
                if not tal:
                    continue
                head, _, texts = tal.partition(b"\x14")
                k = sum(1 for t in texts.split(b"\x14") if t)
                n += k
                new += head + (b"\x14X\x14\x00" if k else b"\x14\x14\x00")
            out[s:s + len(block)] = new.ljust(len(block), b"\x00")
    return bytes(out), n


def edf(a):
    key, names = read_key(a.key), read_names(a.names)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() != ".edf":
            continue
        c = code_of(p.name)
        if c not in key.index or c not in names:
            sys.exit(f"{p.name}: no key row")
        days = int(key.loc[c, "shift_days"])
        raw = p.read_bytes()
        nhead = int(raw[184:192])
        h = bytearray(raw[:nhead])
        d0 = edf_date(h)
        d1 = d0 + timedelta(days=days)
        if not EDF_YEARS[0] <= d1.year <= EDF_YEARS[1]:
            sys.exit(f"{p.name}: shifted year {d1.year} cannot be stored in EDF")
        plus = h[192:197] in (b"EDF+C", b"EDF+D")
        h[8:88] = f"{names[c]} X X X".ljust(80).encode()
        rec = f"Startdate {d1.strftime('%d')}-{d1.strftime('%b').upper()}-{d1.year} X X X" if plus else "X"
        h[88:168] = rec.ljust(80).encode()
        h[168:176] = d1.strftime("%d.%m.%y").encode()
        # EDF+ annotations: onsets are relative to the start and stay; their text may hold dates
        # and names, so it is emptied unless --annotations keep (after a human has read it)
        hits = len(DATE_IN_TEXT.findall(raw[nhead:])) if plus else 0
        body, blanked = blank_annotations(raw, nhead) if plus and a.annotations == "drop" else (raw, 0)
        q = out / (names[c] + ".edf")
        q.write_bytes(bytes(h) + body[nhead:])
        # checks: the date moved by the shift, the clock time and every signal byte did not
        g = q.read_bytes()
        assert edf_date(g[:nhead]) == d1 and g[176:184] == raw[176:184]
        if not blanked:
            assert hashlib.sha256(g[nhead:]).digest() == hashlib.sha256(raw[nhead:]).digest()
        report.append((names[c], hits, blanked))
    with open(out / "edf_report.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", "dates_in_annotation_text", "annotation_texts_emptied"])
        w.writerows(report)
    print(f"{len(report)} EDFs shifted and checked; {sum(1 for _, h, _ in report if h)} had date-like annotation "
          f"text; {sum(b for *_, b in report)} annotation texts emptied (edf_report.csv)")


def column(a):
    names = read_names(a.names)
    # every cell as text: re-parsing floats would rewrite their last digits
    t = pd.read_csv(a.src, dtype=str, keep_default_na=False)
    codes = t[a.col].map(code_of)
    if not codes.isin(names).all():
        sys.exit(f"{(~codes.isin(names)).sum()} rows have no new name")
    t[a.col] = [names[c] + s[len(c):] for c, s in zip(codes, t[a.col])]
    t.to_csv(a.out, index=False)
    print(f"{len(t)} rows renamed in {a.col}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("tables"); s.add_argument("--key", required=True); s.add_argument("--src", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("rename"); s.add_argument("--names", required=True); s.add_argument("--src", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("edf"); s.add_argument("--key", required=True); s.add_argument("--names", required=True); s.add_argument("--src", required=True); s.add_argument("--out", required=True)
    s.add_argument("--annotations", choices=["drop", "keep"], default="drop", help="empty EDF+ annotation texts (default) or keep them")
    s = sub.add_parser("column"); s.add_argument("--names", required=True); s.add_argument("--src", required=True); s.add_argument("--col", default="Filename"); s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"tables": tables, "rename": rename, "edf": edf, "column": column}[a.cmd](a)


if __name__ == "__main__":
    main()
