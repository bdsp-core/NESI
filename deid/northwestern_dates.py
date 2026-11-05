#!/usr/bin/env python3
"""Date-shift and rename the Northwestern NESI files for release.

Every date belonging to a patient moves by that patient's whole-week shift; times of day, durations
and every interval between two of the patient's timestamps are unchanged. Northwestern's recording
codes (RA0017E8, DA001008, ...) are replaced by local subject ids NW0001, NW0002, ..., assigned in
random order, so no Northwestern identifier survives in a name or a cell.

Each recording is a different patient (confirmed by Brandon Westover, 2026-11-04), so a recording's
shift is its patient's shift. Shifts are random nonzero multiples of 7 days in [-364, 364]: whole
weeks keep the day of the week, and zero would leave real dates in place.

The key (recording code -> subject id, shift) is the only link back to Northwestern's records. It is
written once, kept with the real-date files, and never committed or uploaded.

    northwestern_dates.py key --codes FILE_OR_DIR [...] --out KEY.csv
        one row per recording code found in the given tables (Filename column) and file names
    northwestern_dates.py tables --key KEY --src nesi-erika.xlsx --out OUT/nesi-erika.xlsx [--col C] [--dates C1,C2]
        a table, dates shifted and recordings renamed in every cell (nesi-erika.xlsx is what
        external_validation.py reads); --dates names columns that hold dates as text
    northwestern_dates.py rename --key KEY --src DIR --out DIR
        copies every file whose name starts with a recording code under its new name
        (*_ErikaSegment_*.mat, NESI predictions, MORGOTH activations); contents untouched
    northwestern_dates.py column --key KEY --src FILE --col Filename --out FILE.csv
        renames the recording codes in one column of a table that holds no dates
    northwestern_dates.py text --key KEY --src FILE.txt --out FILE.txt
        renames every recording code in a plain-text note

Each command checks its own output and stops on any mismatch.
"""
import argparse
import random
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

SHIFTS = [d for d in range(-364, 365, 7) if d != 0]
CODE = re.compile(r"^[A-Z]{2}[0-9A-Z]{6}$")
CODE_IN_TEXT = re.compile(r"\b[A-Z]{2}[0-9A-Z]{6}(?=_|\b)")
DATA = (".mat", ".csv")   # the data files in Erika's folders; anything else there is reported, not released


def code_of(name):
    """Northwestern recording code at the start of a file or segment name."""
    return re.split(r"[_.]", Path(str(name)).name, maxsplit=1)[0]


def read_table(path, **kw):
    return pd.read_excel(path, **kw) if str(path).endswith(("xlsx", "xls")) else pd.read_csv(path, **kw)


def data_files(src):
    files = [p for p in sorted(Path(src).rglob("*")) if p.is_file() and not p.name.startswith(".")]
    other = [p.name for p in files if p.suffix.lower() not in DATA]
    if other:
        print(f"  {src}: not data, left out: {other}")
    return [p for p in files if p.suffix.lower() in DATA]


def read_key(path):
    k = pd.read_csv(path, dtype={"recording": str, "subject_id": str})
    if k.recording.duplicated().any() or k.subject_id.duplicated().any():
        sys.exit("key lists a recording or a subject id twice")
    bad = k[(k.shift_days % 7 != 0) | (k.shift_days == 0) | (k.shift_days.abs() > 364)]
    if len(bad):
        sys.exit(f"{len(bad)} shifts are not nonzero whole weeks within a year")
    return k.set_index("recording")


def key(a):
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; the key is made once")
    codes = set()
    for s in map(Path, a.codes):
        if s.is_dir():
            codes |= {code_of(p.name) for p in data_files(s)}
        else:
            t = read_table(s, dtype=str)
            codes |= set(t[next(c for c in t.columns if c.lower() == "filename")].map(code_of))
    odd = sorted(c for c in codes if not CODE.match(c))
    if odd:
        sys.exit(f"{len(odd)} names do not start with a recording code, e.g. {odd[:3]}")
    rng = random.SystemRandom()
    order = sorted(codes)
    rng.shuffle(order)
    k = pd.DataFrame({"recording": order, "subject_id": [f"NW{i:04d}" for i in range(1, len(order) + 1)],
                      "shift_days": [rng.choice(SHIFTS) for _ in order]}).sort_values("recording")
    k.to_csv(out, index=False, lineterminator="\n")
    read_key(out)
    print(f"key for {len(k)} recordings written; shifts {k.shift_days.min()} .. {k.shift_days.max()} days")


def tables(a):
    key = read_key(a.key)
    text_dates = [c for c in (a.dates or "").split(",") if c]
    rd = dict(parse_dates=text_dates) if text_dates else {}
    if not str(a.src).endswith(("xlsx", "xls")):
        rd["float_precision"] = "round_trip"   # floats written back exactly as read
    x = read_table(a.src, **rd)
    x["code"] = x[a.col].astype(str).map(code_of)
    missing = set(x.code) - set(key.index)
    if missing:
        sys.exit(f"{len(missing)} recordings have no key row")
    when = [c for c in x.columns if pd.api.types.is_datetime64_any_dtype(x[c])]
    if set(text_dates) - set(when):
        sys.exit(f"{sorted(set(text_dates) - set(when))} did not parse as dates")
    for c in x.columns:
        if c not in when and not pd.api.types.is_numeric_dtype(x[c]) and x[c].astype(str).str.contains(
                r"\b(?:19|20)\d\d[-/.]\d\d?[-/.]\d\d?\b|\b\d\d?[/.]\d\d?[/.](?:19|20)\d\d\b").any():
            sys.exit(f"column {c} holds dates as text; name it in --dates")
    shift = pd.to_timedelta(x.code.map(key.shift_days), unit="D")
    y = x.copy()
    for c in when:
        y[c] = x[c] + shift
    # every recording code in a text cell becomes its subject id
    sub = lambda v: CODE_IN_TEXT.sub(lambda m: key.subject_id[m[0]] if m[0] in key.index else m[0], v) if isinstance(v, str) else v
    for c in y.columns:
        if c not in when and c != "code" and (y[c].dtype == object or pd.api.types.is_string_dtype(y[c])):
            y[c] = y[c].map(sub)
    if "segment_name" in y:
        # segment_name is <code>_<code-derived file id>_<segment>_...: the file id goes too
        y["segment_name"] = [f"{n.split('_', 1)[0]}_{n.split('_', 2)[2]}" for n in y.segment_name.astype(str)]
    y = y.drop(columns="code")
    # checks: each date moved by the patient's shift; clock times and every interval unchanged
    for c in when:
        ok = x[c].isna() & y[c].isna() | ((y[c] - x[c]) == shift)
        if not ok.all() or not (x[c].dt.time.fillna(0) == y[c].dt.time.fillna(0)).all():
            sys.exit(f"shift check failed in {c}")
    for i, c in enumerate(when):
        for d in when[i + 1:]:
            if not ((x[c] - x[d]).fillna(pd.Timedelta(0)) == (y[c] - y[d]).fillna(pd.Timedelta(0))).all():
                sys.exit(f"interval {c} - {d} changed")
    left = [c for c in y.columns if y[c].astype(str).str.contains("|".join(key.index)).any()]
    if left:
        sys.exit(f"recording codes left in {left}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (y.to_excel if out.suffix == ".xlsx" else y.to_csv)(out, index=False)
    back = read_table(out, **({k: v for k, v in rd.items()}))
    if list(back.columns) != list(y.columns) or not back.equals(y.astype(back.dtypes.to_dict())):
        sys.exit("written table does not read back identical")
    print(f"{len(y)} recordings; shifted {len(when)} datetime columns; checks passed")


def rename(a):
    key = read_key(a.key)
    src, out = Path(a.src), Path(a.out)
    n, skipped = 0, []
    for p in data_files(src):
        c = code_of(p.name)
        if c not in key.index:
            skipped.append(p.name)
            continue
        q = out / p.relative_to(src).parent / (key.subject_id[c] + p.name[len(c):])
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        if q.stat().st_size != p.stat().st_size:
            sys.exit(f"{q} copied short")
        n += 1
    print(f"{n} files renamed into {out}")
    if skipped:
        sys.exit(f"{len(skipped)} files carry no known recording code, e.g. {skipped[:3]}; check them before release")


def column(a):
    key = read_key(a.key)
    # every cell as text: re-parsing floats would rewrite their last digits
    t = read_table(a.src, dtype=str, keep_default_na=False)
    codes = t[a.col].map(code_of)
    if not codes.isin(key.index).all():
        sys.exit(f"{(~codes.isin(key.index)).sum()} rows have no new name")
    t[a.col] = [key.subject_id[c] + s[len(c):] for c, s in zip(codes, t[a.col])]
    left = [c for c in t.columns if t[c].str.contains("|".join(key.index)).any()]
    if left:
        sys.exit(f"recording codes left in {left}")
    t.to_csv(a.out, index=False, lineterminator="\n")
    print(f"{len(t)} rows renamed in {a.col}")


def text(a):
    key = read_key(a.key)
    t = Path(a.src).read_text()
    found = CODE_IN_TEXT
    unknown = {c for c in found.findall(t) if c not in key.index}
    if unknown:
        sys.exit(f"{len(unknown)} code-like words have no key row, e.g. {sorted(unknown)[:3]}")
    t = found.sub(lambda m: key.subject_id[m[0]], t)
    Path(a.out).write_text(t)
    print(f"codes renamed in {a.src}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("key"); s.add_argument("--codes", nargs="+", required=True); s.add_argument("--out", required=True)
    for name in ["tables", "rename", "column", "text"]:
        s = sub.add_parser(name); s.add_argument("--key", required=True); s.add_argument("--src", required=True); s.add_argument("--out", required=True)
        if name in ("column", "tables"):
            s.add_argument("--col", default="Filename", help="the column naming each recording")
        if name == "tables":
            s.add_argument("--dates", help="comma-separated columns that hold dates as text")
    a = ap.parse_args()
    {"key": key, "tables": tables, "rename": rename, "column": column, "text": text}[a.cmd](a)


if __name__ == "__main__":
    main()
