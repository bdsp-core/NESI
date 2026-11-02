# build_tables.py
"""
Main-text propofol table and supplementary tables S1-S6, written to one Word file.

Reads saved outputs only (no models are fit):
  agent_stratified_model.py  -> results_*.csv, report_*.md   (stratified_feasibility/model/...)
  variance_decomposition_eleveld_ce_vs_pump.py -> *_v3*.md   (OUTPUT_PARQUET)

Each table gets a title line and an empty note line for you to write.
Missing inputs are listed in the console and in the document; the rest still builds.
Rerun after any reanalysis. Needs: pip install python-docx

Tables
  Main  Propofol exposure-response and variance decomposition (two cohorts)
  S1    Strata: coverage and inclusion rule
  S2    Agent-stratified model, every stratum: offset, slope, slope difference
  S3    Pooled groups (main-text figure b, c) in numbers
  S4    Offset at each RASS level, pooled groups
  S5    Robustness of offsets (primary, capped, strict rule, light sedation); S5b same for slopes
  S6    Propofol sensitivity analyses (pump rate, no intermittent doses); S6b within/between Ce
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Repo layout: med_config.py lives in ../medication-exposure (no effect if already importable)
import sys as _sys
from pathlib import Path as _Path
_MC = _Path(__file__).resolve().parent.parent / "medication-exposure"
if _MC.is_dir() and str(_MC) not in _sys.path:
    _sys.path.insert(0, str(_MC))
from med_config import I0001_ASM_ANTIPSYCH_DIR, OUTPUT_PARQUET

# ============ PATHS ============
MODEL_DIR = Path(I0001_ASM_ANTIPSYCH_DIR) / "stratified_feasibility" / "model"
CAP_DIR   = MODEL_DIR / "cap_p95"                       # cap sensitivity run
VD_DIR    = Path(OUTPUT_PARQUET)                        # variance decomposition reports
VD_STEM   = "variance_decomposition_eleveld_ce_vs_pump_v3"
OUT_DOCX  = MODEL_DIR.parent / "tables" / "nesi_tables.docx"

# Expanded cohort (propofol with or without opioids): report from variance_decomposition_expanded_opioid.py
EXPANDED_REPORT = VD_DIR / "variance_decomposition_expanded_opioid_v2.md"
PRIMARY_LABEL = "Propofol only"

# ============ CONFIG ============
REF_LABEL  = "Reference (no sedative or opioid)"
VASO_LABEL = "Vasopressor (within reference)"
OLD_VASO   = ["Vasopressor, within reference (negative control)"]
POOLED_ORDER = ["Propofol (± opioid)", "Dexmedetomidine (± opioid)", "Midazolam (± opioid)",
                "Ketamine (± others)", "Midazolam + propofol (± opioid)", "Opioid only"]
STRATA_ORDER = ["propofol only", "propofol + opioid", "dexmedetomidine only", "dexmedetomidine + opioid",
                "midazolam only", "midazolam + opioid", "ketamine (+/- others)",
                "midazolam + propofol", "midazolam + propofol + opioid",
                "midazolam + dexmedetomidine", "midazolam + dexmedetomidine + opioid", "opioid only"]
LEVELS   = [-5, -4, -3, -2, -1, 0]           # -5 to 0, matching the figures
DEC      = 2                                 # decimals for offsets and slopes
FONT_PT  = 8                                 # table text
MINUS    = "−"

missing = []

# Note lines under tables (matched by the start of the table title); others get an empty "Note: ".
NOTES = {
    "Table S1": "*Pre-specified inclusion rule: ≥100 observations from ≥20 patients within the "
                "RASS range analyzed. Strata not meeting it were not modeled.",
    "Table S4": "— = fewer than 30 observations or 10 patients at that RASS level (pre-specified).",
    "Table X": "Cohort n = observations / patients. Mixed models with a patient random intercept; "
               "Ce centered at the cohort mean. Opioid = opioid on board (0/1). In the model with the "
               "interaction, the Ce slope is the slope with no opioid on board. % sedation = variance "
               "explained by all fixed effects. 1000-resample cluster bootstrap over patients.",
}


# ============ HELPERS ============
def num(x, d=DEC):
    return f"{x:.{d}f}".replace("-", MINUS)


def est_ci(e, lo, hi, d=DEC):
    """0.35 (0.29–0.40); 'to' when a bound is negative so the dash is not read as a minus."""
    if e is None or not np.isfinite(e):
        return "—"
    if lo is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return num(e, d)
    sep = " to " if (lo < 0 or hi < 0) else "–"
    return f"{num(e, d)} ({num(lo, d)}{sep}{num(hi, d)})"


def col_ci(r, k, d=DEC):
    return est_ci(r.get(k, np.nan), r.get(f"{k}_lo", np.nan), r.get(f"{k}_hi", np.nan), d)


def nice(name):
    if name in (REF_LABEL, VASO_LABEL):
        return name
    name = name.replace("+/-", "±")
    return name[0].upper() + name[1:]


def read_csv(path):
    if not Path(path).exists():
        if str(path) not in missing:
            missing.append(str(path))
        return None
    d = pd.read_csv(path)
    for c in ("agent", "group"):
        if c in d:
            d[c] = d[c].replace(OLD_VASO, VASO_LABEL)
    return d


def read_md(path):
    if not Path(path).exists():
        missing.append(str(path))
        return None
    return Path(path).read_text(encoding="utf-8")


def md_section_rows(text, heading_start):
    """Rows (list of cells) of the first markdown table after a heading that starts with heading_start."""
    if text is None:
        return []
    lines = text.splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.startswith("#") and l.lstrip("# ").startswith(heading_start))
    except StopIteration:
        return []
    rows, started = [], False
    for l in lines[i + 1:]:
        if l.startswith("|"):
            started = True
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
        elif started:
            break
    return rows[1:]   # drop header


def parse_ci(cell):
    """'0.3969 (0.3486 – 0.4230)%' or '+0.009 (-0.01 to +0.02)' -> (est, lo, hi)."""
    m = re.match(r"\s*([-+−]?\d*\.?\d+)\s*\(\s*([-+−]?\d*\.?\d+)\s*(?:–|to|-)\s*([-+−]?\d*\.?\d+)\s*\)", cell or "")
    if not m:
        return (np.nan, np.nan, np.nan)
    return tuple(float(g.replace("−", "-")) for g in m.groups())


def vd_components(text, heading, slope_dec=DEC):
    """{'NESI': {...}, 'RASS': {...}} from a 'Variance components: ...' section."""
    out = {"NESI": {}, "RASS": {}}
    for cells in md_section_rows(text, heading):
        key = {"% propofol": "pct_exposure", "% patient": "pct_patient",
               "% residual": "pct_residual", "slope": "slope"}.get(cells[0])
        if key:
            for o, c in (("NESI", cells[1]), ("RASS", cells[2])):
                e, lo, hi = parse_ci(c)
                out[o][key] = est_ci(e, lo, hi, 1 if key.startswith("pct") else slope_dec)
    return out


def vd_n(text):
    m = re.search(r"\*\*([\d,]+) rows\*\* across \*\*([\d,]+) patients\*\*", text or "")
    return f"{m.group(1)} / {m.group(2)}" if m else ""


# ============ TABLE BUILDERS (each returns title, header, rows) ============
def t_main():
    """Propofol-only cohort (Ce) and expanded cohort (Ce; + opioid; + Ce x opioid), NESI then RASS."""
    txt = read_md(VD_DIR / f"{VD_STEM}.md")
    prim = vd_components(txt, "Variance components: Eleveld Ce")
    exp_txt = read_md(EXPANDED_REPORT)
    exp_rows = md_section_rows(exp_txt, "All models")
    n_prim = vd_n(txt)
    m = re.search(r"\*\*([\d,]+) rows, ([\d,]+) patients\*\*", exp_txt or "")
    n_exp = f"{m.group(1)} / {m.group(2)}" if m else ""
    model_names = {"Ce only": "Ce", "Ce + opioid (no interaction)": "Ce + opioid",
                   "Sedation block (Ce + opioid + Ce×opioid)": "Ce + opioid + Ce × opioid"}
    header = ["Outcome", "Cohort", "Model", "Ce slope per µg/mL (95% CI)", "Opioid shift (95% CI)",
              "Ce × opioid (95% CI)", "% sedation (95% CI)", "% between-patient (95% CI)",
              "% residual (95% CI)"]
    rows = []
    for o in ("NESI", "RASS"):
        c = prim.get(o, {})
        rows.append([o, f"{PRIMARY_LABEL} ({n_prim})", "Ce", c.get("slope", ""), "—", "—",
                     c.get("pct_exposure", ""), c.get("pct_patient", ""), c.get("pct_residual", "")])
        first = True
        for cells in exp_rows:
            if len(cells) < 8 or cells[0] != o:
                continue
            ci_or_dash = lambda x, d: est_ci(*parse_ci(x), d) if x.strip() != "—" else "—"
            rows.append(["", f"Including opioid co-infusion ({n_exp})" if first else "",
                         model_names.get(cells[1], cells[1]),
                         ci_or_dash(cells[2], DEC), ci_or_dash(cells[3], DEC), ci_or_dash(cells[4], DEC),
                         ci_or_dash(cells[5], 1), ci_or_dash(cells[6], 1), ci_or_dash(cells[7], 1)])
            first = False
    return "Table X. Propofol exposure–response and variance decomposition", header, rows


def t_s1():
    res = read_csv(MODEL_DIR / "results_stratum_full.csv")
    rpt = read_md(MODEL_DIR / "report_stratum_full.md")
    if res is None:
        return None
    header = ["Stratum", "Observations", "Patients", "Met inclusion rule*"]
    rows = []
    m = re.search(r"n = ([\d,]+) obs, ([\d,]+) patients", rpt or "")
    if m:
        rows.append([REF_LABEL, m.group(1), m.group(2), "Yes"])
    res = res[res["agent"] != VASO_LABEL]
    for g in [s for s in STRATA_ORDER if s in set(res["agent"])] + \
             [s for s in res["agent"] if s not in STRATA_ORDER]:
        r = res[res["agent"] == g].iloc[0]
        rows.append([nice(g), f"{int(r['n_obs']):,}", f"{int(r['n_patients']):,}", "Yes"])
    # Strata below the inclusion rule were never modeled, so count them from the strata file
    m = re.search(r"Strata below the inclusion rule in this range: (.*)\.", rpt or "")
    if m and m.group(1).strip() != "none":
        dropped = [x.strip().strip("'") for x in m.group(1).split(",")]
        rows_csv = MODEL_DIR.parent / "strata_analysis_rows.csv"
        counts = None
        if rows_csv.exists():
            d = pd.read_csv(rows_csv, usecols=["_pid", "nesi", "rass", "stratum"], low_memory=False)
            d = d.dropna(subset=["nesi", "rass", "stratum"])
            d = d[d["rass"].between(-5, 0)]
            counts = d.groupby("stratum").agg(n_obs=("nesi", "size"), n_pat=("_pid", "nunique"))
        else:
            missing.append(str(rows_csv))
        for g in dropped:
            if counts is not None and g in counts.index:
                rows.append([nice(g), f"{int(counts.loc[g, 'n_obs']):,}",
                             f"{int(counts.loc[g, 'n_pat']):,}", "No"])
            else:
                rows.append([nice(g), "", "", "No"])
    return "Table S1. Sedation strata: coverage (RASS −5 to 0)", header, rows


def interaction_rows(rpt):
    """{agent: (slope cell, diff cell)} from the report's interaction-model table."""
    out = {}
    for cells in md_section_rows(rpt, "Supplement: interaction model"):
        if len(cells) >= 3:
            s, d = parse_ci(cells[1]), parse_ci(cells[2])
            out[cells[0].replace(OLD_VASO[0], VASO_LABEL)] = (est_ci(*s), est_ci(*d) if np.isfinite(d[0]) else "0")
    return out


def t_s2():
    res = read_csv(MODEL_DIR / "results_stratum_full.csv")
    rpt = read_md(MODEL_DIR / "report_stratum_full.md")
    if res is None:
        return None
    inter = interaction_rows(rpt)
    header = ["Stratum", "Patients", "Offset vs reference (95% CI)", "Slope, NESI per RASS point (95% CI)",
              "Slope difference from reference (95% CI)"]
    rows = [[REF_LABEL, "", "0", inter.get(REF_LABEL, ("—", ""))[0], "0"]]
    order = [s for s in STRATA_ORDER if s in set(res["agent"])] + \
            [s for s in res["agent"] if s not in STRATA_ORDER and s != VASO_LABEL] + [VASO_LABEL]
    for g in order:
        if g not in set(res["agent"]):
            continue
        r = res[res["agent"] == g].iloc[0]
        sl, df = inter.get(g, (col_ci(r, "slope"), "—"))
        rows.append([nice(g), f"{int(r['n_patients']):,}", col_ci(r, "add_offset"), sl, df])
    return "Table S2. Agent-stratified model, every stratum (RASS −5 to 0)", header, rows


def t_s3():
    # full model run writes slopes into results_main_*; a SLOPES_ONLY run writes results_main_slopes_*
    f = MODEL_DIR / "results_main_slopes_stratum_full.csv"
    m = read_csv(f if f.exists() else MODEL_DIR / "results_main_stratum_full.csv")
    ref = read_csv(MODEL_DIR / "ref_slope_main_stratum_full.csv")
    if m is None:
        return None
    header = ["Group", "Patients", "Offset vs reference (95% CI)", "Slope, NESI per RASS point (95% CI)",
              "Slope difference from reference (95% CI)"]
    rows = []
    if ref is not None:
        r = ref.iloc[0]
        rows.append([REF_LABEL, "", "0", est_ci(r["ref_slope"], r["ref_slope_lo"], r["ref_slope_hi"]), "0"])
    order = [g for g in POOLED_ORDER if g in set(m["agent"])] + [VASO_LABEL]
    for g in order:
        if g not in set(m["agent"]):
            continue
        r = m[m["agent"] == g].iloc[0]
        rows.append([g, f"{int(r['n_patients']):,}", col_ci(r, "add_offset"), col_ci(r, "slope"),
                     col_ci(r, "slope_diff")])
    return "Table S3. Pooled groups: offsets and slopes (Fig. Xb, c)", header, rows


def t_s4():
    lv = read_csv(MODEL_DIR / "results_by_level_stratum.csv")
    if lv is None:
        return None
    header = ["Group"] + [f"RASS {num(l, 0)}" for l in LEVELS]
    rows = []
    for g in [g for g in POOLED_ORDER if g in set(lv["group"])]:
        r = lv[lv["group"] == g].iloc[0]
        rows.append([g] + [col_ci(r, f"level{l}") for l in LEVELS])
    return "Table S4. Offset vs reference at each RASS level, pooled groups (95% CI)", header, rows


def t_s5(kind):
    runs = [("Primary", MODEL_DIR / "results_stratum_full.csv"),
            ("Capped (p95)", CAP_DIR / "results_stratum_full.csv"),
            ("Strict rule", MODEL_DIR / "results_stratum_strict_full.csv"),
            ("Light sedation (RASS −2 to 0)", MODEL_DIR / "results_stratum_light.csv")]
    data = [(n, read_csv(p)) for n, p in runs]
    data = [(n, d) for n, d in data if d is not None]
    if not data:
        return None
    key = "add_offset" if kind == "offset" else "slope"
    agents = set().union(*[set(d["agent"]) for _, d in data])
    order = [s for s in STRATA_ORDER if s in agents] + [VASO_LABEL]
    header = ["Stratum"] + [n for n, _ in data]
    rows = []
    for g in order:
        if g not in agents:
            continue
        row = [nice(g)]
        for _, d in data:
            r = d[d["agent"] == g]
            row.append(col_ci(r.iloc[0], key) if len(r) else "—")
        rows.append(row)
    if kind == "offset":
        return "Table S5. Robustness: offset vs reference (95% CI) across analyses", header, rows
    return "Table S5b. Robustness: slope, NESI per RASS point (95% CI) across analyses", header, rows


def t_s6():
    prim = read_md(VD_DIR / f"{VD_STEM}.md")
    noint = read_md(VD_DIR / f"{VD_STEM}_no_intermittent.md")
    header = ["Analysis", "Obs / patients", "Outcome", "Slope (95% CI)", "% exposure", "% between-patient",
              "% residual"]
    rows = []
    specs = [("Pump rate instead of Ce (per mg/kg/h)", prim, "Variance components: pump rate", 3),
             ("Ce, excluding intermittent benzodiazepine or opioid doses (per µg/mL)", noint,
              "Variance components: Eleveld Ce", DEC)]
    for label, txt, head, sd in specs:
        if txt is None:
            continue
        comp = vd_components(txt, head, sd)
        for i, o in enumerate(("NESI", "RASS")):
            c = comp[o]
            rows.append([label if i == 0 else "", vd_n(txt) if i == 0 else "", o, c.get("slope", ""),
                         c.get("pct_exposure", ""), c.get("pct_patient", ""), c.get("pct_residual", "")])
    return "Table S6. Propofol exposure–response: sensitivity analyses", header, rows


def t_s6b():
    rows = []
    for label, f in (("Primary", VD_DIR / f"{VD_STEM}_within_between.md"),):
        txt = read_md(f)
        if txt is None:
            continue
        for i, cells in enumerate(md_section_rows(txt, "Model")):
            rows.append([label if i == 0 else "", cells[0]] +
                        [est_ci(*parse_ci(c)) for c in cells[1:4]])
    if not rows:
        return None
    header = ["Cohort", "Outcome", "Within-patient slope per µg/mL (95% CI)",
              "Between-patient slope per µg/mL (95% CI)", "Between − within (95% CI)"]
    return "Table S6b. Within- and between-patient propofol Ce slopes", header, rows


# ============ WORD OUTPUT ============
def write_docx(tables, path):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.shared import Pt, Inches

    doc = Document()
    sec = doc.sections[0]                       # portrait, letter
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(0.7))
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(FONT_PT + 1)

    for k, t in enumerate(tables):
        title, header, rows = t
        p = doc.add_paragraph()
        p.add_run(title).bold = True
        tbl = doc.add_table(rows=1, cols=len(header))
        tbl.style = "Table Grid"
        for j, h in enumerate(header):
            cell = tbl.rows[0].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(h)
            run.bold = True
            run.font.size = Pt(FONT_PT)
        for r in rows:
            cells = tbl.add_row().cells
            for j, v in enumerate(r):
                cells[j].text = ""
                cells[j].paragraphs[0].add_run(str(v)).font.size = Pt(FONT_PT)
        note = next((v for k, v in NOTES.items() if title.startswith(k)), "")
        doc.add_paragraph(f"Note: {note}")
        if k < len(tables) - 1:
            doc.add_page_break()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def main():
    builders = [t_main, t_s1, t_s2, t_s3, t_s4, lambda: t_s5("offset"), lambda: t_s5("slope"), t_s6, t_s6b]
    tables = []
    for b in builders:
        t = b()
        if t is not None and t[2]:
            tables.append(t)
            print(f"  built: {t[0]}  ({len(t[2])} rows)")
    write_docx(tables, OUT_DOCX)
    # CSV copies, one per table, for anyone who wants the numbers outside Word
    csv_dir = OUT_DOCX.parent / "csv"
    csv_dir.mkdir(exist_ok=True)
    for title, header, rows in tables:
        name = re.sub(r"[^A-Za-z0-9]+", "_", title.split(".")[0]).strip("_")
        pd.DataFrame(rows, columns=header).to_csv(csv_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    if missing:
        print("\nInputs not found (tables built without them):")
        for m in missing:
            print(f"  {m}")
    print(f"\nSaved {OUT_DOCX.resolve()}")


if __name__ == "__main__":
    main()
