# agent_stratified_model.py
"""
Agent-stratified analysis of NESI vs RASS (reviewer R2.5 / R3.11, R1.2).

Reads the analysis rows written by build_strata.py (strata are defined
there only), so run that script first.

Analyses (patient random intercept throughout):
  A. Interaction : NESI ~ RASS + agent + agent x RASS
       offset(agent, r) = gap vs reference at matched RASS r; slope(agent) = NESI per RASS point
  B. Additive    : NESI ~ RASS + agent
       one offset per agent, also in RASS points (offset / -reference slope)
  C. Offset by RASS level (no linearity assumption): NESI ~ pooled group, fit separately at
       each RASS level, using the pooled groups in POOLED (each agent with and without an
       opioid infusion) vs the reference; a cell is estimated only if it has >= MIN_CELL_OBS
       obs and >= MIN_CELL_PAT patients in both the pooled group and the reference group
  Negative control: vasopressor vs none within the reference group, same models.

  A and B are run on the full RASS range (-5 to 0) and on light sedation (-2 to 0), where
  unsedated and sedated patients are more comparable (unsedated RASS -4/-5 is largely
  coma from brain injury). A and B keep every stratum separate. C runs on the full range.

Descriptive figures (no bootstrap), pooled groups plus the reference only:
  - diagnostic plot: mean NESI at each RASS level, one line per group (patient means)
  - NESI distribution at RASS -4 and -5 by group (one point per patient per level)

Reference: "no sedative or opioid" (no sedative or opioid infusion, no intermittent
benzodiazepine dose). Strata enter a range only if they pass the pre-specified rule
(>= MIN_OBS observations and >= MIN_PATIENTS patients) within that range.

CIs: patient-level cluster bootstrap, all models refit on each resample.
Offsets are descriptive EEG-exam discrepancies, not drug effects.

Run with Play. Paths are printed at start.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

# Repo layout: med_config.py lives in ../medication-exposure (no effect if already importable)
import sys as _sys
from pathlib import Path as _Path
_MC = _Path(__file__).resolve().parent.parent / "medication-exposure"
if _MC.is_dir() and str(_MC) not in _sys.path:
    _sys.path.insert(0, str(_MC))
from med_config import I0001_ASM_ANTIPSYCH_DIR

# ============ PATHS ============
IN_DIR  = Path(I0001_ASM_ANTIPSYCH_DIR) / "stratified_feasibility"
IN_ROWS = IN_DIR / "strata_analysis_rows.csv"
# Sensitivity: drop every patient with overlapping EEG files (all their rows).
# Outputs then go to model/no_overlap_patients so the primary results are not overwritten.
DROP_OVERLAP_PATIENTS = False

# Command-line flags (run_stratified_all.py uses these; no flags = primary analysis as configured below)
import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=True)
_ap.add_argument("--cap", default=None, help='per-patient cap: "p95" or an integer')
_ap.add_argument("--strata", choices=["primary", "strict", "no_washout"], default=None,
                 help="primary = stratum; strict = stratum_strict (intermittent opioid doses also exclude); "
                      "no_washout = stratum_no_washout (infusions count only if running at the assessment)")
_ap.add_argument("--plots-only", action="store_true", help="redraw figures from saved results; no models")
_ap.add_argument("--slopes-only", action="store_true", help="bootstrap only the pooled-group slopes")
_ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples (default 1000)")
_ap.add_argument("--n-jobs", type=int, default=None)
ARGS, _ = _ap.parse_known_args()
# Sensitivity: keep at most CAP_PER_PATIENT observations per patient within each stratum, so a
# few long-monitored patients cannot dominate a small stratum. Selection is random (fixed seed)
# but stratified by RASS: each patient keeps the same mix of RASS levels as in their full data
# (largest-remainder allocation).
#   None  = no cap (primary)
#   "p95" = K is the 95th percentile of obs per patient in the reference group, rounded
#           (pre-specified rule, from counts only)
#   an integer = fixed K
# Outputs go to model/cap_<setting>.
CAP_PER_PATIENT = None
if ARGS.cap:
    CAP_PER_PATIENT = int(ARGS.cap) if ARGS.cap.isdigit() else ARGS.cap
_sub = ("no_overlap_patients" if DROP_OVERLAP_PATIENTS else "") + \
       (("_" if DROP_OVERLAP_PATIENTS else "") + f"cap_{CAP_PER_PATIENT}" if CAP_PER_PATIENT else "")
# Smoke tests (--n-boot under 100) write to model/smoke/ so they never overwrite real results
if ARGS.n_boot is not None and ARGS.n_boot < 100:
    _sub = str(Path("smoke") / _sub) if _sub else "smoke"
OUT_DIR = IN_DIR / "model" / _sub
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ CONFIG ============
STRATUM_COL   = "stratum"            # "stratum_strict" for the strict sensitivity analysis
if ARGS.strata:
    STRATUM_COL = {"primary": "stratum", "strict": "stratum_strict",
                   "no_washout": "stratum_no_washout"}[ARGS.strata]
REFERENCE     = "no sedative or opioid"                  # stratum value in the data
REF_LABEL     = "Reference (no sedative or opioid infusion)"      # how it is labeled in output
VASO_LABEL    = "Vasopressor (within reference)"
OLD_VASO_LABELS = ["Vasopressor, within reference (negative control)"]   # renamed in saved results
# Strata shown in the forest and slopes figures, top to bottom (tables keep every stratum).
# Combination strata (midazolam + propofol/dex) are left out: wide CIs, not interpreted.
FIGURE_STRATA = ["propofol only", "propofol + opioid", "midazolam only", "midazolam + opioid",
                 "dexmedetomidine only", "dexmedetomidine + opioid", "ketamine only", "ketamine + opioid",
                 "opioid only"]
# Groups shown in the slopes figure and the combined main figure, top to bottom: the performance-metric groups (pooled, main-text model)
SLOPE_FIGURE_GROUPS = ["Propofol (± opioid)", "Dexmedetomidine (± opioid)", "Midazolam (± opioid)",
                       "Ketamine (± opioid)", "Midazolam + propofol (± opioid)", "Opioid only"]   # a group appears only if it meets the inclusion rule
EXCLUDE       = {"intermittent dose involved", "unknown medication status",
                 "other combinations"}   # dropped Oct 29 after seeing output: mixed combinations, not interpretable
# Pooled groups for analysis C and the descriptive figures (fixed in advance, not data-driven).
# Member names must match the stratum values in the data exactly.
POOLED = {
    "Propofol (± opioid)":        ["propofol only", "propofol + opioid"],
    "Midazolam (± opioid)":       ["midazolam only", "midazolam + opioid"],
    "Dexmedetomidine (± opioid)": ["dexmedetomidine only", "dexmedetomidine + opioid"],
    "Midazolam + propofol (± opioid)": ["midazolam + propofol", "midazolam + propofol + opioid"],
    "Ketamine (± opioid)":        ["ketamine only", "ketamine + opioid"],
    "Opioid only":                ["opioid only"],             # opioid infusion, no sedative
}
OPIOID_ONLY   = "opioid only"         # shown as its own row in the main-text model
POOL_OF       = {s: p for p, members in POOLED.items() for s in members}
MIN_OBS, MIN_PATIENTS = 100, 20      # pre-specified, applied within each RASS range
MIN_CELL_OBS, MIN_CELL_PAT = 30, 10  # per RASS level, for analysis C
RANGES        = {"full": (-5, 0), "light": (-2, 0)}
RASS_CENTER   = -2                   # centering point for model A offsets
RASS_REPORT   = {"full": [0, -1, -2, -3, -4, -5], "light": [0, -1, -2]}
LEVELS        = [0, -1, -2, -3, -4, -5]
LEVEL_PLOT_SKIP = set()   # groups left off ALL plots (still in the tables); none since Nov 2
PLOTS_ONLY      = False   # True = redraw plots (diagnostic, deep-RASS, forest from saved results) and stop; no models
SLOPES_ONLY     = False   # True = bootstrap ONLY the pooled-group slope model; offsets read from saved
                          # results_main_*.csv; writes results_main_slopes_*.csv (nothing overwritten)
N_BOOTSTRAP   = 1000              # smoke test: run with --n-boot 2
N_JOBS        = -2
PLOTS_ONLY  = PLOTS_ONLY or ARGS.plots_only
SLOPES_ONLY = SLOPES_ONLY or ARGS.slopes_only
N_BOOTSTRAP = ARGS.n_boot or N_BOOTSTRAP
N_JOBS      = ARGS.n_jobs or N_JOBS
RNG_SEED      = 42

BLUE, ORANGE, GRAY = "#2F6F8F", "#C8702A", "#8A929B"
# One color per group, used in every figure (matched to co-author's figures). Strata take their
# pooled group's color (propofol only and propofol + opioid are both propofol blue).
GROUP_COLORS = {
    REFERENCE:                         "#000000",
    "Propofol (± opioid)":             "#0978B6",
    "Dexmedetomidine (± opioid)":      "#0BA37B",
    "Midazolam (± opioid)":            "#DB7627",
    "Opioid only":                     "#9E75B4",
    "Ketamine (± opioid)":             "#CA75A4",
    "Midazolam + propofol (± opioid)": "#EDBD51",
    VASO_LABEL:                        GRAY,
}


def color_of(name):
    return GROUP_COLORS.get(name) or GROUP_COLORS.get(POOL_OF.get(name), GRAY)


def dots(ax, x, y, lo, hi, names):
    """Point + CI per row, each in its group's color; vasopressor row as a gray square."""
    for xi, yi, l, h, g in zip(x, y, lo, hi, names):
        ax.errorbar([xi], [yi], xerr=[[l], [h]], fmt="s" if g == VASO_LABEL else "o",
                    color=color_of(g), capsize=3)

# Fonts: set before any figure is drawn. Only DejaVu Sans is named in the SVG (no fallback list).
import matplotlib
import matplotlib.ticker
matplotlib.rcParams["svg.fonttype"] = "none"   # keep text as text, not outlines
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
FIG_WIDTH_IN = 183 / 25.4   # double column (Nature journals: 183 mm); combined figure drawn at print size
SMALL_FONT = 9   # smallest text anywhere (legends, table, tick labels in boxplots); nothing below this
matplotlib.rcParams["axes.labelsize"] = 10    # axis labels 10 pt
matplotlib.rcParams["xtick.labelsize"] = 9     # tick numbers
matplotlib.rcParams["ytick.labelsize"] = 9
matplotlib.rcParams["legend.fontsize"] = SMALL_FONT


def save(fig, name, **kw):
    """Save PNG (quick look) and SVG with live text (editable in Illustrator)."""
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200, **kw)
    fig.savefig(OUT_DIR / f"{name}.svg", **kw)
    # Illustrator ignores CSS font-style on SVG import, so name the italic face directly
    # (PostScript name DejaVuSans-Oblique; needs DejaVuSans-Oblique.ttf installed).
    import re
    f = OUT_DIR / f"{name}.svg"
    txt = f.read_text(encoding="utf-8")
    txt = re.sub(r"font-style: oblique; (font-size: [\d.]+px); font-family: 'DejaVu Sans'",
                 r"\1; font-family: 'DejaVuSans-Oblique', 'DejaVu Sans'; font-style: oblique", txt)
    f.write_text(txt, encoding="utf-8")


# ============ MODEL FUNCTIONS ============
def fit(formula, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("lbfgs", "bfgs", "powell"):
            try:
                r = smf.mixedlm(formula, data=data, groups=data["_pid"]).fit(method=method, reml=True)
            except Exception:
                continue
            if np.isfinite(r.llf) and np.all(np.isfinite(r.fe_params.values)):
                return r.fe_params
    return None


def main_slopes(full, spec):
    """Pooled groups (main-text model), interaction model: within-group NESI-RASS slope."""
    out, m, safe = {}, spec["main"], spec["safe"]
    if not m["groups"]:
        return out
    dm = full[full["grp"].isin(m["other"] + [REFERENCE]) | full["pool"].isin(m["groups"])]
    terms = [spec["pool_safe"][g] for g in m["groups"]] + [safe[g] for g in m["other"]]
    p = fit("nesi ~ rass_c + " + " + ".join(f"{t} + {t}:rass_c" for t in terms), dm)
    if p is not None:
        b_r = p.get("rass_c", np.nan)
        out["main|ref_slope_inter"] = b_r
        for g in m["groups"] + m["other"]:
            t = spec["pool_safe"][g] if g in spec["pool_safe"] else safe[g]
            out[f"main:{g}|slope"] = b_r + p.get(f"{t}:rass_c", np.nan)
            out[f"main:{g}|slope_diff"] = p.get(f"{t}:rass_c", np.nan)
    return out


def estimates(data, spec):
    """All quantities for one dataset. spec: safe names, formulas, report levels, do_levels."""
    out, safe = {}, spec["safe"]
    full = data
    data = full[full["grp"].isin(spec["keep"])]       # A, B and the vasopressor control
    p = fit(spec["f_inter"], data)
    if p is not None:
        b_r = p.get("rass_c", np.nan)
        out["ref_slope"] = b_r
        for g, s in safe.items():
            b, bi = p.get(s, np.nan), p.get(f"{s}:rass_c", np.nan)
            out[f"{g}|slope"] = b_r + bi
            out[f"{g}|slope_diff"] = bi
            for r in spec["report"]:
                out[f"{g}|offset@{r}"] = b + bi * (r - RASS_CENTER)
    p = fit(spec["f_add"], data)
    if p is not None:
        b_r = p.get("rass_c", np.nan)
        for g, s in safe.items():
            out[f"{g}|add_offset"] = p.get(s, np.nan)
            out[f"{g}|rass_equiv"] = p.get(s, np.nan) / (-b_r) if b_r else np.nan

    ref = data[(data["grp"] == REFERENCE) & data["vaso"].notna()]
    if ref["vaso"].nunique() == 2:
        p = fit("nesi ~ rass_c + vaso", ref)
        if p is not None:
            b_r = p.get("rass_c", np.nan)
            out[f"{VASO_LABEL}|add_offset"] = p.get("vaso", np.nan)
            out[f"{VASO_LABEL}|rass_equiv"] = p.get("vaso", np.nan) / (-b_r) if b_r else np.nan
        p = fit("nesi ~ rass_c + vaso + vaso:rass_c", ref)
        if p is not None:
            b_r, b, bi = p.get("rass_c", np.nan), p.get("vaso", np.nan), p.get("vaso:rass_c", np.nan)
            out[f"{VASO_LABEL}|slope"] = b_r + bi
            out[f"{VASO_LABEL}|slope_diff"] = bi
            for r in spec["report"]:
                out[f"{VASO_LABEL}|offset@{r}"] = b + bi * (r - RASS_CENTER)

    # Main-text model: additive, pooled groups + the other strata as their own indicators
    m = spec["main"]
    if m["groups"]:
        dm = full[full["grp"].isin(m["other"] + [REFERENCE]) | full["pool"].isin(m["groups"])]
        p = fit("nesi ~ rass_c + " + " + ".join([spec["pool_safe"][g] for g in m["groups"]] +
                                                 [safe[g] for g in m["other"]]), dm)
        if p is not None:
            b_r = p.get("rass_c", np.nan)
            out["main|ref_slope"] = b_r
            for g in m["groups"] + m["other"]:
                s = spec["pool_safe"][g] if g in spec["pool_safe"] else safe[g]
                out[f"main:{g}|add_offset"] = p.get(s, np.nan)
                out[f"main:{g}|rass_equiv"] = p.get(s, np.nan) / (-b_r) if b_r else np.nan
        out.update(main_slopes(full, spec))

    if spec["do_levels"]:                       # analysis C: pooled groups at each RASS level
        pool_safe = spec["pool_safe"]
        dp = full[full["pool"].notna()]         # reference + pooled groups only
        for lv in LEVELS:
            dl = dp[dp["rass"] == lv]
            if not (len(dl[dl["pool"] == REFERENCE]) >= MIN_CELL_OBS
                    and dl.loc[dl["pool"] == REFERENCE, "_pid"].nunique() >= MIN_CELL_PAT):
                continue
            groups = [g for g in pool_safe
                      if len(dl[dl["pool"] == g]) >= MIN_CELL_OBS
                      and dl.loc[dl["pool"] == g, "_pid"].nunique() >= MIN_CELL_PAT]
            if groups:
                dl = dl[dl["pool"].isin(groups + [REFERENCE])]
                p = fit("nesi ~ " + " + ".join(pool_safe[g] for g in groups), dl)
                if p is not None:
                    for g in groups:
                        out[f"{g}|level{lv}"] = p.get(pool_safe[g], np.nan)
    return out


def boot_once(seed, data, pt_rows, spec):
    """One cluster-bootstrap resample: draw patients with replacement, relabel duplicates."""
    rng = np.random.default_rng(seed)
    keys = list(pt_rows.keys())
    pick = rng.integers(0, len(keys), len(keys))
    idx = np.concatenate([pt_rows[keys[k]] for k in pick])
    rs = data.iloc[idx].copy()
    rs["_pid"] = np.repeat(np.arange(len(pick)), [len(pt_rows[keys[k]]) for k in pick])
    return main_slopes(rs, spec) if SLOPES_ONLY else estimates(rs, spec)


# ============ ONE RASS RANGE ============
def run_range(d_all, label):
    lo, hi = RANGES[label]
    d = d_all[d_all["rass"].between(lo, hi)].copy()
    size = d.groupby("grp").agg(n_obs=("nesi", "size"), n_pat=("_pid", "nunique"))
    keep = size[(size.n_obs >= MIN_OBS) & (size.n_pat >= MIN_PATIENTS)].index.tolist()
    if REFERENCE not in keep:
        print(f"[{label}] reference group too small; skipped.")
        return
    dropped = [g for g in size.index if g not in keep]
    # Keep rows used by A/B (strata passing the rule) and by C (pooled groups, whose members
    # are pooled even when one member alone is below the stratum rule).
    d = d[d["grp"].isin(keep) | d["pool"].notna()].reset_index(drop=True)
    agents = [g for g in size.loc[keep].sort_values("n_obs", ascending=False).index if g != REFERENCE]
    safe = {g: f"a{i}" for i, g in enumerate(agents)}
    for g, s in safe.items():
        d[s] = (d["grp"] == g).astype(float)
    pool_safe = {g: f"p{i}" for i, g in enumerate(POOLED)}
    for g, s in pool_safe.items():
        d[s] = (d["pool"] == g).astype(float)
    d["rass_c"] = d["rass"] - RASS_CENTER
    # Main-text model: pooled groups that pass the inclusion rule as a group, plus every other
    # stratum that passes it on its own (modeled separately; only opioid only is displayed)
    psize = d[d["pool"].isin(POOLED)].groupby("pool").agg(n_obs=("nesi", "size"), n_pat=("_pid", "nunique"))
    main_groups = [g for g in POOLED if g in psize.index
                   and psize.loc[g, "n_obs"] >= MIN_OBS and psize.loc[g, "n_pat"] >= MIN_PATIENTS]
    main_other = [g for g in agents if g not in POOL_OF]
    spec_main = {"groups": main_groups, "other": main_other,
                 "size": {**{g: (int(psize.loc[g, "n_obs"]), int(psize.loc[g, "n_pat"])) for g in main_groups},
                          **{g: (int(size.loc[g, "n_obs"]), int(size.loc[g, "n_pat"])) for g in main_other}}}
    spec = {"safe": safe, "pool_safe": pool_safe, "keep": keep, "main": spec_main,
            "report": RASS_REPORT[label], "do_levels": label == "full",
            "f_inter": "nesi ~ rass_c + " + " + ".join(f"{s} + {s}:rass_c" for s in safe.values()),
            "f_add": "nesi ~ rass_c + " + " + ".join(safe.values())}

    print(f"\n=== RASS {lo} to {hi} ({label}) ===")
    d_ab = d[d["grp"].isin(keep)]
    print(f"Rows {len(d_ab):,}; patients {d_ab['_pid'].nunique():,}; agents {len(agents)} + {REF_LABEL}")
    if dropped:
        print(f"Strata below the inclusion rule in this range (not modeled): {dropped}")

    if SLOPES_ONLY:
        return run_slopes_only(d, spec, label)
    t0 = time.time()
    point = estimates(d, spec)
    print(f"Point estimates done ({time.time() - t0:.0f} s; bootstrap ~"
          f"{(time.time() - t0) * N_BOOTSTRAP / 60:.0f} core-minutes)")
    pt_rows = dict(d.groupby("_pid").indices)
    t0 = time.time()
    boots = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(boot_once)(RNG_SEED + i, d, pt_rows, spec) for i in range(N_BOOTSTRAP))
    print(f"Bootstrap done in {(time.time() - t0) / 60:.1f} min")
    B = pd.DataFrame(boots)
    B.to_csv(OUT_DIR / f"bootstrap_{STRATUM_COL}_{label}.csv", index=False)
    report(d_ab, size, dropped, agents, point, B, label, spec)


def run_slopes_only(d, spec, label):
    """Bootstrap only the pooled-group slopes; merge with the saved offsets; redraw the main figure.
    Same seed and resampling as the full run, so the resamples are the same patients."""
    fm = OUT_DIR / f"results_main_{STRATUM_COL}_{label}.csv"
    fr = OUT_DIR / f"results_{STRATUM_COL}_{label}.csv"
    if not (fm.exists() and fr.exists()):
        print(f"[{label}] SLOPES_ONLY needs the saved {fm.name} and {fr.name}; skipped.")
        return
    t0 = time.time()
    point = main_slopes(d, spec)
    print(f"Point estimates done ({time.time() - t0:.0f} s)")
    pt_rows = dict(d.groupby("_pid").indices)
    t0 = time.time()
    boots = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(boot_once)(RNG_SEED + i, d, pt_rows, spec) for i in range(N_BOOTSTRAP))
    print(f"Bootstrap done in {(time.time() - t0) / 60:.1f} min")
    B = pd.DataFrame(boots)
    B.to_csv(OUT_DIR / f"bootstrap_main_slopes_{STRATUM_COL}_{label}.csv", index=False)

    def ci(key):
        v = B[key].dropna() if key in B else pd.Series(dtype=float)
        return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)

    main = pd.read_csv(fm)
    main["agent"] = main["agent"].replace(OLD_VASO_LABELS, VASO_LABEL)
    res = pd.read_csv(fr)
    res["agent"] = res["agent"].replace(OLD_VASO_LABELS, VASO_LABEL)
    for col in ["slope", "slope_lo", "slope_hi", "slope_diff", "slope_diff_lo", "slope_diff_hi"]:
        main[col] = np.nan
    for i, g in main["agent"].items():
        if g == VASO_LABEL:   # vasopressor slope already estimated in the full run
            v = res[res["agent"] == VASO_LABEL]
            if len(v):
                main.loc[i, ["slope", "slope_lo", "slope_hi"]] = v[["slope", "slope_lo", "slope_hi"]].values[0]
            continue
        for k in ["slope", "slope_diff"]:
            main.loc[i, k] = point.get(f"main:{g}|{k}", np.nan)
            main.loc[i, f"{k}_lo"], main.loc[i, f"{k}_hi"] = ci(f"main:{g}|{k}")
    main.to_csv(OUT_DIR / f"results_main_slopes_{STRATUM_COL}_{label}.csv", index=False)
    mref = point.get("main|ref_slope_inter", np.nan)
    mrl, mrh = ci("main|ref_slope_inter")
    pd.DataFrame([{"ref_slope": mref, "ref_slope_lo": mrl, "ref_slope_hi": mrh}]).to_csv(
        OUT_DIR / f"ref_slope_main_{STRATUM_COL}_{label}.csv", index=False)

    fmt = lambda e, l, h, p=3: f"{e:+.{p}f} ({l:+.{p}f} to {h:+.{p}f})" if np.isfinite(e) else "n/a"
    lo, hi = RANGES[label]
    md = [f"# Pooled-group slopes, RASS {lo} to {hi} ({STRATUM_COL})\n",
          f"Interaction model, same pooled groups as the main-text offsets. Cluster bootstrap, "
          f"{N_BOOTSTRAP} resamples; 95% percentile CIs.\n",
          "| Group | NESI per RASS point | Difference from reference slope |", "|---|---|---|",
          f"| {REF_LABEL} | {fmt(mref, mrl, mrh)} | 0 |"]
    for _, r in main.iterrows():
        md.append(f"| {r['agent']} | {fmt(r['slope'], r['slope_lo'], r['slope_hi'])} | "
                  f"{fmt(r['slope_diff'], r['slope_diff_lo'], r['slope_diff_hi'])} |")
    md.append(f"\nBootstrap resamples with no successful fit: {int(B.isna().all(axis=1).sum())} of {N_BOOTSTRAP}.")
    (OUT_DIR / f"report_main_slopes_{STRATUM_COL}_{label}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))
    slopes_plot(main, mref, mrl, mrh, label)
    main_figure(main, mref, mrl, mrh, label)
    combined_figure(main, mref, mrl, mrh, label)


# ============ REPORT ============
def report(d, size, dropped, agents, point, B, label, spec):
    lo, hi = RANGES[label]
    rep = spec["report"]

    def ci(key):
        v = B[key].dropna() if key in B else pd.Series(dtype=float)
        return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)

    fmt = lambda e, l, h, p=2: f"{e:+.{p}f} ({l:+.{p}f} to {h:+.{p}f})" if np.isfinite(e) else "n/a"

    ref_v = d[(d["grp"] == REFERENCE) & (d["vaso"] == 1)]
    counts = {g: (int(size.loc[g, "n_obs"]), int(size.loc[g, "n_pat"])) for g in agents}
    counts[VASO_LABEL] = (len(ref_v), ref_v["_pid"].nunique())
    def has_data(x, lv):  # enough data at this RASS level to report an offset (not an extrapolation)
        x = x[x["rass"] == lv]
        return len(x) >= MIN_CELL_OBS and x["_pid"].nunique() >= MIN_CELL_PAT

    ref_d = d[d["grp"] == REFERENCE]
    rows = []
    for g in agents + [VASO_LABEL]:
        r = {"agent": g, "n_obs": counts[g][0], "n_patients": counts[g][1]}
        for k in [f"offset@{x}" for x in rep] + ["slope", "add_offset", "rass_equiv"]:
            r[k] = point.get(f"{g}|{k}", np.nan)
            r[f"{k}_lo"], r[f"{k}_hi"] = ci(f"{g}|{k}")
        gd = ref_v if g == VASO_LABEL else d[d["grp"] == g]
        cmp_d = ref_d[ref_d["vaso"] == 0] if g == VASO_LABEL else ref_d
        for x in rep:
            r[f"has_data@{x}"] = bool(has_data(gd, x) and has_data(cmp_d, x))
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / f"results_{STRATUM_COL}_{label}.csv", index=False)

    n_ref, p_ref = int(size.loc[REFERENCE, "n_obs"]), int(size.loc[REFERENCE, "n_pat"])
    rl, rh = ci("ref_slope")
    md = [f"# Agent-stratified analysis: NESI vs RASS, RASS {lo} to {hi} ({STRATUM_COL})\n",
          f"Rows {len(d):,}; patients {d['_pid'].nunique():,}. {REF_LABEL}: no sedative or opioid "
          f"infusion and no intermittent benzodiazepine dose (n = {n_ref:,} obs, {p_ref:,} patients). "
          f"Cluster bootstrap, {N_BOOTSTRAP} resamples; 95% percentile CIs.\n",
          f"Strata below the inclusion rule in this range: {', '.join(dropped) if dropped else 'none'}.\n",
          "Offsets are EEG-exam discrepancies at matched RASS (positive = NESI higher than the exam "
          "predicts), not drug effects. The vasopressor row is the negative control, fit within the "
          "reference group only.\n"]

    # Main-text model: pooled groups + opioid only + vasopressor control
    m = spec["main"]
    shown = m["groups"] + ([OPIOID_ONLY] if OPIOID_ONLY in m["other"] else [])
    main_rows = []
    for g in shown:
        r = {"agent": g, "n_obs": m["size"][g][0], "n_patients": m["size"][g][1]}
        for k in ["add_offset", "rass_equiv", "slope", "slope_diff"]:
            r[k] = point.get(f"main:{g}|{k}", np.nan)
            r[f"{k}_lo"], r[f"{k}_hi"] = ci(f"main:{g}|{k}")
        main_rows.append(r)
    vrow = res[res["agent"] == VASO_LABEL].iloc[0]
    vd = {k: vrow[k] for k in ["agent", "n_obs", "n_patients", "add_offset", "add_offset_lo",
                               "add_offset_hi", "rass_equiv", "rass_equiv_lo", "rass_equiv_hi",
                               "slope", "slope_lo", "slope_hi"]}
    vd["slope_diff"] = point.get(f"{VASO_LABEL}|slope_diff", np.nan)
    vd["slope_diff_lo"], vd["slope_diff_hi"] = ci(f"{VASO_LABEL}|slope_diff")
    main_rows.append(vd)
    main = pd.DataFrame(main_rows)
    mref = point.get("main|ref_slope_inter", np.nan)
    mrl, mrh = ci("main|ref_slope_inter")
    main.to_csv(OUT_DIR / f"results_main_{STRATUM_COL}_{label}.csv", index=False)
    hidden = [g for g in m["other"] if g != OPIOID_ONLY]
    md += ["## Main-text model: additive, pooled groups\n",
           "Each agent pooled with and without an opioid infusion (pooling fixed in advance). Strata "
           f"modeled separately but not shown here: {', '.join(hidden) if hidden else 'none'}.\n",
           "| Group | Obs | Patients | Offset (NESI units) | In RASS points |", "|---|---|---|---|---|",
           f"| {REF_LABEL} | {n_ref:,} | {p_ref:,} | 0 | 0 |"]
    for _, r in main.iterrows():
        md.append(f"| {r['agent']} | {r['n_obs']:,} | {r['n_patients']:,} | "
                  f"{fmt(r['add_offset'], r['add_offset_lo'], r['add_offset_hi'])} | "
                  f"{fmt(r['rass_equiv'], r['rass_equiv_lo'], r['rass_equiv_hi'], 1)} |")
    md += ["\n## Main-text model: NESI-RASS slope within each pooled group\n",
           "Same groups as above, interaction model.\n",
           "| Group | NESI per RASS point | Difference from reference slope |", "|---|---|---|",
           f"| {REF_LABEL} | {fmt(mref, mrl, mrh, 3)} | 0 |"]
    for _, r in main.iterrows():
        md.append(f"| {r['agent']} | {fmt(r['slope'], r['slope_lo'], r['slope_hi'], 3)} | "
                  f"{fmt(r['slope_diff'], r['slope_diff_lo'], r['slope_diff_hi'], 3)} |")
    md += ["\n## Supplement: additive model, every stratum separate\n",
          "| Agent | Obs | Patients | Offset (NESI units) | In RASS points |", "|---|---|---|---|---|",
          f"| {REF_LABEL} | {n_ref:,} | {p_ref:,} | 0 | 0 |"]
    for _, r in res.iterrows():
        md.append(f"| {r['agent']} | {r['n_obs']:,} | {r['n_patients']:,} | "
                  f"{fmt(r['add_offset'], r['add_offset_lo'], r['add_offset_hi'])} | "
                  f"{fmt(r['rass_equiv'], r['rass_equiv_lo'], r['rass_equiv_hi'], 1)} |")
    md += ["\n'In RASS points': offset divided by the reference RASS slope; positive means the EEG "
           "reads that many RASS points deeper than the exam.\n",
           "## Supplement: interaction model, NESI-RASS slope within each stratum\n",
           "Offsets by RASS level are in the per-level table below (estimated from each level's "
           "own data); the interaction model is used here only for slopes.\n",
           "| Agent | NESI per RASS point | Difference from reference slope |", "|---|---|---|",
           f"| {REF_LABEL} | {fmt(point.get('ref_slope', np.nan), rl, rh, 3)} | 0 |"]
    for _, r in res.iterrows():
        dk = f"{r['agent']}|slope_diff"
        dl, dh = ci(dk)
        md.append(f"| {r['agent']} | {fmt(r['slope'], r['slope_lo'], r['slope_hi'], 3)} | "
                  f"{fmt(point.get(dk, np.nan), dl, dh, 3)} |")
    md += [
           "\nSlope: NESI change per RASS point within each stratum (negative = NESI rises as RASS "
           "deepens). Vasopressor row: offset near zero -> illness severity alone does not create an "
           "EEG-exam gap; clearly nonzero -> confounding by illness likely affects agent offsets too.\n"]

    if spec["do_levels"]:
        lv_rows = []
        for g in POOLED:
            r = {"group": g}
            for lv in LEVELS:
                r[f"level{lv}"] = point.get(f"{g}|level{lv}", np.nan)
                r[f"level{lv}_lo"], r[f"level{lv}_hi"] = ci(f"{g}|level{lv}")
            lv_rows.append(r)
        lvd = pd.DataFrame(lv_rows)
        lvd.to_csv(OUT_DIR / f"results_by_level_{STRATUM_COL}.csv", index=False)
        level_plot(lvd)
        md += ["\n## Offset at each RASS level, pooled groups (no linearity assumption)\n",
               "Each agent pooled with and without an opioid infusion, vs the reference. Fit "
               f"separately within each level; '—' = fewer than {MIN_CELL_OBS} obs or "
               f"{MIN_CELL_PAT} patients in that cell.\n",
               "| Group | " + " | ".join(f"RASS {lv}" for lv in LEVELS) + " |",
               "|---|" + "---|" * len(LEVELS)]
        for _, r in lvd.iterrows():
            cells = [fmt(r[f"level{lv}"], r[f"level{lv}_lo"], r[f"level{lv}_hi"])
                     if np.isfinite(r[f"level{lv}"]) else "—" for lv in LEVELS]
            md.append(f"| {r['group']} | " + " | ".join(cells) + " |")

    md.append(f"\nBootstrap resamples with no successful fit: {int(B.isna().all(axis=1).sum())} of {N_BOOTSTRAP}.")
    (OUT_DIR / f"report_{STRATUM_COL}_{label}.md").write_text("\n".join(md), encoding="utf-8")
    forest(res, point, label)
    slopes_plot(main, mref, mrl, mrh, label)
    main_figure(main, mref, mrl, mrh, label)
    combined_figure(main, mref, mrl, mrh, label)
    pd.DataFrame([{"ref_slope": mref, "ref_slope_lo": mrl, "ref_slope_hi": mrh}]).to_csv(
        OUT_DIR / f"ref_slope_main_{STRATUM_COL}_{label}.csv", index=False)
    forest_main(main, label)
    print("\n" + "\n".join(md[4:]))


def level_plot(lvd):
    """Offset vs reference at each RASS level, pooled groups: one panel per group (shared axes)
    plus a combined panel. Points only where the cell has enough data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lvd = lvd[~lvd["group"].isin(LEVEL_PLOT_SKIP)].reset_index(drop=True)
        groups = list(lvd["group"])
        x = np.array(LEVELS)

        def series(r):
            y = np.array([r[f"level{lv}"] for lv in LEVELS], dtype=float)
            lo = np.array([r[f"level{lv}_lo"] for lv in LEVELS], dtype=float)
            hi = np.array([r[f"level{lv}_hi"] for lv in LEVELS], dtype=float)
            ok = np.isfinite(y)
            err = [np.clip(np.nan_to_num(y - lo), 0, None), np.clip(np.nan_to_num(hi - y), 0, None)]
            return y, err, ok

        # small multiples
        n = len(groups)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for i, (ax, (_, r)) in enumerate(zip(axes, lvd.iterrows())):
            y, err, ok = series(r)
            ax.axhline(0, color=GRAY, lw=1)
            ax.errorbar(x[ok], y[ok], yerr=[err[0][ok], err[1][ok]], marker="o", ms=4, capsize=2,
                        lw=1.4, color=color_of(r["group"]))
            ax.set_title(r["group"], loc="left", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[n:]:
            ax.set_visible(False)
        for ax in axes[:n]:
            ax.set_xticks(LEVELS)
        fig.supxlabel("RASS", fontsize=10)
        fig.supylabel("Offset vs reference (NESI units, 95% CI)", fontsize=10)
        fig.tight_layout()
        save(fig, f"offset_by_level_panels_{STRATUM_COL}")
        plt.close(fig)

        # combined, points dodged so CIs don't overlap
        fig, ax = plt.subplots(figsize=(8, 5.5))
        dodge = np.linspace(-0.25, 0.25, n) if n > 1 else [0]
        ax.axhline(0, color="black", lw=1.6, label="Reference (no sedative or opioid infusion)")
        for i, (_, r) in enumerate(lvd.iterrows()):
            y, err, ok = series(r)
            ax.errorbar(x[ok] + dodge[i], y[ok], yerr=[err[0][ok], err[1][ok]], marker="o", ms=4,
                        capsize=2, lw=1.2, color=color_of(r["group"]), label=r["group"])
        ax.set_xticks(LEVELS)
        ax.set_xlabel("RASS")
        ax.set_ylabel("Offset vs reference (NESI units, 95% CI)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=SMALL_FONT, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        save(fig, f"offset_by_level_{STRATUM_COL}")
        plt.close(fig)
    except Exception as e:
        print(f"Offset-by-level plot skipped: {e}")


def forest_main(main, label):
    """Main-text figure: one offset per pooled group, vasopressor control set apart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = main.iloc[::-1].reset_index(drop=True)
        is_v = (order["agent"] == VASO_LABEL).values
        y = np.arange(len(order))
        lo = np.clip((order["add_offset"] - order["add_offset_lo"]).fillna(0).values, 0, None)
        hi = np.clip((order["add_offset_hi"] - order["add_offset"]).fillna(0).values, 0, None)
        fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(order) + 1.8))
        dots(ax, order["add_offset"], y, lo, hi, order["agent"])
        ax.axvline(0, color=GRAY, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a[0].upper() + a[1:]}\n($\\it{{n}}$ = {p:,})"
                            for a, p in zip(order["agent"], order["n_patients"])])
        ax.set_xlabel("Offset vs reference (NESI units)")
        lo_r, hi_r = RANGES[label]
        ax.set_title(f"RASS {lo_r} to {hi_r}", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save(fig, f"forest_main_{STRATUM_COL}_{label}", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Main forest plot skipped: {e}")


def fig_name(g):
    """Figure label: '(± opioid)' dropped to keep labels short (stated once in the figure legend)."""
    return g.replace(" (± opioid)", "")


def draw_offsets_slopes(a1, a2, main, ref, ref_lo, ref_hi, letters):
    """Offsets (a1) and within-group slopes (a2) for the performance groups, shared y-axis."""
    keep = [g for g in SLOPE_FIGURE_GROUPS if g in set(main["agent"])] + \
           ([VASO_LABEL] if VASO_LABEL in set(main["agent"]) else [])
    order = main.set_index("agent").loc[keep].reset_index().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(order))

    def bars(ax, col):
        lo = np.clip((order[col] - order[f"{col}_lo"]).fillna(0).values, 0, None)
        hi = np.clip((order[f"{col}_hi"] - order[col]).fillna(0).values, 0, None)
        dots(ax, order[col], y, lo, hi, order["agent"])
        ax.spines[["top", "right"]].set_visible(False)

    bars(a1, "add_offset")
    a1.axvline(0, color=GRAY, lw=1)
    a1.set_xlabel("Offset vs reference (NESI units)")
    a1.set_title(letters[0], loc="left", fontsize=12, fontweight="bold")
    a1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))   # fewer ticks, no crowding
    if np.isfinite(ref_lo) and np.isfinite(ref_hi):
        a2.axvspan(ref_lo, ref_hi, color="#E4E2DA", lw=0)
    if np.isfinite(ref):
        a2.axvline(ref, color="black", lw=1.2)
    a2.axvline(0, color=GRAY, lw=1, ls=":")
    bars(a2, "slope")
    a2.set_xlabel("NESI per RASS point (slope)")
    a2.set_title(letters[1], loc="left", fontsize=12, fontweight="bold")
    a2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    a1.set_yticks(y)
    a1.set_yticklabels([f"{fig_name(a)}\n($\\it{{n}}$ = {p:,})" for a, p in zip(order["agent"], order["n_patients"])])
    a2.tick_params(labelleft=False)


def combined_figure(main, ref, ref_lo, ref_hi, label):
    """Main-text figure at print size (183 mm, double column): a, diagnostic plot with
    patients-at-level table; b, offsets; c, within-group slopes."""
    if not DIAG or label != "full":
        return
    try:
        import matplotlib.pyplot as plt
        d = DIAG
        fig = plt.figure(figsize=(FIG_WIDTH_IN, 9.4))
        # fixed margins (no tight bbox) so the saved figure is exactly 183 mm wide
        # two gridspecs: top (diagnostic + table) and bottom (b, c), each with its own left
        # margin, since the bottom row's group labels are wider than the table labels
        top = fig.add_gridspec(2, 1, height_ratios=[4.0, 0.3 * len(d["shown"]) + 0.4], hspace=0.06,
                               left=0.21, right=0.97, top=0.90, bottom=0.42)
        ax = fig.add_subplot(top[0])
        tab_ax = fig.add_subplot(top[1], sharex=ax)
        draw_diagnostic(ax, tab_ax, d["summ"], d["total_pat"], d["shown"])
        ax.text(-0.25, 1.02, "a", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
        bot = fig.add_gridspec(1, 2, wspace=0.10, left=0.31, right=0.97, top=0.34, bottom=0.06)
        a1 = fig.add_subplot(bot[0])
        a2 = fig.add_subplot(bot[1], sharey=a1)
        draw_offsets_slopes(a1, a2, main, ref, ref_lo, ref_hi, ("b", "c"))
        save(fig, f"figure_sedation_combined_{STRATUM_COL}")
        plt.close(fig)
    except Exception as e:
        print(f"Combined figure skipped: {e}")


def main_figure(main, ref, ref_lo, ref_hi, label):
    """Main-text figure: A, offset vs reference at matched RASS; B, within-group NESI-RASS slope.
    Same pooled groups and rows in both panels (shared y-axis)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_rows = len([g for g in SLOPE_FIGURE_GROUPS if g in set(main["agent"])]) + 1
        fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(10, 0.55 * n_rows + 2))
        draw_offsets_slopes(a1, a2, main, ref, ref_lo, ref_hi, ("a", "b"))
        lo_r, hi_r = RANGES[label]
        fig.suptitle(f"RASS {lo_r} to {hi_r}", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(w_pad=3)
        save(fig, f"main_offsets_slopes_{STRATUM_COL}_{label}", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Main figure skipped: {e}")


def figure_rows(res):
    """Main strata in a fixed order, then the vasopressor row."""
    keep = [g for g in FIGURE_STRATA if g in set(res["agent"])] + \
           ([VASO_LABEL] if VASO_LABEL in set(res["agent"]) else [])
    return res.set_index("agent").loc[keep].reset_index()


def slopes_plot(res, ref, ref_lo, ref_hi, label):
    """Within-group NESI-RASS slope for each pooled group (main-text model), with the reference slope and its CI as a
    band. All slopes away from 0 = NESI tracks RASS in every group; distance from the band =
    steeper or flatter than in unsedated patients."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keep = [g for g in SLOPE_FIGURE_GROUPS if g in set(res["agent"])] + \
               ([VASO_LABEL] if VASO_LABEL in set(res["agent"]) else [])
        order = res.set_index("agent").loc[keep].reset_index().iloc[::-1].reset_index(drop=True)
        y = np.arange(len(order))
        is_v = (order["agent"] == VASO_LABEL).values
        lo = np.clip((order["slope"] - order["slope_lo"]).fillna(0).values, 0, None)
        hi = np.clip((order["slope_hi"] - order["slope"]).fillna(0).values, 0, None)
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 2))
        if np.isfinite(ref_lo) and np.isfinite(ref_hi):
            ax.axvspan(ref_lo, ref_hi, color="#E4E2DA", lw=0, label="Reference slope (95% CI)")
        if np.isfinite(ref):
            ax.axvline(ref, color="black", lw=1.2)
        ax.axvline(0, color=GRAY, lw=1, ls=":")
        dots(ax, order["slope"], y, lo, hi, order["agent"])
        ax.set_yticks(y)
        ax.set_yticklabels(order["agent"])
        ax.set_xlabel("NESI–RASS slope (NESI per RASS point)")
        lo_r, hi_r = RANGES[label]
        ax.set_title(f"RASS {lo_r} to {hi_r}", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save(fig, f"slopes_{STRATUM_COL}_{label}", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Slopes plot skipped: {e}")


def _ref_slope_from_saved(label):
    """Reference slope and CI for PLOTS_ONLY: from ref_slope_*.csv, or parsed from the report."""
    f = OUT_DIR / f"ref_slope_main_{STRATUM_COL}_{label}.csv"
    if f.exists():
        r = pd.read_csv(f).iloc[0]
        return r["ref_slope"], r["ref_slope_lo"], r["ref_slope_hi"]
    rpt = OUT_DIR / f"report_{STRATUM_COL}_{label}.md"
    if rpt.exists():
        import re
        for line in rpt.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and cells[0] == REF_LABEL and len(cells) == 3:
                m = re.match(r"([-+]?\d*\.\d+) \(([-+]?\d*\.\d+) to ([-+]?\d*\.\d+)\)", cells[1])
                if m:
                    return tuple(float(x) for x in m.groups())
    return np.nan, np.nan, np.nan


def forest(res, point, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def err(o, k):  # clip at 0: with few resamples the estimate can fall outside its CI
            return [np.clip((o[k] - o[f"{k}_lo"]).fillna(0).values, 0, None),
                    np.clip((o[f"{k}_hi"] - o[k]).fillna(0).values, 0, None)]

        order = figure_rows(res).iloc[::-1].reset_index(drop=True)
        y = np.arange(len(order))
        is_v = (order["agent"] == VASO_LABEL).values
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 2))
        dots(ax, order["add_offset"], y, *err(order, "add_offset"), order["agent"])
        ax.axvline(0, color=GRAY, lw=1)
        ax.set_xlabel("Offset vs reference (NESI units)")
        ax.set_yticks(y)
        ax.set_yticklabels(order["agent"])
        lo, hi = RANGES[label]
        ax.set_title(f"RASS {lo} to {hi}", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save(fig, f"forest_{STRATUM_COL}_{label}")
        plt.close(fig)
    except Exception as e:
        print(f"Forest plot skipped: {e}")


# ============ DESCRIPTIVE FIGURES ============
DIAG = {}   # diagnostic-plot data, filled by descriptive()


def draw_diagnostic(ax, tab_ax, summ, total_pat, shown, title=None):
    """Mean NESI by RASS level per group (top) and patients-at-level table (bottom)."""
    colors = {g: color_of(g) for g in shown}
    TABLE_FONT = SMALL_FONT
    for g in shown:
        s = summ[(summ["grp"] == g) & (summ["count"] >= MIN_CELL_PAT)].set_index("rass")
        # every RASS level, NaN where a cell is too small: the line breaks there instead of
        # joining across missing levels
        s = s.reindex(sorted(LEVELS)).reset_index()
        name = REF_LABEL if g == REFERENCE else g
        ax.errorbar(s["rass"], s["mean"], yerr=1.96 * s["se"], marker="o", ms=4, capsize=2,
                    lw=2.2 if g == REFERENCE else 1.2, color=colors[g],
                    label=f"{'Reference' if g == REFERENCE else fig_name(name)} ($\\it{{n}}$ = {int(total_pat.get(g, 0)):,})")
    ax.set_xlim(min(LEVELS) - 0.5, max(LEVELS) + 0.5)
    ax.set_xticks(sorted(LEVELS))
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Mean NESI (patient means, 95% CI)")
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    # key above the plot, 3 columns, so it never covers a line; n = patients in each group
    ax.legend(fontsize=SMALL_FONT, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01),
              borderaxespad=0, columnspacing=1.2, handlelength=1.6)

    # number-of-patients table
    tab_ax.set_ylim(len(shown) - 0.5, -1.3)
    for r, g in enumerate(shown):
        short = "Reference" if g == REFERENCE else g.split(" (")[0]
        tab_ax.text(min(LEVELS) - 0.62, r, short, ha="right", va="center", fontsize=TABLE_FONT, color=colors[g])
        for lv in LEVELS:
            cell = summ[(summ["grp"] == g) & (summ["rass"] == lv)]
            n = int(cell["count"].iloc[0]) if len(cell) else 0
            tab_ax.text(lv, r, f"{n:,}", ha="center", va="center", fontsize=TABLE_FONT,
                        color=colors[g] if n >= MIN_CELL_PAT else "#B0B4BA")
    tab_ax.text(min(LEVELS) - 0.5, -1.05, "Patients at each RASS level", fontsize=TABLE_FONT,
                ha="left", va="center", color="#555")
    tab_ax.set_yticks([])
    tab_ax.spines[["top", "right", "left"]].set_visible(False)
    tab_ax.set_xlabel("RASS")


def descriptive(d):
    """Diagnostic plot and deep-RASS NESI distribution, from patient-level means."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Descriptive plots skipped: {e}")
        return
    # Reference + pooled groups only; "grp" below is the pooled group
    pm = (d[d["pool"].notna()].groupby(["pool", "rass", "_pid"])["nesi"].mean()
          .reset_index().rename(columns={"pool": "grp"}))         # one value per patient per level
    strata = [REFERENCE] + [g for g in POOLED if g not in LEVEL_PLOT_SKIP]

    # 1. Diagnostic plot: mean NESI by RASS level (mean of patient means +/- 95% CI)
    summ = (pm.groupby(["grp", "rass"])["nesi"].agg(["mean", "std", "count"]).reset_index())
    summ["se"] = summ["std"] / np.sqrt(summ["count"])
    summ.to_csv(OUT_DIR / f"diagnostic_nesi_by_rass_{STRATUM_COL}.csv", index=False)
    # Layout: plot on top, number-of-patients table underneath (one row per group, one column
    # per RASS level), like a number-at-risk table. RASS runs -5 to 0 left to right.
    shown = [g for g in strata if ((summ["grp"] == g) & (summ["count"] >= MIN_CELL_PAT)).any()]
    colors = {g: color_of(g) for g in strata}
    total_pat = pm.groupby("grp")["_pid"].nunique()
    DIAG.update(summ=summ, total_pat=total_pat, shown=shown)   # reused by the combined figure
    fig = plt.figure(figsize=(7, 6.2 + 0.36 * len(shown)))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.5, 0.42 * len(shown) + 0.5], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    tab_ax = fig.add_subplot(gs[1], sharex=ax)
    draw_diagnostic(ax, tab_ax, summ, total_pat, shown)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.08)
    save(fig, f"diagnostic_nesi_by_rass_{STRATUM_COL}", bbox_inches="tight")
    plt.close(fig)

    # 2. NESI distribution at RASS -4 and -5 by group
    deep = pm[pm["rass"].isin([-4, -5])]
    # values behind the boxplots (one NESI per patient per level; no IDs) for redrawing
    deep[["rass", "grp", "nesi"]].to_csv(OUT_DIR / f"deep_rass_nesi_values_{STRATUM_COL}.csv", index=False)
    tab = (deep.groupby(["rass", "grp"])["nesi"]
           .agg(patients="count", median="median",
                q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75)).round(2).reset_index())
    tab.to_csv(OUT_DIR / f"deep_rass_nesi_distribution_{STRATUM_COL}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, lv in zip(axes, [-5, -4]):   # -5 first, matching the other figures
        groups = [g for g in strata if ((deep["rass"] == lv) & (deep["grp"] == g)).sum() >= MIN_CELL_PAT]
        data = [deep[(deep["rass"] == lv) & (deep["grp"] == g)]["nesi"].values for g in groups]
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": "black", "lw": 1.5})
        for patch, g in zip(bp["boxes"], groups):   # group color, light fill
            patch.set_facecolor(matplotlib.colors.to_rgba(color_of(g), 0.35))
            patch.set_edgecolor(color_of(g))
        labels = [f"{'Reference' if g == REFERENCE else g}\n($\\it{{n}}$ = {len(v)})" for g, v in zip(groups, data)]
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=SMALL_FONT)
        ax.set_title(f"RASS {lv}", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("NESI (one value per patient)")
    fig.tight_layout()
    save(fig, f"deep_rass_nesi_distribution_{STRATUM_COL}")
    plt.close(fig)
    print("\nNESI at RASS -4/-5 by group (patient means):")
    print(tab.to_string(index=False))


def cap_rows(d, k, seed):
    """At most k rows per patient x stratum, preserving each patient's RASS distribution."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, x in d.groupby(["_pid", "grp"], sort=False):
        if len(x) <= k:
            keep.append(x.index.values)
            continue
        levels = x.groupby("rass").indices                    # rass level -> row positions
        counts = pd.Series({lv: len(ix) for lv, ix in levels.items()})
        target = counts * k / counts.sum()
        alloc = np.floor(target).astype(int)
        rem = k - alloc.sum()                                  # largest remainders get the extra
        alloc[(target - alloc).sort_values(ascending=False).index[:rem]] += 1
        for lv, ix in levels.items():
            if alloc[lv]:
                keep.append(x.index.values[rng.choice(ix, alloc[lv], replace=False)])
    return d.loc[np.sort(np.concatenate(keep))]


def concentration(d):
    """Counts only: how much each stratum depends on its heaviest patients."""
    rows = []
    for g, x in d.groupby("grp"):
        n = x.groupby("_pid").size().sort_values(ascending=False)
        rows.append({"stratum": g, "patients": len(n), "obs": int(n.sum()),
                     "obs/patient median": int(n.median()), "90th pct": int(n.quantile(.9)),
                     "max": int(n.max()), "% obs from top 5 patients": round(100 * n.head(5).sum() / n.sum(), 1)})
    t = pd.DataFrame(rows).sort_values("obs", ascending=False)
    t.to_csv(OUT_DIR / f"obs_per_patient_{STRATUM_COL}.csv", index=False)
    print("\nObservations per patient by stratum:")
    print(t.to_string(index=False))


# ============ MAIN ============
def main():
    print("INPUT")
    print(f"  {IN_ROWS.resolve()}")
    print("OUTPUT")
    print(f"  {OUT_DIR.resolve()}\n")
    if not IN_ROWS.exists():
        sys.exit(f"File not found: {IN_ROWS.resolve()}\nRun build_strata.py first.")

    d = pd.read_csv(IN_ROWS, low_memory=False).rename(columns={STRATUM_COL: "grp"})
    d = d[~d["grp"].isin(EXCLUDE)].dropna(subset=["nesi", "rass", "grp"]).copy()
    d = d[d["rass"].between(-5, 0)]
    d["vaso"] = pd.to_numeric(d["vaso"], errors="coerce")
    d["_pid"] = d["_pid"].astype(str)
    if DROP_OVERLAP_PATIENTS:
        if "dup_patient" not in d:
            sys.exit("No dup_patient column: rerun build_strata.py first.")
        n0, p0 = len(d), d["_pid"].nunique()
        d = d[d["dup_patient"] == 0].copy()
        print(f"Overlap sensitivity: dropped {n0 - len(d):,} rows, {p0 - d['_pid'].nunique():,} patients")
    concentration(d)
    if CAP_PER_PATIENT:
        n0 = len(d)
        if CAP_PER_PATIENT == "p95":
            ref_n = d[d["grp"] == REFERENCE].groupby("_pid").size()
            k = int(round(np.percentile(ref_n, 95)))
            print(f"Cap K = {k} (95th percentile of obs per patient in the reference group, "
                  f"{np.percentile(ref_n, 95):.1f}, rounded)")
        else:
            k = int(CAP_PER_PATIENT)
        before = d.groupby("grp").size()
        d = cap_rows(d, k, RNG_SEED)
        dropped = (100 * (1 - d.groupby("grp").size() / before)).round(1)
        print("Rows dropped by the cap, % by stratum:")
        print(dropped.sort_values(ascending=False).to_string())
        print(f"Cap sensitivity: at most {k} obs per patient per stratum, "
              f"RASS-stratified random selection; "
              f"kept {len(d):,} of {n0:,} rows")
    d["pool"] = d["grp"].map(POOL_OF)
    d.loc[d["grp"] == REFERENCE, "pool"] = REFERENCE
    print(f"Stratum column: {STRATUM_COL}")
    missing = [s for s in POOL_OF if s not in set(d["grp"])] + \
              ([REFERENCE] if REFERENCE not in set(d["grp"]) else [])
    if missing:
        print(f"WARNING: stratum names not found in the data (check spelling): {missing}")
    print("Pooled groups (analysis C and figures): " +
          "; ".join(f"{p} = {' + '.join(m)}" for p, m in POOLED.items()))

    descriptive(d)
    if PLOTS_ONLY:
        for label in RANGES:   # forest plots from saved results, no bootstrap
            f = OUT_DIR / f"results_{STRATUM_COL}_{label}.csv"
            if f.exists():
                saved = pd.read_csv(f)
                saved["agent"] = saved["agent"].replace(OLD_VASO_LABELS, VASO_LABEL)
                forest(saved, {}, label)
            fm = OUT_DIR / f"results_main_slopes_{STRATUM_COL}_{label}.csv"
            if not fm.exists():
                fm = OUT_DIR / f"results_main_{STRATUM_COL}_{label}.csv"
            if fm.exists() and "slope" in pd.read_csv(fm, nrows=0).columns:
                saved = pd.read_csv(fm)
                saved["agent"] = saved["agent"].replace(OLD_VASO_LABELS, VASO_LABEL)
                slopes_plot(saved, *_ref_slope_from_saved(label), label)
                main_figure(saved, *_ref_slope_from_saved(label), label)
                combined_figure(saved, *_ref_slope_from_saved(label), label)
            else:
                print(f"[{label}] pooled-group slopes not found; run once with SLOPES_ONLY = True.")
        print(f"\nPLOTS_ONLY: descriptive and forest plots redrawn in {OUT_DIR.resolve()}; models not run.")
        return
    for label in RANGES:
        run_range(d, label)

    print(f"\nSaved to {OUT_DIR.resolve()}")
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_OK)
    except ImportError:
        print("\a")


if __name__ == "__main__":
    main()