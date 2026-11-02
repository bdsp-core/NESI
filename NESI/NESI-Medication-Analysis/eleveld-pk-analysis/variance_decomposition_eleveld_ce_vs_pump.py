# variance_decomposition_eleveld_ce_vs_pump.py
"""
Variance decomposition of NESI and RASS by propofol exposure, with two
exposure measures run through identical machinery:
  - Ce:   Eleveld-predicted effect-site concentration (µg/mL)
  - Rate: pump infusion rate at the RASS time, weight-normalized (mg/kg/h).
          No PK model; answers the reviewer concern that Eleveld may be
          inaccurate in ICU patients.

Mixed-effects model: outcome ~ exposure + (1 | patient)

Cohort (identical for both exposures and both outcomes):
  propofol_only=True, PropofolSimStatus='simulated', Ce, pump rate, weight,
  NESI and RASS in [-5, 0] all available, >=5 obs per patient.

Paired cluster bootstrap: each iteration resamples patients ONCE and fits all
four models (NESI/RASS x Ce/Rate) on that same resample, so the NESI - RASS
difference has a valid CI for each exposure.

Why the NESI vs RASS comparison is robust to exposure error: nondifferential
error in the exposure attenuates variance explained for both outcomes by a
similar factor, because both are fit against the same exposure on the same
rows. Rate is a cruder exposure than Ce (ignores dosing history, accumulation,
washout), so absolute % values are expected to be smaller; the comparison is
what matters.

Methods sentence (suggested):
  "We repeated the variance decomposition using weight-normalized pump
  infusion rate (mg/kg/h) in place of Eleveld-predicted Ce, which requires no
  pharmacokinetic model. Both exposures were analyzed on the same
  observations, with NESI and RASS models fit on the same cluster-bootstrap
  resamples (1000 iterations)."

NOTE: RASS_CSV and TS_PARQUET are the _v2 outputs of eleveld_run_cohort.py,
which uses the corrected Eleveld implementation (theta10 = -0.0156) and
writes weight_kg_used. Run that script first.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import duckdb

# Repo layout: med_config.py lives in ../medication-exposure (no effect if already importable)
import sys as _sys
from pathlib import Path as _Path
_MC = _Path(__file__).resolve().parent.parent / "medication-exposure"
if _MC.is_dir() and str(_MC) not in _sys.path:
    _sys.path.insert(0, str(_MC))
from med_config import OUTPUT_CSV, OUTPUT_PARQUET, I0001_ASM_ANTIPSYCH_DIR


# ============ SENSITIVITY SWITCH ============
# False = primary analysis (unchanged).
# True  = combined "no intermittent sedative or opioid doses" sensitivity analysis:
#         drop observations with an intermittent benzodiazepine dose on board without a
#         midazolam infusion, or an intermittent opioid dose on board without an opioid
#         infusion (fast-acting route in prior 6 h, slow-acting in prior 12 h).
#         Flags come from the exposure windows file, joined on patient + RASS time.
#         The report gets a _no_intermittent suffix, so the primary report is not overwritten.
EXCLUDE_INTERMITTENT = False
import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=True)
_ap.add_argument("--exclude-intermittent", action="store_true")
_ap.add_argument("--within-between", action="store_true")
_ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples (default 1000)")
ARGS, _ = _ap.parse_known_args()
EXCLUDE_INTERMITTENT = EXCLUDE_INTERMITTENT or ARGS.exclude_intermittent
WINDOWS_CSV = os.path.join(I0001_ASM_ANTIPSYCH_DIR, 'i0001_per_eeg_master_exposure_windows_opiates.csv')

# ============ WITHIN-BETWEEN SWITCH ============
# True = within-between decomposition of Ce instead of the variance decomposition:
#   outcome ~ Ce_within + Ce_between + (1 | patient)
#   Ce_between = patient's mean Ce over their cohort rows; Ce_within = Ce - that mean.
#   Within slope: NESI change when the same patient's Ce changes.
#   Between slope: difference between patients with different average Ce.
#   Same cohort rows and same bootstrap resamples (same seed) as the variance decomposition.
#   Writes its own report (_within_between suffix); nothing overwritten. Works with
#   EXCLUDE_INTERMITTENT too.
WITHIN_BETWEEN = False or ARGS.within_between

# ============ PATHS ============
RASS_CSV      = os.path.join(OUTPUT_CSV,     'wt_ht_per_hosp_FULL_with_eleveld_expanded.csv')
TS_PARQUET    = os.path.join(OUTPUT_PARQUET, 'eleveld_timeseries_expanded.parquet')
OUT_REPORT_MD = os.path.join(OUTPUT_PARQUET, 'variance_decomposition_eleveld_ce_vs_pump_v3'
                             + ('_no_intermittent' if EXCLUDE_INTERMITTENT else '')
                             + ('_within_between' if WITHIN_BETWEEN else '') + '.md')


# ============ CONFIG ============
N_BOOTSTRAP         = ARGS.n_boot or 1000   # smoke test: --n-boot 2
RNG_SEED            = 42
MIN_OBS_PER_PATIENT = 5
RASS_COL            = 'R PHS IP RASS'
WEIGHT_COL          = 'weight_kg_used'   # same weight used for Eleveld (written by eleveld_run_cohort.py)
RATE_CAP            = 100           # mg/min; caps bolus-encoding spikes (pump display rate)

# label -> column. Both are fit on the same rows.
EXPOSURES = {
    'Ce':   'Ce_ug_per_mL',
    'Rate': 'pump_rate_mg_kg_h',
}
EXPOSURE_DESC = {
    'Ce':   'Eleveld Ce (µg/mL)',
    'Rate': 'pump rate (mg/kg/h), no PK model',
}


# ============ MARKDOWN BUILDER ============
md_lines = []
def md(s=''):
    md_lines.append(s)


# ============ CORE FIT ============
def fit_and_decompose(df, outcome_col, predictor_col):
    """Fit outcome ~ predictor + (1 | patient). Returns dict or None."""
    d = df[['BDSPPatientID', predictor_col, outcome_col]].dropna()
    d = d.rename(columns={outcome_col: 'outcome', predictor_col: 'pred'})
    if len(d) < 10 or d['BDSPPatientID'].nunique() < 3:
        return None
    if d['pred'].std() == 0 or d['outcome'].std() == 0:
        return None

    result = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for method in ('lbfgs', 'bfgs', 'powell'):   # lbfgs first; fall back if it fails or degenerates
            try:
                r = smf.mixedlm('outcome ~ pred', data=d, groups=d['BDSPPatientID']).fit(method=method, reml=True)
            except Exception:
                continue
            if np.isfinite(r.llf) and np.all(np.isfinite(r.fe_params.values)):
                result = r
                break
    if result is None:
        return None

    fe_slope     = result.fe_params['pred']
    random_var   = float(result.cov_re.iloc[0, 0])
    residual_var = float(result.scale)
    fe_var       = (fe_slope ** 2) * d['pred'].var()
    total_var    = fe_var + random_var + residual_var
    if total_var <= 0:
        return None

    return {
        'n':              len(d),
        'n_patients':     d['BDSPPatientID'].nunique(),
        'fe_slope':       fe_slope,
        'fe_slope_se':    result.bse_fe['pred'],
        'fe_slope_p':     result.pvalues['pred'],
        'pct_predictor':  100 * fe_var / total_var,
        'pct_patient':    100 * random_var / total_var,
        'pct_residual':   100 * residual_var / total_var,
        'marginal_r2':    fe_var / total_var,
        'conditional_r2': (fe_var + random_var) / total_var,
        'icc':            random_var / (random_var + residual_var),
    }


def fit_within_between(df, outcome_col):
    """outcome ~ ce_w + ce_b + (1 | patient). ce_w/ce_b must already be on df. Returns dict or None."""
    d = df[['BDSPPatientID', 'ce_w', 'ce_b', outcome_col]].dropna().rename(columns={outcome_col: 'outcome'})
    if len(d) < 10 or d['BDSPPatientID'].nunique() < 3:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for method in ('lbfgs', 'bfgs', 'powell'):
            try:
                r = smf.mixedlm('outcome ~ ce_w + ce_b', data=d, groups=d['BDSPPatientID']).fit(method=method, reml=True)
            except Exception:
                continue
            if np.isfinite(r.llf) and np.all(np.isfinite(r.fe_params.values)):
                w, b = r.fe_params['ce_w'], r.fe_params['ce_b']
                return {'within': w, 'between': b, 'contextual': b - w}
    return None


def run_within_between(cohort, n_rows, n_pts, n_excl):
    """Within-between decomposition of Ce, NESI and RASS, paired cluster bootstrap."""
    c = cohort.copy()
    c['ce_b'] = c.groupby('BDSPPatientID')['Ce_ug_per_mL'].transform('mean')
    c['ce_w'] = c['Ce_ug_per_mL'] - c['ce_b']
    sd_w, sd_b = c['ce_w'].std(), c.groupby('BDSPPatientID')['ce_b'].first().std()
    outs = {'NESI': 'PredictedNESI', 'RASS': '_RASS_numeric'}
    point = {o: fit_within_between(c, col) for o, col in outs.items()}
    if any(v is None for v in point.values()):
        raise RuntimeError('Within-between point estimate failed')
    for o, v in point.items():
        print(f"  point {o}: within {v['within']:+.3f}, between {v['between']:+.3f}")

    keys = ('within', 'between', 'contextual')
    boot = {f'{o}_{k}': [] for o in outs for k in keys}
    rng = np.random.default_rng(RNG_SEED)            # same resamples as the variance decomposition
    pts = c['BDSPPatientID'].unique()
    by_pt = {p: g for p, g in c.groupby('BDSPPatientID')}
    n_ok = n_fail = 0
    t0 = time.time()
    print(f"\nPaired bootstrap ({N_BOOTSTRAP} iterations, 2 fits each)...")
    for i in range(N_BOOTSTRAP):
        sample_pts = rng.choice(pts, size=len(pts), replace=True)
        parts = []
        for j, p in enumerate(sample_pts):
            block = by_pt[p].copy()
            block['BDSPPatientID'] = f'b{i}_{j}'
            parts.append(block)
        rs = pd.concat(parts, ignore_index=True)
        fits = {o: fit_within_between(rs, col) for o, col in outs.items()}
        if any(v is None for v in fits.values()):
            n_fail += 1
            continue
        for o, v in fits.items():
            for k in keys:
                boot[f'{o}_{k}'].append(v[k])
        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{N_BOOTSTRAP}  ok={n_ok} fail={n_fail}  ({time.time()-t0:.0f}s)")

    md('# Within-between decomposition of propofol Ce\n')
    md(f'_Generated from `{os.path.basename(RASS_CSV)}`._\n')
    md('## Cohort\n')
    md(f'- Same rows as the variance decomposition: **{n_rows:,} rows**, **{n_pts:,} patients**.')
    if EXCLUDE_INTERMITTENT:
        md(f'- **Sensitivity analysis:** {n_excl:,} observations with an intermittent benzodiazepine '
           'or opioid dose on board were excluded before the ≥5 obs/patient filter.')
    md(f'- SD of Ce within patients: {sd_w:.2f} µg/mL; SD of patient mean Ce across patients: {sd_b:.2f} µg/mL.\n')
    md('## Model\n')
    md('outcome ~ Ce_within + Ce_between + (1 | patient). Ce_between = patient mean Ce; '
       'Ce_within = Ce − patient mean. Cluster bootstrap over patients, '
       f'{n_ok}/{N_BOOTSTRAP} successful resamples ({n_fail} failed); 95% percentile CIs.\n')
    md('| Outcome | Within-patient slope (per µg/mL) | Between-patient slope (per µg/mL) | Between − within |')
    md('|---|---|---|---|')
    for o in outs:
        v = point[o]
        md(f"| {o} | {fmt_with_ci(v['within'], boot[f'{o}_within'])} | "
           f"{fmt_with_ci(v['between'], boot[f'{o}_between'])} | "
           f"{fmt_with_ci(v['contextual'], boot[f'{o}_contextual'], '{:+.3f}')} |")
    md('\n## Reading it\n')
    md('- **Within-patient slope:** change in the outcome when the same patient\'s Ce changes by 1 µg/mL.')
    md('- **Between-patient slope:** difference between patients whose average Ce differs by 1 µg/mL; '
       'mixes drug effect with who gets more propofol.')
    md('- **Between − within:** CI crossing 0 means the pooled Ce slope is not driven by '
       'between-patient differences.')

    print(f"\nWriting markdown report → {OUT_REPORT_MD} ...")
    tmp = OUT_REPORT_MD + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    os.replace(tmp, OUT_REPORT_MD)
    print('\n'.join(md_lines))


def ci(arr, lo=2.5, hi=97.5):
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (np.percentile(arr, lo), np.percentile(arr, hi))


def fmt_with_ci(point, samples, fmt='{:.3f}'):
    lo, hi = ci(samples)
    return f"{fmt.format(point)} ({fmt.format(lo)} – {fmt.format(hi)})"


def main():
    # ============ LOAD ============
    print(f"Mode: {'SENSITIVITY (no intermittent doses)' if EXCLUDE_INTERMITTENT else 'primary'}"
          f"{' — WITHIN-BETWEEN decomposition of Ce' if WITHIN_BETWEEN else ''}")
    print(f"Report → {OUT_REPORT_MD}")
    print(f"Loading {RASS_CSV} ...")
    rass = pd.read_csv(RASS_CSV, parse_dates=[
        'HospitalAdmitDTS', 'HospitalDischargeDTS', 'RASSRecordedDTS'])
    print(f"  {len(rass):,} rows")
    if WEIGHT_COL not in rass.columns:
        cands = [c for c in rass.columns if 'wt' in c.lower() or 'weight' in c.lower()]
        raise KeyError(f"WEIGHT_COL '{WEIGHT_COL}' not in {os.path.basename(RASS_CSV)}. "
                       f"Candidates: {cands}")

    cohort = rass[
        (rass['propofol_only_recomputed'] == True) &
        (rass['PropofolSimStatus'] == 'simulated') &
        rass['Ce_ug_per_mL'].notna()
    ].copy()
    cohort['_RASS_numeric'] = pd.to_numeric(cohort[RASS_COL], errors='coerce')
    cohort['PredictedNESI'] = pd.to_numeric(cohort['PredictedNESI'], errors='coerce')
    cohort[WEIGHT_COL]      = pd.to_numeric(cohort[WEIGHT_COL], errors='coerce')

    # ============ ATTACH PUMP RATE (vectorized, last rate change at/before RASS) ============
    print(f"Loading {TS_PARQUET} ...")
    ts = duckdb.query(f"""
        SELECT BDSPPatientID, HospitalAdmitDTS, t_min_from_admit AS t_grid,
               infusion_rate_mg_per_min
        FROM read_parquet('{TS_PARQUET.replace(os.sep, '/')}')
    """).df()
    ts['HospitalAdmitDTS'] = pd.to_datetime(ts['HospitalAdmitDTS'])
    print(f"  {len(ts):,} time-series rows")
    t_last = (ts.groupby(['BDSPPatientID', 'HospitalAdmitDTS'])['t_grid']
                .max().rename('t_grid_last').reset_index())

    cohort['t_min_from_admit'] = (
        (cohort['RASSRecordedDTS'] - cohort['HospitalAdmitDTS']).dt.total_seconds() / 60.0)
    cohort['_row'] = np.arange(len(cohort))
    left = cohort[['_row', 'BDSPPatientID', 'HospitalAdmitDTS', 't_min_from_admit']] \
        .dropna(subset=['t_min_from_admit']).sort_values('t_min_from_admit')
    right = ts.sort_values('t_grid')
    # Piecewise-constant rate: take the last rate change at or before the RASS time.
    joined = pd.merge_asof(left, right, left_on='t_min_from_admit', right_on='t_grid',
                           by=['BDSPPatientID', 'HospitalAdmitDTS'], direction='backward')
    joined = joined.merge(t_last, on=['BDSPPatientID', 'HospitalAdmitDTS'], how='left')
    out_of_range = joined['t_min_from_admit'] > joined['t_grid_last']
    joined.loc[out_of_range, 'infusion_rate_mg_per_min'] = np.nan
    del ts, right

    cohort = cohort.merge(joined[['_row', 'infusion_rate_mg_per_min']], on='_row', how='left')
    cohort['pump_rate_mg_per_min'] = cohort['infusion_rate_mg_per_min'].clip(upper=RATE_CAP)
    cohort['pump_rate_mg_kg_h'] = cohort['pump_rate_mg_per_min'] * 60.0 / cohort[WEIGHT_COL]
    print(f"  {cohort['pump_rate_mg_per_min'].notna().sum():,} of {len(cohort):,} rows have pump rate")

    # ============ INTERSECTION COHORT (same rows for every model) ============
    n_pre = len(cohort)
    cohort = cohort[
        cohort['PredictedNESI'].notna() &
        cohort['_RASS_numeric'].between(-5, 0) &
        cohort['Ce_ug_per_mL'].notna() &
        cohort['pump_rate_mg_kg_h'].notna() &
        (cohort[WEIGHT_COL] > 0)
    ].copy()
    n_excluded_intermittent = 0
    if EXCLUDE_INTERMITTENT:
        print(f"Loading {WINDOWS_CSV} (intermittent-dose flags) ...")
        flags = ['benzo_on_board', 'midazolam_inf_now', 'opioid_on_board', 'opioid_inf_now']
        w = pd.read_csv(WINDOWS_CSV, usecols=['BDSPPatientID', 'RASSRecordedDTS'] + flags, low_memory=False)
        norm = lambda x: x.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        w['_pid'] = norm(w['BDSPPatientID'])
        w['_t'] = pd.to_datetime(w['RASSRecordedDTS'], errors='coerce')
        w = w.dropna(subset=['_t']).drop_duplicates(['_pid', '_t'])
        cohort['_pid'] = norm(cohort['BDSPPatientID'])
        cohort['_t'] = pd.to_datetime(cohort['RASSRecordedDTS'], errors='coerce')
        cohort = cohort.merge(w[['_pid', '_t'] + flags], on=['_pid', '_t'], how='left')
        f = cohort[flags].apply(pd.to_numeric, errors='coerce')
        intermittent = (((f['benzo_on_board'] == 1) & (f['midazolam_inf_now'] == 0)) |
                        ((f['opioid_on_board'] == 1) & (f['opioid_inf_now'] == 0)))
        n_unmatched = int(f.isna().all(axis=1).sum())
        n_excluded_intermittent = int(intermittent.sum())
        print(f"  Intermittent benzo/opioid dose on board: {n_excluded_intermittent:,} of {len(cohort):,} rows "
              f"excluded; {n_unmatched:,} rows had no exposure flags (kept)")
        cohort = cohort[~intermittent.values].drop(columns=['_pid', '_t'] + flags).copy()

    n_per_pt = cohort.groupby('BDSPPatientID').size()
    cohort = cohort[cohort['BDSPPatientID'].isin(
        n_per_pt[n_per_pt >= MIN_OBS_PER_PATIENT].index)].copy()
    n_rows, n_pts = len(cohort), cohort['BDSPPatientID'].nunique()
    pct_rate0 = 100 * (cohort['pump_rate_mg_kg_h'] == 0).mean()
    pct_capped = 100 * (cohort['infusion_rate_mg_per_min'] > RATE_CAP).mean()
    print(f"  Intersection cohort: {n_rows:,} rows, {n_pts:,} patients (from {n_pre:,})")
    print(f"  Rate = 0 (pump off, washout): {pct_rate0:.1f}% of rows; "
          f"capped at {RATE_CAP} mg/min: {pct_capped:.2f}%")
    print(f"  Spearman(Ce, rate): {cohort['Ce_ug_per_mL'].corr(cohort['pump_rate_mg_kg_h'], method='spearman'):.3f}")

    if WITHIN_BETWEEN:
        return run_within_between(cohort, n_rows, n_pts, n_excluded_intermittent)

    # ============ POINT ESTIMATES ============
    point = {}
    for lbl, col in EXPOSURES.items():
        point[lbl] = {
            'nesi': fit_and_decompose(cohort, 'PredictedNESI', col),
            'rass': fit_and_decompose(cohort, '_RASS_numeric', col),
        }
        if point[lbl]['nesi'] is None or point[lbl]['rass'] is None:
            raise RuntimeError(f"Point estimate failed for {lbl}")
        print(f"  point {lbl}: NESI {point[lbl]['nesi']['pct_predictor']:.1f}%, "
              f"RASS {point[lbl]['rass']['pct_predictor']:.1f}%")

    # ============ PAIRED CLUSTER BOOTSTRAP ============
    keys = ('pct_predictor', 'pct_patient', 'pct_residual', 'marginal_r2', 'fe_slope')
    boot = {lbl: {f'{o}_{k}': [] for o in ('nesi', 'rass') for k in keys}
            | {'diff_pct_predictor': [], 'diff_marginal_r2': []}
            for lbl in EXPOSURES}
    rng = np.random.default_rng(RNG_SEED)
    pts = cohort['BDSPPatientID'].unique()
    by_pt = {p: g for p, g in cohort.groupby('BDSPPatientID')}
    n_ok = n_fail = 0
    t0 = time.time()
    print(f"\nPaired bootstrap ({N_BOOTSTRAP} iterations, 4 fits each)...")
    for i in range(N_BOOTSTRAP):
        sample_pts = rng.choice(pts, size=len(pts), replace=True)
        parts = []
        for j, p in enumerate(sample_pts):
            block = by_pt[p].copy()
            block['BDSPPatientID'] = f'b{i}_{j}'
            parts.append(block)
        rs = pd.concat(parts, ignore_index=True)

        fits = {lbl: (fit_and_decompose(rs, 'PredictedNESI', col),
                      fit_and_decompose(rs, '_RASS_numeric', col))
                for lbl, col in EXPOSURES.items()}
        if any(rn is None or rr is None for rn, rr in fits.values()):
            n_fail += 1
            continue
        for lbl, (rn, rr) in fits.items():
            for k in keys:
                boot[lbl][f'nesi_{k}'].append(rn[k])
                boot[lbl][f'rass_{k}'].append(rr[k])
            boot[lbl]['diff_pct_predictor'].append(rn['pct_predictor'] - rr['pct_predictor'])
            boot[lbl]['diff_marginal_r2'].append(rn['marginal_r2'] - rr['marginal_r2'])
        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{N_BOOTSTRAP}  ok={n_ok} fail={n_fail}  ({time.time()-t0:.0f}s)")

    # ============ REPORT ============
    md('# Variance decomposition: NESI vs RASS by propofol exposure (Ce and pump rate)\n')
    md(f'_Generated from `{os.path.basename(RASS_CSV)}` and `{os.path.basename(TS_PARQUET)}`._\n')
    md('## Cohort\n')
    md(f'- propofol_only=True, simulated; Ce, pump rate, weight, NESI and RASS in [−5, 0] '
       f'all available; ≥{MIN_OBS_PER_PATIENT} obs/patient: '
       f'**{n_rows:,} rows** across **{n_pts:,} patients**')
    md('- Every model (NESI/RASS × Ce/rate) uses exactly these rows.')
    if EXCLUDE_INTERMITTENT:
        md(f'- **Sensitivity analysis:** {n_excluded_intermittent:,} observations with an intermittent '
           'benzodiazepine or opioid dose on board (fast-acting in prior 6 h, slow-acting in prior 12 h) '
           'were excluded before the ≥5 obs/patient filter.')
    md(f'- Pump rate = 0 at RASS time (infusion off, drug washing out): {pct_rate0:.1f}% of rows. '
       'Ce can still be nonzero here (washout); rate cannot represent this. This is the main way rate is cruder than Ce.')
    md(f'- Rows with rate capped at {RATE_CAP} mg/min (bolus encoding): {pct_capped:.2f}%')
    md(f"- Spearman correlation between Ce and rate: "
       f"{cohort['Ce_ug_per_mL'].corr(cohort['pump_rate_mg_kg_h'], method='spearman'):.3f}\n")

    md('## Summary: NESI − RASS difference in % variance explained by propofol\n')
    md('| Exposure | NESI % | RASS % | NESI − RASS (paired, pp) | Verdict |')
    md('|---|---|---|---|---|')
    for lbl in EXPOSURES:
        pn, prr, b = point[lbl]['nesi'], point[lbl]['rass'], boot[lbl]
        lo, hi = ci(b['diff_pct_predictor'])
        verdict = ('NESI > RASS' if lo > 0 else 'RASS > NESI' if hi < 0
                   else 'not distinguishable')
        md(f"| {EXPOSURE_DESC[lbl]} | "
           f"{fmt_with_ci(pn['pct_predictor'], b['nesi_pct_predictor'], '{:.1f}')} | "
           f"{fmt_with_ci(prr['pct_predictor'], b['rass_pct_predictor'], '{:.1f}')} | "
           f"{fmt_with_ci(pn['pct_predictor'] - prr['pct_predictor'], b['diff_pct_predictor'], '{:+.2f}')} | "
           f"{verdict} |")
    md(f'\n{n_ok}/{N_BOOTSTRAP} successful bootstrap iterations ({n_fail} failed).\n')

    for lbl in EXPOSURES:
        pn, prr, b = point[lbl]['nesi'], point[lbl]['rass'], boot[lbl]
        md(f'## Variance components: {EXPOSURE_DESC[lbl]}\n')
        md('| Component | NESI | RASS | NESI − RASS (paired) |')
        md('|---|---|---|---|')
        md(f"| % propofol | {fmt_with_ci(pn['pct_predictor'], b['nesi_pct_predictor'], '{:.1f}')}% | "
           f"{fmt_with_ci(prr['pct_predictor'], b['rass_pct_predictor'], '{:.1f}')}% | "
           f"{fmt_with_ci(pn['pct_predictor'] - prr['pct_predictor'], b['diff_pct_predictor'], '{:+.2f}')} pp |")
        md(f"| % patient | {fmt_with_ci(pn['pct_patient'], b['nesi_pct_patient'], '{:.1f}')}% | "
           f"{fmt_with_ci(prr['pct_patient'], b['rass_pct_patient'], '{:.1f}')}% | — |")
        md(f"| % residual | {fmt_with_ci(pn['pct_residual'], b['nesi_pct_residual'], '{:.1f}')}% | "
           f"{fmt_with_ci(prr['pct_residual'], b['rass_pct_residual'], '{:.1f}')}% | — |")
        md(f"| marginal R² | {fmt_with_ci(pn['marginal_r2'], b['nesi_marginal_r2'], '{:.3f}')} | "
           f"{fmt_with_ci(prr['marginal_r2'], b['rass_marginal_r2'], '{:.3f}')} | "
           f"{fmt_with_ci(pn['marginal_r2'] - prr['marginal_r2'], b['diff_marginal_r2'], '{:+.3f}')} |")
        md(f"| slope | {fmt_with_ci(pn['fe_slope'], b['nesi_fe_slope'], '{:.4f}')} | "
           f"{fmt_with_ci(prr['fe_slope'], b['rass_fe_slope'], '{:.4f}')} | — |")
        md('')

    md('## Notes\n')
    md('- **Cluster bootstrap** resamples whole patients with replacement.')
    md('- **Paired bootstrap.** Each iteration fits all four models on the same resampled '
       'patients, so NESI − RASS differences have valid CIs.')
    md('- **Why the comparison is robust to exposure error.** Error in the exposure '
       'attenuates % variance explained for NESI and RASS similarly, because both are fit '
       'against the same exposure on the same rows. Rate, being cruder than Ce, is expected '
       'to give smaller absolute percentages; the NESI − RASS comparison is the quantity of interest.')

    print(f"\nWriting markdown report → {OUT_REPORT_MD} ...")
    tmp = OUT_REPORT_MD + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    os.replace(tmp, OUT_REPORT_MD)
    print('\n'.join(md_lines[:md_lines.index('## Summary: NESI − RASS difference in % variance explained by propofol\n') + 6]))


if __name__ == '__main__':
    try:
        main()
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            print('\a')
        print('\nAll done.')
    except Exception:
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            print('\a')
        raise