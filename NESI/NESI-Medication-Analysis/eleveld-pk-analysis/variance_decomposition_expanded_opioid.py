# variance_decomposition_expanded_opioid.py
"""
Expanded-cohort analysis: propofol with or without concurrent opioids.

Answers the reviewer's opioid comment directly: in a cohort where opioid
exposure is common (infusions allowed), does NESI still carry more propofol /
sedation signal than RASS, and do opioids shift the outcome or modify the
Ce-outcome relationship?

Cohort: COHORT_FLAG == True (propofol, no other sedative infusion; opioid
infusions allowed), PropofolSimStatus='simulated', Ce, NESI, RASS in [-5, 0]
and opioid_on_board all available, >=5 obs per patient.

Opioid covariate: opioid_on_board from add_sedation_windows_opiates.py
  (opioid infusion running at RASS time OR fast-acting dose in prior 6h OR
  slow-acting dose in prior 12h), joined on BDSPPatientID + RASSRecordedDTS.

Models, fit for NESI and for RASS (patient random intercept):
  A.  Ce only:        outcome ~ Ce_c + (1 | patient)
  AB. Ce + opioid:    outcome ~ Ce_c + opioid + (1 | patient)
  B.  Sedation block: outcome ~ Ce_c + opioid + Ce_c:opioid + (1 | patient)
Sequential decomposition: propofol = A; + opioid = AB - A; + interaction = B - AB.
  Ce_c = Ce centered at the cohort mean, so the opioid coefficient is the
  opioid shift at average Ce. Centering does not change any variance share.

Variance share explained by the fixed effects (Nakagawa marginal R²):
  var(X·beta, excluding intercept) / [var(X·beta) + patient var + residual var]
For model B this is the share explained by propofol + opioids jointly,
including the interaction, so no credit-splitting between drugs is needed.

Paired cluster bootstrap: each iteration resamples patients once and fits all
six models (A, AB, B x NESI, RASS) on that resample. Reports NESI − RASS
differences for each model, the sequential increments, and the opioid and
Ce×opioid coefficients for each outcome, with 95% CIs.

Paths (med_config.py): OUTPUT_CSV, OUTPUT_PARQUET, I0001_ASM_ANTIPSYCH_DIR
"""

import os
import time
import warnings
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
from med_config import OUTPUT_CSV, OUTPUT_PARQUET, I0001_ASM_ANTIPSYCH_DIR


# ============ PATHS ============
RASS_CSV      = os.path.join(OUTPUT_CSV, 'wt_ht_per_hosp_FULL_with_eleveld_expanded.csv')
WINDOWS_CSV   = os.path.join(I0001_ASM_ANTIPSYCH_DIR,
                             'i0001_per_eeg_master_exposure_windows_opiates.csv')
# _v2: adds every model's slope/opioid/interaction terms and % patient / % residual with CIs.
# The original report is not overwritten.
OUT_REPORT_MD = os.path.join(OUTPUT_PARQUET, 'variance_decomposition_expanded_opioid_v2.md')


# ============ CONFIG ============
COHORT_FLAG         = 'propofol_no_other_sedative'   # written by eleveld_run_cohort_expanded.py
N_BOOTSTRAP         = 1000
N_JOBS              = -4      # parallel resamples (all cores but 3); same results as one core
import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=True)
_ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples (default 1000)")
_ap.add_argument("--n-jobs", type=int, default=None)
ARGS, _ = _ap.parse_known_args()
N_BOOTSTRAP = ARGS.n_boot or N_BOOTSTRAP
N_JOBS      = ARGS.n_jobs or N_JOBS
RNG_SEED            = 42
MIN_OBS_PER_PATIENT = 5
RASS_COL            = 'R PHS IP RASS'
OPIOID_COL          = 'opioid_on_board'
FIT_METHODS         = ['lbfgs', 'bfgs', 'powell']

MODELS = {
    'A':  ('Ce only',                                  'outcome ~ ce_c'),
    'AB': ('Ce + opioid (no interaction)',             'outcome ~ ce_c + opioid'),
    'B':  ('Sedation block (Ce + opioid + Ce×opioid)', 'outcome ~ ce_c * opioid'),
}
# Sequential (order-dependent) increments in % variance explained:
#   Ce alone = A;  + opioid main effect = AB - A;  + Ce×opioid interaction = B - AB
INCREMENTS = [('Propofol (Ce)', None, 'A'), ('+ Opioid main effect', 'A', 'AB'),
              ('+ Ce × opioid interaction', 'AB', 'B'), ('Total sedation block', None, 'B')]
OUTCOMES = {'nesi': 'PredictedNESI', 'rass': '_RASS_numeric'}


def norm_id(s):
    return s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


# ============ CORE FIT ============
def fit_block(d, outcome_col, formula):
    """Mixed model with patient random intercept. Returns dict or None."""
    x = d[['BDSPPatientID', 'ce_c', 'opioid', outcome_col]].dropna() \
        .rename(columns={outcome_col: 'outcome'})
    if len(x) < 10 or x['BDSPPatientID'].nunique() < 3 or x['outcome'].std() == 0:
        return None
    res = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for method in FIT_METHODS:        # lbfgs first (as in primary analysis); fall back if it fails
            try:
                r = smf.mixedlm(formula, data=x, groups=x['BDSPPatientID']).fit(method=method, reml=True)
            except Exception:
                continue
            # Reject degenerate fits (non-finite likelihood); try the next optimizer.
            if np.isfinite(r.llf) and np.all(np.isfinite(r.fe_params.values)):
                res = r
                break
    if res is None:
        return None
    fe = res.fe_params
    X = res.model.exog                       # fixed-effect design matrix, includes intercept column
    names = res.model.exog_names
    keep = [i for i, n in enumerate(names) if n != 'Intercept']
    fe_var = float(np.var(X[:, keep] @ fe.values[keep], ddof=1)) if keep else 0.0
    re_var = float(res.cov_re.iloc[0, 0])
    resid  = float(res.scale)
    total  = fe_var + re_var + resid
    if total <= 0:
        return None
    return {
        'pct_fixed':   100 * fe_var / total,
        'pct_patient': 100 * re_var / total,
        'pct_resid':   100 * resid / total,
        'b_ce':        fe.get('ce_c', np.nan),
        'b_opioid':    fe.get('opioid', np.nan),
        'b_inter':     fe.get('ce_c:opioid', np.nan),
    }


def ci(a):
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)


def fmt(point, samples, f='{:.2f}'):
    lo, hi = ci(samples)
    return f"{f.format(point)} ({f.format(lo)} – {f.format(hi)})"


def fit_all(d):
    out = {}
    for m, (_, formula) in MODELS.items():
        for o, col in OUTCOMES.items():
            r = fit_block(d, col, formula)
            if r is None:
                return None
            out[(m, o)] = r
    return out


def one_resample(i, sample, by_pt, ce_mean):
    """Fit all models on one cluster-bootstrap resample (patients drawn with replacement)."""
    parts = []
    for j, p in enumerate(sample):
        b = by_pt[p].copy()
        b['BDSPPatientID'] = f'b{i}_{j}'
        parts.append(b)
    rs = pd.concat(parts, ignore_index=True)
    rs['ce_c'] = rs['Ce_ug_per_mL'] - ce_mean      # same centering as the point estimate
    return fit_all(rs)


def main():
    # ============ LOAD COHORT ============
    print(f"Loading {RASS_CSV} ...")
    rass = pd.read_csv(RASS_CSV, parse_dates=['RASSRecordedDTS'], low_memory=False)
    if COHORT_FLAG not in rass.columns:
        raise KeyError(f"COHORT_FLAG '{COHORT_FLAG}' not in {os.path.basename(RASS_CSV)}")
    c = rass[(rass[COHORT_FLAG] == True) &
             (rass['PropofolSimStatus'] == 'simulated') &
             rass['Ce_ug_per_mL'].notna()].copy()
    c['_RASS_numeric'] = pd.to_numeric(c[RASS_COL], errors='coerce')
    c['PredictedNESI'] = pd.to_numeric(c['PredictedNESI'], errors='coerce')
    c['BDSPPatientID'] = norm_id(c['BDSPPatientID'])

    # ============ JOIN OPIOID COVARIATE ============
    w = pd.read_csv(WINDOWS_CSV, usecols=['BDSPPatientID', 'RASSRecordedDTS', OPIOID_COL,
                                          'opioid_inf_now'], low_memory=False)
    w = w.dropna(subset=['RASSRecordedDTS'])
    w['BDSPPatientID'] = norm_id(w['BDSPPatientID'])
    w['RASSRecordedDTS'] = pd.to_datetime(w['RASSRecordedDTS'], errors='coerce')
    w = w.drop_duplicates(['BDSPPatientID', 'RASSRecordedDTS'])
    n_pre = len(c)
    c = c.merge(w, on=['BDSPPatientID', 'RASSRecordedDTS'], how='left')
    print(f"  opioid_on_board matched for {c[OPIOID_COL].notna().sum():,} of {n_pre:,} rows")
    c['opioid'] = pd.to_numeric(c[OPIOID_COL], errors='coerce')

    # ============ INTERSECTION COHORT ============
    c = c[c['PredictedNESI'].notna() & c['_RASS_numeric'].between(-5, 0) & c['opioid'].notna()].copy()
    n_pt = c.groupby('BDSPPatientID').size()
    c = c[c['BDSPPatientID'].isin(n_pt[n_pt >= MIN_OBS_PER_PATIENT].index)].copy()
    c['ce_c'] = c['Ce_ug_per_mL'] - c['Ce_ug_per_mL'].mean()
    n_rows, n_pts = len(c), c['BDSPPatientID'].nunique()
    pct_opi = 100 * c['opioid'].mean()
    pct_inf = 100 * (pd.to_numeric(c['opioid_inf_now'], errors='coerce') == 1).mean()
    pts_both = int(c.groupby('BDSPPatientID')['opioid'].nunique().eq(2).sum())
    print(f"  Expanded cohort: {n_rows:,} rows, {n_pts:,} patients; "
          f"opioid on board {pct_opi:.1f}% of rows (infusion running {pct_inf:.1f}%); "
          f"{pts_both:,} patients with both exposed and unexposed rows")

    # ============ POINT ESTIMATES ============
    point = fit_all(c)
    if point is None:
        raise RuntimeError("Point estimate failed")
    for m in MODELS:
        print(f"  {m}: NESI {point[(m, 'nesi')]['pct_fixed']:.1f}%  RASS {point[(m, 'rass')]['pct_fixed']:.1f}%")
    for lbl, lo, hi in INCREMENTS:
        inc = [point[(hi, o)]['pct_fixed'] - (point[(lo, o)]['pct_fixed'] if lo else 0.0) for o in OUTCOMES]
        print(f"    {lbl:<28} NESI {inc[0]:6.2f} pp   RASS {inc[1]:6.2f} pp")

    # ============ PAIRED CLUSTER BOOTSTRAP ============
    boot = {}
    def add(k, v):
        boot.setdefault(k, []).append(v)
    rng = np.random.default_rng(RNG_SEED)
    pts = c['BDSPPatientID'].unique()
    by_pt = {p: g for p, g in c.groupby('BDSPPatientID')}
    ce_mean = c['Ce_ug_per_mL'].mean()
    # draw every resample up front from one generator (fixed seed), then fit in parallel
    draws = [rng.choice(pts, size=len(pts), replace=True) for _ in range(N_BOOTSTRAP)]
    t0 = time.time()
    print(f"\nPaired bootstrap ({N_BOOTSTRAP} iterations, {len(MODELS) * len(OUTCOMES)} fits each, "
          f"N_JOBS={N_JOBS})...")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(one_resample)(i, sample, by_pt, ce_mean) for i, sample in enumerate(draws))
    print(f"Bootstrap done in {(time.time() - t0) / 60:.1f} min")
    n_ok = n_fail = 0
    for f in results:
        if f is None:
            n_fail += 1
            continue
        for (m, o), r in f.items():
            for k, v in r.items():
                add(f'{m}_{o}_{k}', v)
        for m in MODELS:
            add(f'{m}_diff', f[(m, 'nesi')]['pct_fixed'] - f[(m, 'rass')]['pct_fixed'])
        for lbl, lo, hi in INCREMENTS:
            for o in OUTCOMES:
                add(f'inc_{lbl}_{o}', f[(hi, o)]['pct_fixed'] - (f[(lo, o)]['pct_fixed'] if lo else 0.0))
        n_ok += 1

    # ============ REPORT ============
    md = ['# Expanded cohort: propofol with or without opioids\n',
          f'_Generated from `{os.path.basename(RASS_CSV)}`, opioid covariate from '
          f'`{os.path.basename(WINDOWS_CSV)}` ({OPIOID_COL})._\n',
          '## Cohort\n',
          f'- {COHORT_FLAG}=True, simulated; Ce, NESI, RASS in [−5, 0] and opioid status available; '
          f'≥{MIN_OBS_PER_PATIENT} obs/patient: **{n_rows:,} rows, {n_pts:,} patients**',
          f'- Opioid on board: **{pct_opi:.1f}%** of rows (infusion running at RASS time: {pct_inf:.1f}%)',
          f'- Patients with both opioid-exposed and unexposed observations: {pts_both:,}\n',
          '## % variance explained by fixed effects (NESI vs RASS, paired)\n',
          '| Model | NESI % | RASS % | NESI − RASS (pp) |',
          '|---|---|---|---|']
    for m, (lbl, _) in MODELS.items():
        pn, pr = point[(m, 'nesi')]['pct_fixed'], point[(m, 'rass')]['pct_fixed']
        md.append(f"| {lbl} | {fmt(pn, boot.get(f'{m}_nesi_pct_fixed', []), '{:.1f}')} | "
                  f"{fmt(pr, boot.get(f'{m}_rass_pct_fixed', []), '{:.1f}')} | "
                  f"{fmt(pn - pr, boot.get(f'{m}_diff', []), '{:+.2f}')} |")
    # Every model: coefficients and variance components, all with CIs (for the main-text table)
    md += ['\n## All models: coefficients and variance components\n',
           'Ce centered at cohort mean (slope per µg/mL); opioid = opioid on board (0/1). '
           'Ce slope in the interaction model is the slope with no opioid on board. '
           '% sedation = variance explained by all fixed effects in that model.\n',
           '| Outcome | Model | Ce slope | Opioid shift | Ce × opioid | % sedation | % patient | % residual |',
           '|---|---|---|---|---|---|---|---|']
    for o, oname in (('nesi', 'NESI'), ('rass', 'RASS')):
        for m, (lbl, _) in MODELS.items():
            r = point[(m, o)]
            cell = lambda k, f: (fmt(r[k], boot.get(f'{m}_{o}_{k}', []), f)
                                 if np.isfinite(r[k]) else '—')
            md.append(f"| {oname} | {lbl} | {cell('b_ce', '{:.4f}')} | {cell('b_opioid', '{:.4f}')} | "
                      f"{cell('b_inter', '{:.4f}')} | {cell('pct_fixed', '{:.1f}')} | "
                      f"{cell('pct_patient', '{:.1f}')} | {cell('pct_resid', '{:.1f}')} |")
    md += ['\n## Sequential variance decomposition (order: Ce, then opioid, then interaction)\n',
           'Percentage points of total variance; increments depend on entry order.\n',
           '| Component | NESI (pp) | RASS (pp) |', '|---|---|---|']
    for lbl, lo, hi in INCREMENTS:
        vals = []
        for o in OUTCOMES:
            pt = point[(hi, o)]['pct_fixed'] - (point[(lo, o)]['pct_fixed'] if lo else 0.0)
            vals.append(fmt(pt, boot.get(f'inc_{lbl}_{o}', []), '{:.2f}'))
        md.append(f"| {lbl} | {vals[0]} | {vals[1]} |")
    md += ['\n## Sedation-block coefficients (model B)\n',
           'Ce centered at cohort mean; opioid = opioid on board (0/1).\n',
           '| Term | NESI | RASS | Meaning |', '|---|---|---|---|']
    for k, lbl, meaning in [('b_ce', 'Ce (no opioid)', 'Ce slope when no opioid on board'),
                            ('b_opioid', 'Opioid', 'Shift in outcome with opioid on board, at average Ce'),
                            ('b_inter', 'Ce × opioid', 'Change in Ce slope when opioid on board')]:
        md.append(f"| {lbl} | {fmt(point[('B', 'nesi')][k], boot.get(f'B_nesi_{k}', []), '{:.4f}')} | "
                  f"{fmt(point[('B', 'rass')][k], boot.get(f'B_rass_{k}', []), '{:.4f}')} | {meaning} |")
    md += [f'\n{n_ok}/{N_BOOTSTRAP} successful bootstrap iterations ({n_fail} failed).\n',
           '## Notes\n',
           '- Model A in this cohort is directly comparable to the primary analysis; model B adds '
           'opioid exposure. The gap between A and B is the extra variance attributable to opioids '
           '(main effect plus interaction).',
           '- NESI and RASS are on different scales, so compare coefficients within an outcome, '
           'not across outcomes; the % variance rows are the cross-outcome comparison.',
           '- Opioid exposure is a binary flag, cruder than Ce; its contribution is likely underestimated.']

    os.makedirs(os.path.dirname(OUT_REPORT_MD), exist_ok=True)
    tmp = OUT_REPORT_MD + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md))
    os.replace(tmp, OUT_REPORT_MD)
    print('\n'.join(md))
    print(f"\nSaved → {OUT_REPORT_MD}")


if __name__ == '__main__':
    try:
        main()
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            print('\a')
    except Exception:
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            print('\a')
        raise