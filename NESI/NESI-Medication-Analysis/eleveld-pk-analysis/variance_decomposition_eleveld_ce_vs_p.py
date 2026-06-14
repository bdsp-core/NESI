# variance_decomposition_eleveld_ce_vs_pump.py
"""
Variance decomposition comparing two predictors of NESI / RASS:
  - Predicted effect-site concentration (Ce, µg/mL)
  - Pump infusion rate at the RASS time (mg/min, instantaneous)

Mixed-effects model: outcome ~ predictor + (1 | patient)

For each (outcome, predictor, subset), reports:
  - Variance components: % predictor, % patient, % residual
  - Marginal R² (fixed effects alone)
  - Conditional R² (fixed + random)
  - ICC
  - Fixed-effect slope on the predictor
All with cluster-bootstrap 95% CIs (resample patients with replacement).

Cohort: propofol_only=True AND PropofolSimStatus='simulated' AND Ce not null,
≥5 obs per patient. Sensitivity subset additionally requires the predictor
in question to be > 0.

Methods sentence (suggested for paper):
  "We compared the variance in NESI and RASS attributable to two
  alternative propofol exposure proxies — instantaneous pump infusion
  rate (the crude clinical signal) and Eleveld-predicted effect-site
  concentration (Ce, accounting for pharmacokinetic delay and
  redistribution). For each predictor we fit a linear mixed-effects model
  with patient as random intercept (outcome ~ predictor + (1|patient)),
  decomposed total variance into predictor / patient / residual
  components following Nakagawa & Schielzeth, and obtained 95% CIs by
  cluster-resampling patients (1000 iterations)."
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from path_configs import OUTPUT_CSV, OUTPUT_PARQUET


# ============ PATHS ============
RASS_CSV      = os.path.join(OUTPUT_CSV,     'wt_ht_per_hosp_FULL_with_eleveld.csv')
TS_PARQUET    = os.path.join(OUTPUT_PARQUET, 'eleveld_timeseries.parquet')
OUT_REPORT_MD = os.path.join(OUTPUT_PARQUET, 'variance_decomposition_eleveld_ce_vs_pump.md')


# ============ CONFIG ============
N_BOOTSTRAP         = 1000
RNG_SEED            = 42
MIN_OBS_PER_PATIENT = 5
RASS_COL            = 'R PHS IP RASS'


# ============ MARKDOWN BUILDER ============
md_lines = []
def md(s=''):
    md_lines.append(s)


# ============ CORE FIT ============
def fit_and_decompose(df, outcome_col, predictor_col):
    """Fit outcome ~ predictor + (1 | patient). Returns dict or None."""
    if len(df) < 10:
        return None
    d = df[['BDSPPatientID', predictor_col, outcome_col]].dropna()
    d = d.rename(columns={outcome_col: 'outcome', predictor_col: 'pred'})
    if len(d) < 10 or d['BDSPPatientID'].nunique() < 3:
        return None
    if d['pred'].std() == 0 or d['outcome'].std() == 0:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mdl = smf.mixedlm('outcome ~ pred', data=d,
                              groups=d['BDSPPatientID'])
            result = mdl.fit(method='lbfgs', reml=True)
    except Exception:
        return None

    fe_slope     = result.fe_params['pred']
    fe_slope_se  = result.bse_fe['pred']
    fe_slope_p   = result.pvalues['pred']

    random_var   = float(result.cov_re.iloc[0, 0])
    residual_var = float(result.scale)
    pred_var     = d['pred'].var()
    fe_var       = (fe_slope ** 2) * pred_var
    total_var    = fe_var + random_var + residual_var
    if total_var <= 0:
        return None

    return {
        'n':                len(d),
        'n_patients':       d['BDSPPatientID'].nunique(),
        'fe_slope':         fe_slope,
        'fe_slope_se':      fe_slope_se,
        'fe_slope_p':       fe_slope_p,
        'pct_predictor':    100 * fe_var / total_var,
        'pct_patient':      100 * random_var / total_var,
        'pct_residual':     100 * residual_var / total_var,
        'marginal_r2':      fe_var / total_var,
        'conditional_r2':   (fe_var + random_var) / total_var,
        'icc':              random_var / (random_var + residual_var),
    }


def cluster_bootstrap(df, outcome_col, predictor_col, n_iter=N_BOOTSTRAP, seed=RNG_SEED):
    """Cluster-resample patients, refit, collect components."""
    rng    = np.random.default_rng(seed)
    pts    = df['BDSPPatientID'].unique()
    by_pt  = {p: df[df['BDSPPatientID'] == p] for p in pts}
    keys   = ('pct_predictor', 'pct_patient', 'pct_residual',
              'marginal_r2', 'conditional_r2', 'icc', 'fe_slope')
    boot   = {k: [] for k in keys}
    n_ok = n_fail = 0
    t0 = time.time()

    for i in range(n_iter):
        sample_pts = rng.choice(pts, size=len(pts), replace=True)
        parts = []
        for j, p in enumerate(sample_pts):
            block = by_pt[p].copy()
            block['BDSPPatientID'] = f'b{i}_{j}'
            parts.append(block)
        rs = pd.concat(parts, ignore_index=True)

        result = fit_and_decompose(rs, outcome_col, predictor_col)
        if result is None:
            n_fail += 1
            continue
        for k in keys:
            boot[k].append(result[k])
        n_ok += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"      {i+1}/{n_iter}  ok={n_ok} fail={n_fail}  "
                  f"({elapsed:.0f}s)")

    return boot, n_ok, n_fail


def ci(arr, lo=2.5, hi=97.5):
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (np.percentile(arr, lo), np.percentile(arr, hi))


def fmt_with_ci(point, samples, fmt='{:.3f}'):
    lo, hi = ci(samples)
    return f"{fmt.format(point)} ({fmt.format(lo)} – {fmt.format(hi)})"


# ============ LOAD ============
print(f"Loading {RASS_CSV} ...")
rass = pd.read_csv(RASS_CSV, parse_dates=[
    'HospitalAdmitDTS', 'HospitalDischargeDTS', 'RASSRecordedDTS'
])
print(f"  {len(rass):,} rows")

cohort = rass[
    (rass['propofol_only'] == True) &
    (rass['PropofolSimStatus'] == 'simulated') &
    rass['Ce_ug_per_mL'].notna()
].copy()
cohort['_RASS_numeric'] = pd.to_numeric(cohort[RASS_COL],     errors='coerce')
cohort['PredictedNESI'] = pd.to_numeric(cohort['PredictedNESI'], errors='coerce')

# Restrict to the intersection cohort: rows with Ce, NESI, AND RASS all
# available (RASS in [-5, 0]). This ensures both NESI and RASS variance
# decompositions are computed on the same patients, enabling direct
# comparison of the variance components across outcomes.
n_pre_intersection = len(cohort)
cohort = cohort[
    cohort['PredictedNESI'].notna() &
    cohort['_RASS_numeric'].notna() &
    cohort['_RASS_numeric'].between(-5, 0)
].copy()
print(f"  Intersection cohort (Ce + NESI + RASS all available, "
      f"RASS in [-5, 0]): {len(cohort):,} rows "
      f"(from {n_pre_intersection:,} before intersection filter)")

n_per_pt = cohort.groupby('BDSPPatientID').size()
keep_pts = n_per_pt[n_per_pt >= MIN_OBS_PER_PATIENT].index
cohort   = cohort[cohort['BDSPPatientID'].isin(keep_pts)].copy()
print(f"  After ≥{MIN_OBS_PER_PATIENT}/patient: "
      f"{len(cohort):,} rows, {cohort['BDSPPatientID'].nunique():,} patients")


# ============ ATTACH PUMP RATE FROM PARQUET ============
print(f"Loading {TS_PARQUET} ...")
import duckdb
ts = duckdb.query(f"""
    SELECT BDSPPatientID, HospitalAdmitDTS, t_min_from_admit,
           infusion_rate_mg_per_min
    FROM read_parquet('{TS_PARQUET.replace(os.sep, '/')}')
""").df()
print(f"  {len(ts):,} time-series rows")

# Group by (patient, admit) for fast lookup. We use nearest-neighbor on
# t_min_from_admit because the rate is piecewise-constant — linear
# interpolation across a rate-change boundary would average two regimes.
ts_by_stay = {}
for (pt, ad), g in ts.groupby(['BDSPPatientID', 'HospitalAdmitDTS']):
    g = g.sort_values('t_min_from_admit')
    ts_by_stay[(pt, ad)] = (
        g['t_min_from_admit'].values,
        g['infusion_rate_mg_per_min'].values,
    )
del ts

# Bolus segments (very high rate, ~0.1 min wide) would dominate the
# instantaneous rate signal even though the bolus contributes only briefly.
# Cap rate at 100 mg/min for the predictor — this is what a clinician
# would see on the pump display (continuous infusion rate, not the brief
# bolus push).
RATE_CAP = 100  # mg/min

def get_rate_at(pt, admit, t_rass_min):
    key = (pt, admit)
    if key not in ts_by_stay:
        return np.nan
    t_grid, r_grid = ts_by_stay[key]
    if t_rass_min < t_grid[0] or t_rass_min > t_grid[-1]:
        return np.nan
    # Find rightmost grid point ≤ t_rass_min (last rate change before RASS).
    idx = np.searchsorted(t_grid, t_rass_min, side='right') - 1
    idx = max(0, idx)
    rate = r_grid[idx]
    if rate > RATE_CAP:    # cap bolus spikes
        return RATE_CAP
    return rate

print("Attaching pump rate to RASS rows...")
cohort['t_min_from_admit'] = (
    (cohort['RASSRecordedDTS'] - cohort['HospitalAdmitDTS'])
    .dt.total_seconds() / 60.0
)
cohort['pump_rate_mg_per_min'] = cohort.apply(
    lambda r: get_rate_at(r['BDSPPatientID'], r['HospitalAdmitDTS'],
                          r['t_min_from_admit']),
    axis=1,
)
n_with_rate = cohort['pump_rate_mg_per_min'].notna().sum()
print(f"  {n_with_rate:,} rows have pump rate (of {len(cohort):,})")


md('# Variance decomposition: NESI / RASS by predictor (Ce vs pump rate)\n')
md(f'_Generated from `{os.path.basename(RASS_CSV)}` and '
   f'`{os.path.basename(TS_PARQUET)}`._\n')

md('## Cohort\n')
md(f'- propofol_only=True, simulated, with Ce, NESI, and RASS in [−5, 0] '
   f'all available; ≥{MIN_OBS_PER_PATIENT} obs/patient: '
   f'**{len(cohort):,} rows** across **{cohort["BDSPPatientID"].nunique():,} patients**')
md('- Restricting to the intersection of available outcomes ensures the '
   'NESI and RASS variance decompositions are computed on the same '
   'patients, enabling direct comparison.')

md('## Method\n')
md('Linear mixed-effects model: `outcome ~ predictor + (1 | patient)`, '
   'fit by REML. Variance decomposition follows Nakagawa & Schielzeth: '
   'predictor variance estimated as slope² × var(predictor), patient '
   'variance from the random-intercept estimate, residual from the '
   'model scale parameter. 95% CIs from cluster-bootstrap '
   f'(resample patients with replacement, refit; {N_BOOTSTRAP} iterations). '
   'For each (outcome, predictor) pairing we run two analyses: the full '
   'cohort, and a sensitivity subset restricted to predictor > 0 to '
   'control for the "drug present vs absent" contrast as opposed to '
   'genuine dose-response.\n')


# ============ RUN MODELS ============
# Paired bootstrap: for each resample, fit BOTH NESI and RASS models on the
# same resampled patients, then compute differences. This gives valid CIs on
# the NESI - RASS difference in variance components.

subsets = [
    ('full cohort', lambda d: d),
    # ('Ce > 0',      lambda d: d[d['Ce_ug_per_mL'] > 0]),   # optional
]

paired_results = []  # list of dicts, one per subset

for subset_lbl, subset_fn in subsets:
    print(f"\n=== Paired bootstrap: {subset_lbl} ===")
    d = subset_fn(cohort).dropna(subset=['Ce_ug_per_mL', 'PredictedNESI',
                                          '_RASS_numeric'])
    if d['_RASS_numeric'].between(-5, 0).all() is False:
        d = d[d['_RASS_numeric'].between(-5, 0)]
    print(f"   n={len(d):,}, patients={d['BDSPPatientID'].nunique():,}")

    # Point estimates (full data)
    pt_nesi = fit_and_decompose(d, 'PredictedNESI', 'Ce_ug_per_mL')
    pt_rass = fit_and_decompose(d, '_RASS_numeric', 'Ce_ug_per_mL')
    if pt_nesi is None or pt_rass is None:
        print("   point estimate failed for one or both outcomes"); continue
    print(f"   point: NESI Ce%={pt_nesi['pct_predictor']:.1f}, "
          f"RASS Ce%={pt_rass['pct_predictor']:.1f}, "
          f"diff={pt_nesi['pct_predictor']-pt_rass['pct_predictor']:+.2f}")

    # Bootstrap
    rng = np.random.default_rng(RNG_SEED)
    pts = d['BDSPPatientID'].unique()
    by_pt = {p: d[d['BDSPPatientID'] == p] for p in pts}

    boot = {
        'nesi_pct_predictor': [], 'rass_pct_predictor': [],
        'nesi_pct_patient':   [], 'rass_pct_patient':   [],
        'nesi_pct_residual':  [], 'rass_pct_residual':  [],
        'nesi_marginal_r2':   [], 'rass_marginal_r2':   [],
        'nesi_fe_slope':      [], 'rass_fe_slope':      [],
        'diff_pct_predictor': [], 'diff_marginal_r2':   [],
    }
    n_ok = n_fail = 0
    t0 = time.time()

    for i in range(N_BOOTSTRAP):
        sample_pts = rng.choice(pts, size=len(pts), replace=True)
        parts = []
        for j, p in enumerate(sample_pts):
            block = by_pt[p].copy()
            block['BDSPPatientID'] = f'b{i}_{j}'
            parts.append(block)
        rs = pd.concat(parts, ignore_index=True)

        rn = fit_and_decompose(rs, 'PredictedNESI', 'Ce_ug_per_mL')
        rr = fit_and_decompose(rs, '_RASS_numeric', 'Ce_ug_per_mL')
        if rn is None or rr is None:
            n_fail += 1; continue

        boot['nesi_pct_predictor'].append(rn['pct_predictor'])
        boot['rass_pct_predictor'].append(rr['pct_predictor'])
        boot['nesi_pct_patient'].append(rn['pct_patient'])
        boot['rass_pct_patient'].append(rr['pct_patient'])
        boot['nesi_pct_residual'].append(rn['pct_residual'])
        boot['rass_pct_residual'].append(rr['pct_residual'])
        boot['nesi_marginal_r2'].append(rn['marginal_r2'])
        boot['rass_marginal_r2'].append(rr['marginal_r2'])
        boot['nesi_fe_slope'].append(rn['fe_slope'])
        boot['rass_fe_slope'].append(rr['fe_slope'])
        boot['diff_pct_predictor'].append(rn['pct_predictor'] - rr['pct_predictor'])
        boot['diff_marginal_r2'].append(rn['marginal_r2'] - rr['marginal_r2'])
        n_ok += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"      {i+1}/{N_BOOTSTRAP}  ok={n_ok} fail={n_fail}  "
                  f"({elapsed:.0f}s)")

    paired_results.append({
        'subset': subset_lbl,
        'n': len(d),
        'n_patients': d['BDSPPatientID'].nunique(),
        'point_nesi': pt_nesi, 'point_rass': pt_rass,
        'boot': boot, 'n_ok': n_ok, 'n_fail': n_fail,
    })

# ============ ASSEMBLE REPORT ============
# ============ ASSEMBLE REPORT ============
md('## Variance components: NESI vs RASS (paired bootstrap)\n')
md('Models fit on the same intersection cohort and resampled patients '
   'within each bootstrap iteration, allowing direct comparison of '
   'NESI vs RASS variance components.\n')

for pr in paired_results:
    pn, pr_rass, boot = pr['point_nesi'], pr['point_rass'], pr['boot']
    md(f"### Subset: {pr['subset']}  (n={pr['n']:,}, "
       f"{pr['n_patients']:,} patients)\n")

    md('| Component | NESI | RASS | NESI − RASS (paired) |')
    md('|---|---|---|---|')
    md(f"| % predictor (Ce) | "
       f"{fmt_with_ci(pn['pct_predictor'], boot['nesi_pct_predictor'], '{:.1f}')}% | "
       f"{fmt_with_ci(pr_rass['pct_predictor'], boot['rass_pct_predictor'], '{:.1f}')}% | "
       f"{fmt_with_ci(pn['pct_predictor']-pr_rass['pct_predictor'], boot['diff_pct_predictor'], '{:+.2f}')} pp |")
    md(f"| % patient | "
       f"{fmt_with_ci(pn['pct_patient'], boot['nesi_pct_patient'], '{:.1f}')}% | "
       f"{fmt_with_ci(pr_rass['pct_patient'], boot['rass_pct_patient'], '{:.1f}')}% | — |")
    md(f"| % residual | "
       f"{fmt_with_ci(pn['pct_residual'], boot['nesi_pct_residual'], '{:.1f}')}% | "
       f"{fmt_with_ci(pr_rass['pct_residual'], boot['rass_pct_residual'], '{:.1f}')}% | — |")
    md(f"| marginal R² | "
       f"{fmt_with_ci(pn['marginal_r2'], boot['nesi_marginal_r2'], '{:.3f}')} | "
       f"{fmt_with_ci(pr_rass['marginal_r2'], boot['rass_marginal_r2'], '{:.3f}')} | "
       f"{fmt_with_ci(pn['marginal_r2']-pr_rass['marginal_r2'], boot['diff_marginal_r2'], '{:+.3f}')} |")
    md('')

    diff_lo, diff_hi = ci(boot['diff_pct_predictor'])
    verdict = ('NESI > RASS (CI excludes 0)' if diff_lo > 0
               else 'RASS > NESI (CI excludes 0)' if diff_hi < 0
               else 'not distinguishable (CI includes 0)')
    md(f"**Verdict on Δ % predictor**: {verdict}\n")
    md(f"- {pr['n_ok']}/{N_BOOTSTRAP} successful bootstrap iterations.\n")

md('## Notes\n')
md('- **Cluster bootstrap** resamples whole patients with replacement, '
   'so within-patient correlation in repeated outcome observations is '
   'respected when constructing CIs.')
md('- **Paired bootstrap.** Within each iteration both NESI and RASS '
   'models are fit on the *same* resampled patients, then the difference '
   'in variance components is recorded. This accounts for the correlation '
   'between the two estimates that would otherwise make CI-overlap '
   'comparisons unreliable.')
md('- **Intersection cohort.** Analyses use only RASS observations where '
   'C_e, NESI, and RASS (in [−5, 0]) are all simultaneously available, '
   'ensuring NESI and RASS variance decompositions are computed on the '
   'same patients.')


# ============ WRITE ============
print(f"\nWriting markdown report → {OUT_REPORT_MD} ...")
tmp = OUT_REPORT_MD + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))
os.replace(tmp, OUT_REPORT_MD)
print('  done.')


# ============ COMPLETION SOUND ============
try:
    import winsound
    winsound.MessageBeep(winsound.MB_OK)
except Exception:
    print('\a')
print('\nAll done.')