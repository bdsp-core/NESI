# nesi_vs_rass_correlation_comparison.py
"""
Compare the strength of association between predicted Ce and each of two
outcomes (NESI, RASS) in the propofol-only cohort, using a paired cluster
bootstrap.

For each of N bootstrap iterations:
  1. Resample patients with replacement (cluster bootstrap).
  2. Compute Spearman ρ for NESI vs Ce on that resample.
  3. Compute Spearman ρ for RASS vs Ce on that resample.
  4. Compute Δ = |ρ_NESI| − |ρ_RASS| (paired within the resample).

The same patients are used for both correlations within each resample,
so Δ captures the within-resample contrast and the bootstrap CI on Δ
properly reflects the correlation between the two estimates.

Cohort:
  - propofol_only=True, PropofolSimStatus='simulated', Ce not null
  - For RASS: RASS in [−5, 0] (positive RASS excluded — agitation is
    delirium/withdrawal-driven, not propofol-driven)
  - For NESI: PredictedNESI not null

Outputs:
  - nesi_vs_rass_correlation_comparison.md
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from path_configs import OUTPUT_CSV, OUTPUT_PARQUET


# ============ PATHS ============
RASS_CSV      = os.path.join(OUTPUT_CSV,     'wt_ht_per_hosp_FULL_with_eleveld.csv')
OUT_REPORT_MD = os.path.join(OUTPUT_PARQUET, 'nesi_vs_rass_correlation_comparison.md')


# ============ CONFIG ============
N_BOOTSTRAP = 1000
RNG_SEED    = 42
RASS_COL    = 'R PHS IP RASS'


# ============ MARKDOWN BUILDER ============
md_lines = []
def md(s=''):
    md_lines.append(s)


md('# NESI vs RASS — comparing strength of association with predicted Ce\n')
md(f'_Generated from `{os.path.basename(RASS_CSV)}`._\n')


# ============ LOAD ============
print(f"Loading {RASS_CSV} ...")
rass = pd.read_csv(RASS_CSV, parse_dates=[
    'HospitalAdmitDTS', 'HospitalDischargeDTS', 'RASSRecordedDTS'
])
print(f"  {len(rass):,} rows")

# Base cohort: propofol-only, simulated, Ce non-null
base = rass[
    (rass['propofol_only'] == True) &
    (rass['PropofolSimStatus'] == 'simulated') &
    rass['Ce_ug_per_mL'].notna()
].copy()
base['rass_numeric'] = pd.to_numeric(base[RASS_COL],     errors='coerce')
base['nesi_numeric'] = pd.to_numeric(base['PredictedNESI'], errors='coerce')

# Per-row availability flags
base['_has_nesi'] = base['nesi_numeric'].notna()
base['_has_rass'] = (base['rass_numeric'].notna() &
                     base['rass_numeric'].between(-5, 0))

# For the paired comparison we need rows that have BOTH
both_avail = base[base['_has_nesi'] & base['_has_rass']].copy()

n_both = len(both_avail)
n_pts  = both_avail['BDSPPatientID'].nunique()
print(f"  Rows with both NESI and RASS in [-5, 0]: "
      f"{n_both:,} across {n_pts:,} patients")

md('## Cohort\n')
md(f'- Rows with **both** NESI and RASS available '
   f'(propofol_only=True, simulated, Ce not null, RASS in [−5, 0], '
   f'NESI not null): **{n_both:,}** across **{n_pts:,} patients**')
md('- The paired comparison requires both outcomes on the same row so '
   'each bootstrap resample contributes to both ρ estimates simultaneously.\n')


# ============ PAIRED CLUSTER BOOTSTRAP ============
def paired_cluster_bootstrap(df, n_iter=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    pts = df['BDSPPatientID'].unique()
    by_pt = {p: df[df['BDSPPatientID'] == p][
                  ['Ce_ug_per_mL', 'nesi_numeric', 'rass_numeric']
              ].values for p in pts}

    rho_obs_nesi = spearmanr(df['Ce_ug_per_mL'], df['nesi_numeric']).statistic
    rho_obs_rass = spearmanr(df['Ce_ug_per_mL'], df['rass_numeric']).statistic
    delta_obs    = abs(rho_obs_nesi) - abs(rho_obs_rass)

    rho_nesi_b = np.full(n_iter, np.nan)
    rho_rass_b = np.full(n_iter, np.nan)
    delta_b    = np.full(n_iter, np.nan)

    n_ok = 0
    for i in range(n_iter):
        sample_pts = rng.choice(pts, size=len(pts), replace=True)
        rows = np.vstack([by_pt[p] for p in sample_pts])
        if len(rows) < 3:
            continue
        ce, nesi, rass_ = rows[:, 0], rows[:, 1], rows[:, 2]
        if (np.std(ce) == 0 or np.std(nesi) == 0 or np.std(rass_) == 0):
            continue
        rn = spearmanr(ce, nesi).statistic
        rr = spearmanr(ce, rass_).statistic
        if np.isnan(rn) or np.isnan(rr):
            continue
        rho_nesi_b[i] = rn
        rho_rass_b[i] = rr
        delta_b[i]    = abs(rn) - abs(rr)
        n_ok += 1

    return {
        'rho_obs_nesi': rho_obs_nesi,
        'rho_obs_rass': rho_obs_rass,
        'delta_obs':    delta_obs,
        'rho_nesi_b':   rho_nesi_b[~np.isnan(rho_nesi_b)],
        'rho_rass_b':   rho_rass_b[~np.isnan(rho_rass_b)],
        'delta_b':      delta_b[~np.isnan(delta_b)],
        'n_ok':         n_ok,
    }


def ci(arr, lo=2.5, hi=97.5):
    if len(arr) == 0:
        return (np.nan, np.nan)
    return (np.percentile(arr, lo), np.percentile(arr, hi))


print(f"\nRunning paired bootstrap ({N_BOOTSTRAP} iters) — full cohort ...")
res_full = paired_cluster_bootstrap(both_avail)
print(f"  ρ NESI vs Ce: {res_full['rho_obs_nesi']:+.3f}  "
      f"95% CI {ci(res_full['rho_nesi_b'])}")
print(f"  ρ RASS vs Ce: {res_full['rho_obs_rass']:+.3f}  "
      f"95% CI {ci(res_full['rho_rass_b'])}")
print(f"  Δ |ρ| (NESI − RASS): {res_full['delta_obs']:+.3f}  "
      f"95% CI {ci(res_full['delta_b'])}  "
      f"({res_full['n_ok']}/{N_BOOTSTRAP} valid)")


# Sensitivity: Ce > 0
both_pos = both_avail[both_avail['Ce_ug_per_mL'] > 0]
n_pos     = len(both_pos)
n_pts_pos = both_pos['BDSPPatientID'].nunique()

print(f"\nSensitivity (Ce > 0): {n_pos:,} rows, {n_pts_pos:,} patients ...")
res_pos = paired_cluster_bootstrap(both_pos)
print(f"  ρ NESI vs Ce: {res_pos['rho_obs_nesi']:+.3f}  "
      f"95% CI {ci(res_pos['rho_nesi_b'])}")
print(f"  ρ RASS vs Ce: {res_pos['rho_obs_rass']:+.3f}  "
      f"95% CI {ci(res_pos['rho_rass_b'])}")
print(f"  Δ |ρ| (NESI − RASS): {res_pos['delta_obs']:+.3f}  "
      f"95% CI {ci(res_pos['delta_b'])}  "
      f"({res_pos['n_ok']}/{N_BOOTSTRAP} valid)")


# ============ MARKDOWN REPORT ============
def fmt_rho(point, samples):
    lo, hi = ci(samples)
    return f"{point:+.3f} (CI {lo:+.3f}, {hi:+.3f})"

def verdict(samples):
    lo, hi = ci(samples)
    if np.isnan(lo) or np.isnan(hi):
        return '—'
    if lo > 0:
        return '|NESI| > |RASS| (CI excludes 0)'
    if hi < 0:
        return '|RASS| > |NESI| (CI excludes 0)'
    return 'no significant difference (CI includes 0)'


md('## Method\n')
md('Spearman rank correlation was computed between predicted effect-site '
   'propofol concentration (Ce) and each of two outcomes (NESI, RASS), '
   'using only RASS observations where both outcomes were available '
   'on the same row. To compare the strength of association between the '
   'two outcomes, we used a paired cluster bootstrap: in each of '
   f'{N_BOOTSTRAP} iterations, patients were resampled with replacement, '
   'and Spearman ρ was computed for both NESI vs Ce and RASS vs Ce on the '
   'identical resample. The difference in absolute correlations '
   '(Δ = |ρ_NESI| − |ρ_RASS|) was recorded for each iteration. A 95% '
   'percentile confidence interval on Δ that excludes zero indicates the '
   'two associations differ in strength at α = 0.05 *as observed in this '
   'cohort*; this is a paired within-cohort comparison rather than a test '
   'of the underlying population correlations.\n')

md('## Results\n')

md('### Full cohort\n')
md('| comparison | value (95% CI) |')
md('|---|---|')
md(f'| Spearman ρ, NESI vs Ce | {fmt_rho(res_full["rho_obs_nesi"], res_full["rho_nesi_b"])} |')
md(f'| Spearman ρ, RASS vs Ce | {fmt_rho(res_full["rho_obs_rass"], res_full["rho_rass_b"])} |')
md(f'| Δ \\|ρ\\| (NESI − RASS) | {fmt_rho(res_full["delta_obs"], res_full["delta_b"])} |')
md(f'| Verdict | {verdict(res_full["delta_b"])} |')
md(f'| Valid bootstrap iters | {res_full["n_ok"]} / {N_BOOTSTRAP} |')
md('')

md('### Sensitivity analysis: Ce > 0 only\n')
md(f'_Restricted to {n_pos:,} rows ({n_pts_pos:,} patients) where the '
   'patient still has measurable propofol exposure — controls for the '
   'possibility that the correlations are driven by the "drug present vs '
   'absent" contrast rather than dose-response._\n')
md('| comparison | value (95% CI) |')
md('|---|---|')
md(f'| Spearman ρ, NESI vs Ce | {fmt_rho(res_pos["rho_obs_nesi"], res_pos["rho_nesi_b"])} |')
md(f'| Spearman ρ, RASS vs Ce | {fmt_rho(res_pos["rho_obs_rass"], res_pos["rho_rass_b"])} |')
md(f'| Δ \\|ρ\\| (NESI − RASS) | {fmt_rho(res_pos["delta_obs"], res_pos["delta_b"])} |')
md(f'| Verdict | {verdict(res_pos["delta_b"])} |')
md(f'| Valid bootstrap iters | {res_pos["n_ok"]} / {N_BOOTSTRAP} |')
md('')


md('## Notes\n')
md('- **Why paired bootstrap.** Computing two separate bootstrap CIs and '
   'comparing whether they overlap is not a valid test of difference, '
   'because the two ρ estimates share patients and are not independent. '
   'Resampling the same patients for both ρ values within each iteration '
   'lets the bootstrap automatically capture this correlation, so the CI '
   'on Δ accounts for it.')
md('- **Absolute values.** RASS ρ is expected to be negative (more drug '
   '→ deeper sedation → more negative RASS), NESI ρ is expected to be '
   'positive. Comparing magnitudes is the meaningful question: which '
   'outcome is more strongly tied to predicted concentration, regardless '
   'of sign.')
md('- **Cohort-level claim.** A significant Δ here means the two '
   'correlations differ as observed in this cohort. Generalizing to the '
   'underlying population would require additional assumptions (e.g., '
   'Steiger\'s test for dependent correlations) that are stronger than '
   'what cluster bootstrap requires.')


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