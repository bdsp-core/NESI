# rank_correlation.py
"""
Spearman rank correlation of propofol Ce with NESI and with RASS, with 95% cluster-bootstrap CIs.

Not run on its own: the variance-decomposition scripts call it with --correlations, after
building their cohort, so the correlations use exactly the same rows as the variance
decomposition (same patients, same >=5 obs/patient rule).

  variance_decomposition_eleveld_ce_vs_pump.py --correlations   -> propofol-only cohort
  variance_decomposition_expanded_opioid.py --correlations      -> opioid infusions allowed

Pooled rho treats repeated observations as independent, so the CI comes from resampling
patients (cluster bootstrap, percentile CI). Descriptive only: no NESI-vs-RASS test.
Also reports rho restricted to Ce > 0 (dose-response among exposed rows, not drug on vs off),
and cohort descriptives on the same rows: median (IQR) of RASS, NESI and Ce, across
observations and across patient medians (so patients with many observations don't dominate).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PID = 'BDSPPatientID'


def _rho(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return spearmanr(x, y).statistic


def _boot(df, ce, outcomes, n_boot, seed):
    """Point rho and bootstrap samples for each outcome; same patient resample for all outcomes."""
    d = df[[PID, ce] + list(outcomes.values())].dropna()
    pts = d[PID].unique()
    idx = {p: np.flatnonzero(d[PID].values == p) for p in pts}
    vals = d[[ce] + list(outcomes.values())].to_numpy(float)
    point = {k: _rho(vals[:, 0], vals[:, j + 1]) for j, k in enumerate(outcomes)}
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in outcomes}
    for _ in range(n_boot):
        rows = np.concatenate([idx[p] for p in rng.choice(pts, size=len(pts), replace=True)])
        v = vals[rows]
        for j, k in enumerate(outcomes):
            r = _rho(v[:, 0], v[:, j + 1])
            if np.isfinite(r):
                samples[k].append(r)
    return point, {k: np.array(s) for k, s in samples.items()}, len(d), len(pts)


def _fmt(point, s):
    """0.31 (0.27–0.35); with a negative bound: −0.27 (−0.31 to −0.23). Unicode minus, as in the tables."""
    num = lambda x: f"{x:.2f}".replace("-", "−")
    if len(s) == 0:
        return f"{num(point)} (—)"
    lo, hi = np.percentile(s, [2.5, 97.5])
    sep = " to " if lo < 0 or hi < 0 else "–"
    return f"{num(point)} ({num(lo)}{sep}{num(hi)})"


def _miqr(x, dec):
    x = pd.Series(x).dropna()
    q = x.quantile([0.25, 0.5, 0.75]).round(dec)
    f = lambda v: f"{v:.{dec}f}".replace("-", "−")
    return f"{f(q[0.5])} ({f(q[0.25])} to {f(q[0.75])})"


def descriptives(df, ce, nesi, rass):
    """Median (IQR) per variable, across observations and across patient medians."""
    out = ['## Cohort descriptives (same rows)\n',
           '| Variable | Median (IQR), observations | Median (IQR) of patient medians |',
           '|---|---|---|']
    for name, col, dec in [('RASS', rass, 0), ('NESI', nesi, 2), ('Ce (µg/mL)', ce, 2)]:
        pm = df.groupby(PID)[col].median()
        out.append(f"| {name} | {_miqr(df[col], dec)} | {_miqr(pm, 1 if name == 'RASS' else dec)} |")
        print(f"  {name}: obs {_miqr(df[col], dec)}; patient medians {_miqr(pm, 1 if name == 'RASS' else dec)}")
    return out + ['']


def run_correlations(df, out_md, cohort_desc, ce='Ce_ug_per_mL',
                     nesi='PredictedNESI', rass='_RASS_numeric', n_boot=1000, seed=42):
    outcomes = {'NESI': nesi, 'RASS': rass}
    md = ['# Spearman correlation of propofol Ce with NESI and RASS\n',
          f'Cohort: {cohort_desc}. Same rows as the variance decomposition.\n',
          f'95% percentile CIs from a cluster bootstrap over patients ({n_boot:,} resamples; '
          'the same resample for NESI and RASS). Descriptive; no NESI-vs-RASS test.\n',
          '| Rows | Observations | Patients | ρ, Ce vs NESI (95% CI) | ρ, Ce vs RASS (95% CI) |',
          '|---|---|---|---|---|']
    for label, sub in [('All', df), ('Ce > 0', df[df[ce] > 0])]:
        point, s, n_obs, n_pts = _boot(sub, ce, outcomes, n_boot, seed)
        line = (f"| {label} | {n_obs:,} | {n_pts:,} | {_fmt(point['NESI'], s['NESI'])} "
                f"| {_fmt(point['RASS'], s['RASS'])} |")
        md.append(line)
        print(f"  {label}: {n_obs:,} obs, {n_pts:,} patients; "
              f"rho NESI {_fmt(point['NESI'], s['NESI'])}; rho RASS {_fmt(point['RASS'], s['RASS'])}")
    md.append('')
    md += descriptives(df, ce, nesi, rass)
    md.append('Signs: NESI rises and RASS falls with deeper sedation, so rho is expected to be '
              'positive for NESI and negative for RASS.')
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    tmp = out_md + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    os.replace(tmp, out_md)
    print(f"Saved → {out_md}")
