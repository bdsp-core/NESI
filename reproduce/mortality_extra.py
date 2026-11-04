#!/usr/bin/env python3
"""Mortality model comparisons requested in review (R3.8): likelihood-ratio tests, net
reclassification and calibration, alongside the AUROC and DeLong results of mortality.py.

    python reproduce/mortality_extra.py     # minutes, from intermediates/mortality (run mortality.py analyze first)

- Likelihood-ratio tests: unpenalized logistic regressions fitted on all patients with the
  covariates, at 4, 12 and 20 hours, nested models with and without the NESI sequence (or GCS).
  The paper's models are elastic-net fits, which have no likelihood-ratio test of their own.
- Continuous (category-free) NRI on the held-out predictions of mortality.py, per repeat, with a
  patient bootstrap per repeat; estimates and standard errors averaged over the five repeats.
- Calibration of the held-out predictions: logistic calibration slope and intercept and the Brier
  score. The models use balanced class weights, so predicted probabilities are not absolute risks;
  the intercept and Brier score reflect that, and the slope is the informative quantity.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import mortality as M

HOURS = [4, 12, 20]
PAIRS = [("Clinical* + GCS + NESI", "Clinical* + GCS"), ("Clinical* + NESI", "Clinical*"),
         ("Clinical* + GCS", "Clinical*"), ("Clinical* + GCS + NESI", "Clinical* + NESI")]
rng = np.random.default_rng(0)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def lr_tests(p):
    import statsmodels.api as sm
    rows = []
    for h in HOURS:
        d = p[(p.n >= h)].dropna(subset=M.COVARIATES)
        y = d.died.values
        for a, b in PAIRS:
            fits = {}
            for name in (a, b):
                x = M.design(d, h, M.SETS[name])
                x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fits[name] = sm.Logit(y, sm.add_constant(x)).fit(disp=0, method="lbfgs", maxiter=2000)
            lr = 2 * (fits[a].llf - fits[b].llf)
            df = fits[a].df_model - fits[b].df_model
            rows.append(dict(hour=h, full=a, reduced=b, n=len(d), deaths=int(y.sum()), lr=lr, df=df, p=stats.chi2.sf(lr, df)))
    return pd.DataFrame(rows)


def nri(y, new, old):
    up, down = new > old, new < old
    ev, ne = y == 1, y == 0
    return (up[ev].mean() - down[ev].mean()) + (down[ne].mean() - up[ne].mean())


def nri_table(pr):
    rows = []
    for a, b in PAIRS:
        for h in HOURS:
            ests, ses = [], []
            for fold in sorted(pr.fold.unique()):
                g = pr[(pr.fold == fold) & (pr.hour == h)]
                w = g[g.model == a].merge(g[g.model == b], on=["pid", "died"], suffixes=("_a", "_b"))
                y, pa, pb = w.died.values, w.prob_a.values, w.prob_b.values
                ests.append(nri(y, pa, pb))
                bs = [nri(y[i], pa[i], pb[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
                ses.append(np.std(bs, ddof=1))
            e, se = np.mean(ests), np.mean(ses)
            rows.append(dict(new=a, old=b, hour=h, nri=e, lo=e - 1.96 * se, hi=e + 1.96 * se))
    return pd.DataFrame(rows)


def calibration(pr):
    from sklearn.linear_model import LogisticRegression
    rows = []
    for (fold, h, name), g in pr[pr.hour.isin(HOURS)].groupby(["fold", "hour", "model"]):
        lp = logit(g.prob.values).reshape(-1, 1)
        fit = LogisticRegression(C=1e6, max_iter=1000).fit(lp, g.died.values)
        rows.append(dict(fold=fold, hour=h, model=name, slope=fit.coef_[0, 0], intercept=fit.intercept_[0],
                         brier=np.mean((g.prob.values - g.died.values) ** 2), event_rate=g.died.mean(), mean_pred=g.prob.mean()))
    c = pd.DataFrame(rows)
    return c.groupby(["hour", "model"])[["slope", "intercept", "brier", "event_rate", "mean_pred"]].mean().reset_index()


def main():
    M.TABLES.mkdir(parents=True, exist_ok=True)
    pr = pd.read_csv(M.INTER / "mortality_predictions.csv", dtype={"pid": str})
    p, ids, died = M.assemble("numeric")
    lr = lr_tests(p)
    lr.round(5).to_csv(M.TABLES / "mortality_lr.csv", index=False)
    print(lr.round(4).to_string(index=False))
    nr = nri_table(pr)
    nr.round(4).to_csv(M.TABLES / "mortality_nri.csv", index=False)
    print(nr.round(3).to_string(index=False))
    cal = calibration(pr)
    cal.round(4).to_csv(M.TABLES / "mortality_calibration.csv", index=False)
    print(cal.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
