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
- Calibration plots (Supplementary Fig. 28E-G): the balanced class weights shift each model's logit
  by the log odds of death in its training data, so that shift is subtracted, using each repeat's own
  training set (no test outcomes), before held-out predictions are binned into deciles. The figure
  combines these panels with mortality.py's decision curves (panels A-D).
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


def training_log_odds(p, ids, died):
    """Log odds of death in each repeat's training set, as split by mortality.run_models."""
    from sklearn.model_selection import train_test_split

    rows = []
    for fold, seed in enumerate(M.SEEDS):
        tr_ids, _ = train_test_split(np.array(ids), test_size=0.15, random_state=seed, stratify=died)
        tr = p[p.pid.isin(set(tr_ids))]
        for h in HOURS:
            a = tr[tr.n >= h]
            for name, members in M.SETS.items():
                ok = a.dropna(subset=M.COVARIATES) if "cov" in members else a
                r = ok.died.mean()
                rows.append(dict(fold=fold, hour=h, model=name, train_log_odds=np.log(r / (1 - r))))
    return pd.DataFrame(rows)


def corrected(pr, odds):
    """Held-out probabilities with the balanced-class-weight shift removed."""
    c = pr[pr.hour.isin(HOURS)].merge(odds, on=["fold", "hour", "model"])
    c["prob_corrected"] = 1 / (1 + np.exp(-(logit(c.prob.values) + c.train_log_odds.values)))
    return c


def calibration_corrected(c):
    from sklearn.linear_model import LogisticRegression

    rows = []
    for (fold, h, name), g in c.groupby(["fold", "hour", "model"]):
        lp = logit(g.prob_corrected.values).reshape(-1, 1)
        fit = LogisticRegression(C=1e6, max_iter=1000).fit(lp, g.died.values)
        rows.append(dict(hour=h, model=name, intercept_corrected=fit.intercept_[0], mean_pred_corrected=g.prob_corrected.mean(),
                         brier_corrected=np.mean((g.prob_corrected.values - g.died.values) ** 2)))
    return pd.DataFrame(rows).groupby(["hour", "model"]).mean().reset_index()


SHOW = ["Clinical*", "GCS", "NESI", "Clinical* + GCS", "Clinical* + NESI", "Clinical* + GCS + NESI"]


def plot_supp_fig28(c):
    """Decision curves (A-D, from mortality.py's table) above calibration plots (E-G)."""
    import matplotlib.pyplot as plt

    dc = pd.read_csv(M.TABLES / "mortality_decision_curve.csv")
    fig = plt.figure(figsize=(9, 10.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.05], hspace=0.42, wspace=0.9)
    top = [fig.add_subplot(gs[r, k * 3:(k + 1) * 3]) for r in range(2) for k in range(2)]
    for ax, h, lab in zip(top, [8, 12, 16, 20], "ABCD"):
        d = dc[dc.hour == h]
        for name in SHOW:
            g = d[d.model == name]
            ax.plot(g.th, g["mean"], label=name, lw=1.4)
        g = d[d.model == "Treat all"]
        ax.plot(g.th, g["mean"], color="0.5", ls="--", lw=1, label="Treat all")
        ax.axhline(0, color="k", lw=0.8, label="Treat none")
        ax.set_title(f"{lab}. Decision curve, {h} hours", loc="left", fontsize=10)
        ax.set_ylim(-0.03, 0.22)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Threshold probability of in-hospital death", fontsize=8)
        ax.set_ylabel("Net benefit", fontsize=8)
    top[0].legend(frameon=False, fontsize=7)  # one legend: colors are the same in every panel
    bins = np.linspace(0, 1, 11)
    for k, (h, lab) in enumerate(zip(HOURS, "EFG")):
        ax = fig.add_subplot(gs[2, k * 2:(k + 1) * 2])
        ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls=":")
        for name in SHOW:
            g = c[(c.hour == h) & (c.model == name)]
            q = pd.qcut(g.prob_corrected, 10, duplicates="drop")
            b = g.groupby(q, observed=True).agg(pred=("prob_corrected", "mean"), obs=("died", "mean"))
            ax.plot(b.pred, b.obs, marker="o", ms=3, lw=1.2, label=name)
        ax.set_xlim(0, 0.85)
        ax.set_ylim(0, 0.85)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{lab}. Calibration, {h} hours", loc="left", fontsize=10)
        ax.set_xlabel("Predicted risk of death", fontsize=8)
        if k == 0:
            ax.set_ylabel("Observed proportion died", fontsize=8)
    fig.savefig(M.FIGS / "SuppFig28_decision_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    M.TABLES.mkdir(parents=True, exist_ok=True)
    M.FIGS.mkdir(parents=True, exist_ok=True)
    pr = pd.read_csv(M.INTER / "mortality_predictions.csv", dtype={"pid": str})
    p, ids, died = M.assemble("numeric")
    lr = lr_tests(p)
    lr.round(5).to_csv(M.TABLES / "mortality_lr.csv", index=False)
    print(lr.round(4).to_string(index=False))
    nr = nri_table(pr)
    nr.round(4).to_csv(M.TABLES / "mortality_nri.csv", index=False)
    print(nr.round(3).to_string(index=False))
    cal = calibration(pr)
    c = corrected(pr, training_log_odds(p, ids, died))
    cal = cal.merge(calibration_corrected(c), on=["hour", "model"])
    cal.round(4).to_csv(M.TABLES / "mortality_calibration.csv", index=False)
    print(cal.round(3).to_string(index=False))
    plot_supp_fig28(c)


if __name__ == "__main__":
    main()
