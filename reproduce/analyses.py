"""Recompute manuscript numbers from the released result files.

Each public function fills ``ctx.num`` with key -> value pairs and may write a
table to ``ctx.out``. ``run_all.py`` then compares those values with
``claims.csv``. Nothing here prints or writes patient-level rows.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, cohen_kappa_score, roc_auc_score, roc_curve

SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
SEVERE = {
    "RASS": lambda s: s <= -4,
    "GCS": lambda s: s <= 8,
    "CAMS": lambda s: s >= 6,
    "ICANS": lambda s: s >= 3,
}


class Context:
    def __init__(self, repo, root, out, boot, seed=0):
        self.repo, self.root, self.out = Path(repo), Path(root), Path(out)
        self.gh, self.s3, self.drive = self.root / "github_moved", self.root / "s3", self.root / "drive"
        self.boot, self.num = boot, {}
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def table(self, df, name):
        path = self.out / "tables" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def load(self, key, path, **kw):
        if key not in self._cache:
            self._cache[key] = pd.read_csv(path, low_memory=False, **kw)
        return self._cache[key]

    def scores(self):
        return self.load("scores", self.gh / "NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv")

    def model_sets(self):
        if "sets" not in self._cache:
            r = pd.concat(
                pd.read_csv(self.s3 / f"NESI_{n}_results.csv", low_memory=False) for n in ["train", "val", "test"]
            ).rename(columns={"Filename": "MorgothOutputFilename"})
            self._cache["sets"] = self.scores().merge(
                r[["Dataset", "MorgothOutputFilename", "WhichSet", "PredictedNESI"]],
                on=["Dataset", "MorgothOutputFilename"],
            )
        return self._cache["sets"]

    def meds(self):
        if "meds" not in self._cache:
            m = pd.read_csv(self.drive / "i0001_RASS_Meds_NESI.csv", low_memory=False)
            m["n"] = pd.to_numeric(m.NESI, errors="coerce")
            m["r"] = pd.to_numeric(m["R PHS IP RASS"], errors="coerce")
            self._cache["meds"] = m[m.n.notna() & m.r.between(-5, 0)].reset_index(drop=True)
        return self._cache["meds"]

    def meta(self, cohort):
        files = {
            "RASS": ["RASS_BWH", "RASS_MGH"],
            "GCS": ["GCS_BWH", "GCS_MGH"],
        }[cohort]
        out = []
        for f in files:
            d = pd.read_csv(self.gh / f"cohort_models/{cohort}/Cohort/{f}_HarvardEEG_metadata.csv", low_memory=False)
            d["Site"] = f.split("_")[1]
            out.append(d)
        return pd.concat(out)


def rho(a, b):
    return abs(stats.spearmanr(a, b)[0])


def _patient_boot(ctx, t, fn, n=None):
    groups = {p: g.index.values for p, g in t.groupby("BDSPPatientID")}
    ids = list(groups)
    vals = []
    for _ in range(n or ctx.boot):
        ix = np.concatenate([groups[ids[j]] for j in ctx.rng.integers(0, len(ids), len(ids))])
        vals.append(fn(t.loc[ix]))
    return np.array(vals)


def _segment_boot(ctx, a, b):
    v = [rho(a[i], b[i]) for i in (ctx.rng.integers(0, len(a), len(a)) for _ in range(ctx.boot))]
    return np.mean(v), *np.percentile(v, [2.5, 97.5])


def cohorts_and_splits(ctx):
    f, n = ctx.scores(), ctx.num
    for ds in SCALES:
        t = f[f.Dataset == ds]
        n[f"{ds}.patients"] = t.BDSPPatientID.nunique()
        n[f"{ds}.obs"] = len(t)
        n[f"{ds}.test_obs"] = (t.Split == "Test").sum()
        n[f"{ds}.test_patients"] = t[t.Split == "Test"].BDSPPatientID.nunique()
    # ICANS study IDs mapped to BDSP IDs where known, so patients in several cohorts count once
    idx = pd.read_csv(ctx.s3 / "segment_index.csv", low_memory=False, dtype={"subject_id": str, "resolved_BDSPPatientID": str})
    ic = idx[idx.cohort == "ICANS"].dropna(subset=["resolved_BDSPPatientID"]).drop_duplicates("subject_id")
    ic_map = dict(zip(ic.subject_id, ic.resolved_BDSPPatientID.str.replace(r"\.0$", "", regex=True)))
    pid = f.BDSPPatientID.astype(str).where(f.Dataset != "ICANS", f.BDSPPatientID.astype(str).map(lambda x: ic_map.get(x, "ICANS_" + x)))
    n["all.unique_patients"] = pid.nunique()
    n["all.test_patients"] = f[f.Split == "Test"].BDSPPatientID.nunique()
    n["all.unique_segments"] = f.MorgothOutputFilename.nunique()
    n["all.segments_multi_cohort"] = (f.groupby("MorgothOutputFilename").Dataset.nunique() > 1).sum()
    pts = {k: set(v.BDSPPatientID) for k, v in f.groupby("Dataset")}
    n["overlap.RASS_GCS"] = len(pts["RASS"] & pts["GCS"])
    n["split.patients_multi_repo_labels"] = (f.groupby("BDSPPatientID").Split.nunique() > 1).sum()
    m = ctx.model_sets()
    # IDs harmonized as above. Within a cohort no patient crosses sets; across cohorts they do.
    m_pid = m.merge(pd.DataFrame({"Dataset": f.Dataset, "MorgothOutputFilename": f.MorgothOutputFilename, "pid": pid}), on=["Dataset", "MorgothOutputFilename"], how="left")
    n["split.patients_multi_model_sets"] = (m_pid.groupby("pid").WhichSet.nunique() > 1).sum()
    n["split.patients_multi_within_cohort"] = (m_pid.groupby(["Dataset", "pid"]).WhichSet.nunique() > 1).sum()
    te = m_pid[m_pid.MorgothOutputFilename.isin(f[f.Split == "Test"].MorgothOutputFilename) & (m_pid.WhichSet == "Test")]
    tv = m_pid[m_pid.WhichSet != "Test"]
    n["split.test_patients"] = te.pid.nunique()
    n["split.test_patients_with_train_val"] = len(set(te.pid) & set(tv.pid))
    n["split.train_val_rows_of_test_patients"] = tv.pid.isin(set(te.pid)).sum()
    for ds in ["CAMS", "ICANS"]:
        n[f"split.{ds}.test_patients_with_train_val"] = len(set(te[te.Dataset == ds].pid) & set(tv.pid))
    n["split.segments_multi_model_sets"] = (m.groupby("MorgothOutputFilename").WhichSet.nunique() > 1).sum()
    n["split.model_test_rows"] = (m.WhichSet == "Test").sum()


def correlations(ctx):
    f, n, rows = ctx.scores(), ctx.num, []
    for ds in SCALES:
        b = pd.read_csv(ctx.gh / f"NESI/Bespoke_models/Results/{ds}Bespokescore_results.csv")
        t = f[(f.Dataset == ds) & (f.Split == "Test")].merge(
            b[["MorgothOutputFilename", "BespokeBadnessScore"]], on="MorgothOutputFilename"
        ).reset_index(drop=True)
        a, u, c = t.RawScore.values, t.NESI.values, t.BespokeBadnessScore.values
        su, sc = _segment_boot(ctx, a, u), _segment_boot(ctx, a, c)
        B = _patient_boot(ctx, t, lambda x: (rho(x.RawScore, x.NESI), rho(x.RawScore, x.BespokeBadnessScore)))
        d = B[:, 0] - B[:, 1]
        n[f"{ds}.rho"], n[f"{ds}.rho_mean_boot"] = rho(a, u), su[0]
        n[f"{ds}.rho_seg_lo"], n[f"{ds}.rho_seg_hi"] = su[1], su[2]
        n[f"{ds}.rho_pt_lo"], n[f"{ds}.rho_pt_hi"] = np.percentile(B[:, 0], [2.5, 97.5])
        n[f"{ds}.bespoke"] = rho(a, c)
        n[f"{ds}.bespoke_seg_lo"], n[f"{ds}.bespoke_seg_hi"] = sc[1], sc[2]
        n[f"{ds}.delta"] = rho(a, u) - rho(a, c)
        n[f"{ds}.delta_lo"], n[f"{ds}.delta_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append({k.split(".", 1)[1]: round(v, 3) for k, v in n.items() if k.startswith(ds + ".")} | {"Scale": ds})
    ctx.table(pd.DataFrame(rows), "fig2_correlations")


def cams_sensitivity(ctx):
    """CAM-S test patients who also have GCS/RASS data in training or validation."""
    m, n = ctx.model_sets(), ctx.num
    b = pd.read_csv(ctx.gh / "NESI/Bespoke_models/Results/CAMSBespokescore_results.csv")
    c = m[(m.Dataset == "CAMS") & (m.Split == "Test")].merge(
        b[["MorgothOutputFilename", "BespokeBadnessScore"]], on="MorgothOutputFilename"
    ).reset_index(drop=True)
    seen = set(m[m.WhichSet.isin(["Train", "Validation"]) & (m.Dataset != "CAMS")].BDSPPatientID)
    clean = c[~c.BDSPPatientID.isin(seen)].reset_index(drop=True)
    n["CAMS.leak_patients"] = len(c) - len(clean)
    n["CAMS.clean_rho"] = rho(clean.RawScore, clean.NESI)
    n["CAMS.clean_bespoke"] = rho(clean.RawScore, clean.BespokeBadnessScore)
    B = _patient_boot(ctx, clean, lambda x: (rho(x.RawScore, x.NESI), rho(x.RawScore, x.BespokeBadnessScore)), n=4000)
    n["CAMS.clean_rho_lo"], n["CAMS.clean_rho_hi"] = np.percentile(B[:, 0], [2.5, 97.5])
    n["CAMS.clean_delta"] = n["CAMS.clean_rho"] - n["CAMS.clean_bespoke"]
    n["CAMS.clean_delta_lo"], n["CAMS.clean_delta_hi"] = np.percentile(B[:, 0] - B[:, 1], [2.5, 97.5])


def supp_table1(ctx):
    f, rows = ctx.scores(), []
    for ds in SCALES:
        for lvl, v in f[(f.Dataset == ds) & (f.Split == "Test")].groupby("RawScore").NESI:
            q = (v.median(), v.quantile(0.25), v.quantile(0.75))
            rows.append(dict(Scale=ds, Score=int(lvl), Median=q[0], Q1=q[1], Q3=q[2], N=len(v)))
            for name, val in zip(["med", "q1", "q3"], q):
                ctx.num[f"T1.{ds}.{int(lvl)}.{name}"] = val
    ctx.table(pd.DataFrame(rows).round(3), "supp_table1")


def supp_tables_2_3(ctx):
    f, rows = ctx.scores(), []
    for ds, sev in SEVERE.items():
        t = f[f.Dataset == ds]
        tr, te = t[t.Split != "Test"], t[t.Split == "Test"]
        y, yt = sev(tr.RawScore).astype(int), sev(te.RawScore).astype(int)
        p = LogisticRegression(C=1e6).fit(tr[["NESI"]], y).predict_proba(te[["NESI"]])[:, 1]
        fp, tp, th = roc_curve(y, tr.NESI)
        cut = th[np.argmax(tp - fp)]
        hit = te.NESI >= cut
        res = dict(
            cut=cut,
            sens=(hit & (yt == 1)).sum() / yt.sum(),
            spec=((~hit) & (yt == 0)).sum() / (yt == 0).sum(),
            ppv=(hit & (yt == 1)).sum() / hit.sum(),
            npv=((~hit) & (yt == 0)).sum() / (~hit).sum(),
            auc=roc_auc_score(yt, te.NESI),
            brier=brier_score_loss(yt, p),
        )
        for k, v in res.items():
            ctx.num[f"T3.{ds}.{k}"] = v
        rows.append(dict(Scale=ds, **res))
    ctx.table(pd.DataFrame(rows).round(3), "supp_tables_2_3")


def threshold_aurocs(ctx):
    f = ctx.scores()
    for ds in SCALES:
        te = f[(f.Dataset == ds) & (f.Split == "Test")]
        vals = []
        for k in np.sort(te.RawScore.unique())[:-1]:
            y = (te.RawScore <= k) if ds in ("RASS", "GCS") else (te.RawScore > k)
            vals.append(roc_auc_score(y, te.NESI))
        ctx.num[f"AUC.{ds}.min"], ctx.num[f"AUC.{ds}.max"] = min(vals), max(vals)


def ordinal_agreement(ctx):
    """Cutoffs are midpoints of the median NESI of adjacent classes in training and validation data."""
    f = ctx.scores()
    for ds in SCALES:
        t = f[f.Dataset == ds]
        te, tr = t[t.Split == "Test"], t[t.Split != "Test"]
        g = np.sort(tr.GroupedScore.unique())
        med = [tr[tr.GroupedScore == x].NESI.median() for x in g]
        cuts = sorted((med[i] + med[i + 1]) / 2 for i in range(len(g) - 1))
        truth = te.GroupedScore.rank(method="dense").astype(int) - 1
        if med[0] > med[-1]:
            truth = truth.max() - truth
        pred = np.digitize(te.NESI, cuts)
        ctx.num[f"ORD.{ds}.maoe"] = np.abs(truth - pred).mean()
        ctx.num[f"ORD.{ds}.qwk"] = cohen_kappa_score(truth, pred, weights="quadratic")


def institutions_age_duration(ctx):
    f, rows = ctx.scores(), []
    for ds in ["RASS", "GCS"]:
        md = ctx.meta(ds)
        x = md.drop_duplicates(["BDSPPatientID", "SessionID"]).DurationInSeconds / 3600
        ctx.num[f"DUR.{ds}.median_h"] = x.median()
        ctx.num[f"DUR.{ds}.pct_le1h"] = (x <= 1).mean() * 100
        one = md.drop_duplicates("BDSPPatientID").set_index("BDSPPatientID")
        t = f[(f.Dataset == ds) & (f.Split == "Test")].copy()
        t["Site"] = t.BDSPPatientID.map(one.Site).fillna("BIDMC")
        t["Age"] = t.BDSPPatientID.map(one.AgeAtVisit)
        for s, v in t.groupby("Site"):
            ctx.num[f"SITE.{ds}.{s}.rho"] = rho(v.RawScore, v.NESI)
            rows.append(dict(Scale=ds, Group=s, Patients=v.BDSPPatientID.nunique(), Segments=len(v), rho=rho(v.RawScore, v.NESI)))
        for nm, v in [("lt65", t[t.Age < 65]), ("ge65", t[t.Age >= 65])]:
            ctx.num[f"AGE.{ds}.{nm}.rho"] = rho(v.RawScore, v.NESI)
            rows.append(dict(Scale=ds, Group=nm, Patients=v.BDSPPatientID.nunique(), Segments=len(v), rho=rho(v.RawScore, v.NESI)))
    ctx.table(pd.DataFrame(rows).round(3), "supp_table12_site_age")


def table1(ctx):
    n = ctx.num
    rm = ctx.meta("RASS")
    num = pd.to_numeric(rm["R PHS IP RASS"], errors="coerce")
    for s in [1, 2, 3, 4]:
        n[f"RASSpos.{s}"] = (num == s).sum()
    n["RASS.raw_patients"] = rm.BDSPPatientID.nunique()
    p = rm.sort_values("EEGBeginDTS").groupby("BDSPPatientID").first()
    n["T1.RASS.male"], n["T1.RASS.female"] = (p.SexDSC == "Male").sum(), (p.SexDSC == "Female").sum()
    n["T1.RASS.age_med"] = p.AgeAtVisit.median()
    for ds, d in [("RASS", rm), ("GCS", ctx.meta("GCS"))]:
        a, e = pd.to_datetime(d.HospitalAdmitDTS, errors="coerce").dt.normalize(), pd.to_datetime(d.HospitalDischargeDTS, errors="coerce").dt.normalize()
        x = pd.to_datetime(d.DateOfDeath, errors="coerce").dt.normalize()
        g = d.assign(i=(x >= a) & (x <= e), h=x.notna()).groupby("BDSPPatientID").agg(i=("i", "max"), h=("h", "max"))
        if ds == "GCS":
            # BIDMC has placeholders in death_date: parse dates rather than test for non-null
            b = pd.read_csv(ctx.s3 / "GCS_BIDMC_HarvardEEG_metadata.csv").groupby("BDSPPatientID").first()
            bx = pd.to_datetime(b.death_date, errors="coerce").dt.normalize()
            ba, be = pd.to_datetime(b.t0_admit, errors="coerce").dt.normalize(), pd.to_datetime(b.t1_discharge, errors="coerce").dt.normalize()
            g = pd.concat([g, pd.DataFrame({"i": (bx >= ba) & (bx <= be), "h": bx.notna()})])
            n["GCS.raw_patients"] = len(g)
            n["T1.GCS.placeholder_deaths"] = b.death_date.notna().sum() - bx.notna().sum()
        n[f"T1.{ds}.inhosp"] = g.i.sum()
        n[f"T1.{ds}.other_death"] = (g.h & ~g.i).sum()
        n[f"T1.{ds}.no_death"] = (~g.h).sum()
    c = pd.read_csv(ctx.s3 / "CAMS_cohort_metadata.csv").drop_duplicates("BDSPPatientID")
    n["CAMS.raw_patients"] = len(c)
    n["T1.CAMS.age_med"] = pd.to_numeric(c["Age "], errors="coerce").median()
    n["T1.CAMS.male"], n["T1.CAMS.female"] = (c.Gender == "Male").sum(), (c.Gender == "Female").sum()
    n["T1.CAMS.any_death"] = c["DateOfDeath (shifted)"].notna().sum()


def mortality_folds(ctx):
    import pickle

    p = ctx.repo / "NESI/DeathPrediction_NESIvsGCS/Results"
    a = {k: pickle.load(open(p / f"results_{k}_Death.pkl", "rb")) for k in ["NESI", "GCS"]}
    for h in [4, 20]:
        u = np.array([x[h]["auroc"] for x in a["NESI"]])
        g = np.array([x[h]["auroc"] for x in a["GCS"]])
        ctx.num[f"MORT.NESI.{h}h"], ctx.num[f"MORT.GCS.{h}h"] = u.mean(), g.mean()
        if h == 20:
            ctx.num["MORT.p_paired_t_one_sided"] = stats.ttest_rel(u, g)[1] / 2


def retest(ctx):
    n, t = ctx.num, ctx.meds().copy()
    flag = lambda c: t[c].fillna(0).astype(float) > 0
    t["t"] = pd.to_datetime(t.RASSRecordedDTS, errors="coerce")
    t["state"] = flag("sedative_inf_now").astype(int) * 2 + flag("opioid_inf_now").astype(int)
    t = t.sort_values(["BDSPPatientID", "SessionID", "t"])
    g = t.groupby(["BDSPPatientID", "SessionID"])
    t["dt"], t["pr"], t["ps"], t["pn"] = g.t.diff().dt.total_seconds() / 3600, g.r.shift(), g.state.shift(), g.n.shift()
    p = t[(t.dt > 0) & (t.dt <= 2) & (t.r == t.pr) & (t.state == t.ps)]
    sem = np.sqrt(((p.n - p.pn) ** 2).mean() / 2)
    n["RETEST.pairs"], n["RETEST.patients"], n["RETEST.sem"] = len(p), p.BDSPPatientID.nunique(), sem
    n["RETEST.mdc"] = 1.96 * np.sqrt(2) * sem
    n["RETEST.icc"] = 1 - sem**2 / np.concatenate([p.n, p.pn]).var(ddof=1)


def supp_table11(ctx):
    """NESI-RASS performance by sedative group. Groups are Christine Eckhardt's per-assessment labels
    (build_strata_labels.py: infusion running or stopped within 6 h; each sedative pooled with or
    without an opioid infusion), joined on patient and RASS time."""
    d, n = ctx.meds().copy(), ctx.num
    lab = pd.read_csv(ctx.root / "sedation_repo/medication-exposure/data/stratified_feasibility/labels/strata_labels_rass.csv")
    lab["t"] = pd.to_datetime(lab.assessment_time)
    d["t"] = pd.to_datetime(d.RASSRecordedDTS, errors="coerce")
    d = d.merge(lab.drop(columns=["assessment_time", "score"]), on=["BDSPPatientID", "t"], how="left", validate="many_to_one")
    b = lambda c: d[c].fillna(0).astype(float) > 0
    bz, asm = b("FastActing_Benzo_Exposure"), b("LongActingASM_Exposure")
    none24 = ~b("sedative_inf_24h") & ~b("opioid_inf_24h") & ~bz
    G = {
        "Reference": b("Reference (no sedative or opioid infusion)"),
        "Propofol": b("Propofol (+/- opioid)"),
        "Dexmedetomidine": b("Dexmedetomidine (+/- opioid)"),
        "Midazolam": b("Midazolam (+/- opioid)"),
        "MidazolamPropofol": b("Midazolam + propofol (+/- opioid)"),
        "Ketamine": b("Ketamine (+/- opioid)"),
        "OpioidOnly": b("Opioid only"),
        "All": pd.Series(True, index=d.index),
        "NoInfusion24h": none24,
        "NoInfusion24hNoASM": none24 & ~asm,
    }
    metrics = {"rho": lambda x: rho(x.r, x.n), "auc3": lambda x: roc_auc_score(x.r <= -3, x.n), "auc4": lambda x: roc_auc_score(x.r <= -4, x.n)}
    rows = []
    for k, v in G.items():
        x = d[v].reset_index(drop=True)
        res = dict(obs=len(x), patients=x.BDSPPatientID.nunique())
        for name, fn in metrics.items():
            res[name] = fn(x)
            res[name + "_lo"], res[name + "_hi"] = np.percentile(_patient_boot(ctx, x, fn), [2.5, 97.5])
        for name, val in res.items():
            n[f"T11.{k}.{name}"] = val
        rows.append(dict(Group=k, **res))
    ctx.table(pd.DataFrame(rows).round(3), "supp_table11")


def propofol(ctx):
    import statsmodels.formula.api as smf

    n = ctx.num
    # Corrected Eleveld V2 age coefficient (Christine Eckhardt's revision file)
    w = pd.read_csv(ctx.root / "sedation_repo/eleveld-pk-analysis/data/wt_ht_per_hosp_FULL_with_eleveld_expanded.csv", low_memory=False)
    w["NE"], w["r"] = pd.to_numeric(w.PredictedNESI, errors="coerce"), pd.to_numeric(w["R PHS IP RASS"], errors="coerce")
    q = w[(w.propofol_only_recomputed == True) & (w.PropofolSimStatus == "simulated") & w.Ce_ug_per_mL.notna() & w.NE.notna() & w.r.between(-5, 0)].rename(columns={"Ce_ug_per_mL": "CE"})  # noqa: E712
    n["PROP.obs"], n["PROP.patients"] = len(q), q.BDSPPatientID.nunique()
    Q = q[q.groupby("BDSPPatientID").NE.transform("size") >= 5].copy()
    n["VAR.obs"], n["VAR.patients"] = len(Q), Q.BDSPPatientID.nunique()
    # Spearman rho on the same rows as the variance decomposition (rank_correlation.py, Nov 3)
    n["PROP.rho_nesi"], n["PROP.rho_rass"] = stats.spearmanr(Q.CE, Q.NE)[0], stats.spearmanr(Q.CE, Q.r)[0]
    Q["m"], Q["d"] = Q.groupby("BDSPPatientID").CE.transform("mean"), Q.CE - Q.groupby("BDSPPatientID").CE.transform("mean")
    for y, tag in [("NE", "nesi"), ("r", "rass")]:
        fit = smf.mixedlm(f"{y}~CE", Q, groups=Q.BDSPPatientID).fit(reml=True)
        vf, vr, ve = np.var(fit.fe_params.CE * Q.CE), float(fit.cov_re.iloc[0, 0]), fit.scale
        tot = vf + vr + ve
        n[f"VAR.{tag}.pct_ce"], n[f"VAR.{tag}.pct_patient"], n[f"VAR.{tag}.pct_resid"] = 100 * vf / tot, 100 * vr / tot, 100 * ve / tot
        n[f"VAR.{tag}.slope"] = fit.fe_params.CE
        wb = smf.mixedlm(f"{y}~d+m", Q, groups=Q.BDSPPatientID).fit(reml=True)
        n[f"WB.{tag}.within"], n[f"WB.{tag}.between"] = wb.fe_params.d, wb.fe_params.m
        per = Q.groupby("BDSPPatientID")[["CE", y]].apply(lambda g: stats.spearmanr(g.CE, g[y])[0] if g.CE.nunique() > 1 and g[y].nunique() > 1 else np.nan).dropna()
        n[f"WB.{tag}.median_rho"], n[f"WB.{tag}.n"] = per.median(), len(per)


def sedation_strata(ctx):
    """Point estimates from Christine Eckhardt's pipeline, read from its output when present."""
    f = ctx.root / "sedation_repo/medication-exposure/data/stratified_feasibility/model/results_main_stratum_full.csv"
    d = pd.read_csv(f)
    for r in d.itertuples():
        k = re.sub(r"[^A-Za-z]+", "", r.agent.split("(")[0])
        ctx.num[f"T5.{k}.obs"], ctx.num[f"T5.{k}.patients"], ctx.num[f"T5.{k}.offset"] = r.n_obs, r.n_patients, r.add_offset


def revision_tables(ctx):
    """Values from reproduce/by_diagnosis.py and reproduce/rass_positive.py, when run."""
    t = ctx.out / "tables"
    d = pd.read_csv(t / "performance_by_diagnosis.csv").dropna(subset=["rho"])
    for ds in ["RASS", "GCS"]:
        v = d[d.Scale == ds]
        ctx.num[f"DX.{ds}.min"], ctx.num[f"DX.{ds}.max"] = v.rho.min(), v.rho.max()
    r = pd.read_csv(t / "rass_positive_by_level.csv").set_index("RASS").Median
    for lvl in [0, 1, 2]:
        ctx.num[f"RASSPOS.median.{lvl}"] = r.loc[lvl]
    a = pd.read_csv(t / "ablation_summary.csv")
    a = a[a.Seed == 2026]
    for x in a.itertuples():
        ctx.num[f"ABL.{x.Head}.{x.Scale}.rho"], ctx.num[f"ABL.{x.Head}.{x.Scale}.delta"] = x.rho, x.delta
    ctx.num.update(json.load(open(ctx.root / "intermediates/external/numbers.json")))
    m = pd.read_csv(t / "mortality_summary.csv", header=None, index_col=0)[1]
    for k, v in m.items():
        ctx.num[f"MORT2.{k}"] = v
    d = pd.read_csv(t / "mortality_delong.csv")
    for x in d.itertuples():
        k = f"MORT2.delta.{x.a}|{x.b}|{x.hour}"
        ctx.num[k], ctx.num[k + ".lo"], ctx.num[k + ".hi"], ctx.num[k + ".p_holm"] = x.delta, x.lo, x.hi, x.p_holm
    # likelihood-ratio tests, NRI and calibration (mortality_extra.py)
    for x in pd.read_csv(t / "mortality_lr.csv").itertuples():
        ctx.num[f"MORT3.lr.{x.full}|{x.reduced}|{x.hour}.p"] = x.p
    for x in pd.read_csv(t / "mortality_nri.csv").itertuples():
        k = f"MORT3.nri.{x.new}|{x.old}|{x.hour}"
        ctx.num[k], ctx.num[k + ".lo"], ctx.num[k + ".hi"] = x.nri, x.lo, x.hi
    cal = pd.read_csv(t / "mortality_calibration.csv")
    ctx.num["MORT3.cal_slope_min"], ctx.num["MORT3.cal_slope_max"] = cal.slope.min(), cal.slope.max()
    ctx.num["MORT3.cal_int_corrected_min"], ctx.num["MORT3.cal_int_corrected_max"] = cal.intercept_corrected.min(), cal.intercept_corrected.max()
    # Supplementary Table 19: Eleveld implementation check (sedation.py)
    e = pd.read_csv(t / "eleveld_vs_pas_agreement.csv")
    for _, x in e.iterrows():
        for k in ["MDPE_%", "MDAPE_%", "P95_abs_%", "max_abs_%", "max_abs_ugml", "n"]:
            ctx.num[f"ELEV.{x['subset']}.{k}"] = float(x["ce_" + k])
    # Supplementary Fig. 30: qEEG-trained heads (qeeg_heads.py)
    for x in pd.read_csv(t / "qeeg_heads.csv").itertuples():
        k = f"QHEAD.{x.Dataset}.{x.Feature}"
        ctx.num[k], ctx.num[k + ".lo"], ctx.num[k + ".hi"] = x.Spearman, x.CI_Lower, x.CI_Upper
    # Supplementary Table 20, the rows computed from the analyzed recordings
    r = pd.read_csv(ctx.root / "intermediates/external/recordings.csv")
    ctx.num["EXT20.patients"] = len(r)
    for c in ["gcs", "rass"]:
        ctx.num[f"EXT20.{c}.median"], ctx.num[f"EXT20.{c}.q1"], ctx.num[f"EXT20.{c}.q3"] = r[c].median(), r[c].quantile(0.25), r[c].quantile(0.75)
    for lab, v in {"positive": "Positive", "negative": "Negative", "uta": "UTA, RASS score -4 or -5"}.items():
        ctx.num[f"EXT20.cam.{lab}"] = int((r.cam == v).sum())
    m = r.duration_s / 60
    ctx.num.update({"EXT20.minutes.median": m.median(), "EXT20.minutes.q1": m.quantile(0.25), "EXT20.minutes.q3": m.quantile(0.75),
                    "EXT20.minutes.min": m.min(), "EXT20.minutes.max": m.max()})
    # and the rows for all enrolled patients, from Northwestern's date-free per-recording file (ages over 89 coded 90)
    d = pd.read_csv(ctx.root / "intermediates/external/demographics.csv")
    ctx.num["EXT20.enrolled"] = len(d)
    ctx.num["EXT20.age.median"], ctx.num["EXT20.age.q1"], ctx.num["EXT20.age.q3"] = d.Age.median(), d.Age.quantile(0.25), d.Age.quantile(0.75)
    for c in ["Sex", "Race", "Ethnicity", "AdmissionDiagnosis_categories", "EEG_indication"]:
        for v, k in d[c].value_counts().items():
            ctx.num[f"EXT20.{c}.{v}"] = int(k)
    ctx.num["EXT20.death.hospital"] = int(d.Death_during_hospitalization.sum())
    ctx.num["EXT20.death.1yr"] = int(d.Death_within_1yr_postdischarge.sum())
    # Methods: sampling rate of every Northwestern EDF (Northwestern's channel check) and the RASS range analyzed
    f = pd.read_csv(ctx.root / "intermediates/external/eeg_sampling.csv")
    ctx.num["EXT.edf.n"] = len(f)
    for hz, k in f.Sampling_frequency_Hz.value_counts().items():
        ctx.num[f"EXT.edf.fs.{hz}"] = int(k)
    ctx.num["EXT.rass.max"] = r.rass.max()
    a = pd.read_csv(t / "artifact_tolerance.csv")
    for x in a.itertuples():
        ctx.num[f"ART.{x.Scale}.{x.Artifact}.rho"], ctx.num[f"ART.{x.Scale}.{x.Artifact}.lo"], ctx.num[f"ART.{x.Scale}.{x.Artifact}.hi"] = x.rho, x.lo, x.hi
    r = pd.read_csv(t / "retrain_summary.csv")
    for x in r.itertuples():
        ctx.num[f"RETRAIN.{x.Run}.{x.Scale}.rho"], ctx.num[f"RETRAIN.{x.Run}.{x.Scale}.lo"], ctx.num[f"RETRAIN.{x.Run}.{x.Scale}.hi"] = x.rho_new, x.lo, x.hi
    q = pd.read_csv(t / "qeeg_supp_fig31.csv")
    for x in q.itertuples():
        ctx.num[f"QEEG.{x.Scale}.{x.Feature}.rho"], ctx.num[f"QEEG.{x.Scale}.{x.Feature}.lo"], ctx.num[f"QEEG.{x.Scale}.{x.Feature}.hi"] = x.rho, x.lo, x.hi
    lo = pd.read_csv(t / "loco_summary.csv")
    for x in lo.itertuples():
        for c in ["full_rho", "full_lo", "full_hi", "loco_rho", "loco_lo", "loco_hi", "uni_rho", "uni_lo", "uni_hi", "delta", "delta_lo", "delta_hi"]:
            ctx.num[f"LOCO.{x.Scale}.{c}"] = getattr(x, c)
    mp = pd.read_csv(t / "mapping_summary.csv")
    for x in mp.itertuples():
        k = f"MAP.{x.Mapping.replace(' ', '_')}.{x.Scale}"
        ctx.num[k + ".rho"], ctx.num[k + ".lo"], ctx.num[k + ".hi"], ctx.num[k + ".delta"], ctx.num[k + ".p_holm"] = x.rho, x.lo, x.hi, x.delta, x.p_holm
    ia = pd.read_csv(t / "icans_segment_ablation.csv")
    for x in ia.itertuples():
        ctx.num[f"ICSEG.{x.Set}.rho"], ctx.num[f"ICSEG.{x.Set}.lo"], ctx.num[f"ICSEG.{x.Set}.hi"] = x.rho, x.lo, x.hi
    sub = pd.read_csv(t / "icans_subgroups.csv")
    for i, x in enumerate(sub.itertuples()):
        ctx.num[f"ICSUB.{i}.rho"], ctx.num[f"ICSUB.{i}.lo"], ctx.num[f"ICSUB.{i}.hi"] = x.rho, x.lo, x.hi
    ind = t.parent / "indication"
    short = lambda s: s.split(" / ")[0].replace(" ", "_")
    for x in pd.read_csv(ind / "NESI_Indication_Spearman_Train_Test_Bootstrap1000.csv").itertuples():
        k = f"IND.{x.Split}.{short(x.Indication)}.{x.Scale}"
        ctx.num[k + ".rho"], ctx.num[k + ".lo"], ctx.num[k + ".hi"] = x.Rho, x.CI_Lower, x.CI_Upper
    for x in pd.read_csv(ind / "NESI_Indication_UniquePatientCounts_Train_Test_All.csv").itertuples():
        ctx.num[f"INDN.{short(x.Indication)}.{x.Scale}"] = x._5
    for x in pd.read_csv(t / "supp_fig10_mae_by_dx.csv").itertuples():
        ctx.num[f"SF10.{x.Dx.split('/')[0]}.median"] = x.median
    f6 = pd.read_csv(t / "figure6_counts.csv")
    for x in f6.itertuples():
        ctx.num["FIG6." + x.Panel.replace(" ", "_")] = x.Selected


ALL = [
    cohorts_and_splits,
    correlations,
    cams_sensitivity,
    supp_table1,
    supp_tables_2_3,
    threshold_aurocs,
    ordinal_agreement,
    institutions_age_duration,
    table1,
    mortality_folds,
    retest,
    supp_table11,
    propofol,
    sedation_strata,
    revision_tables,
]
